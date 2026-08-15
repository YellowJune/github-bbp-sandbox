import Foundation
import Metal
import MetalPerformanceShaders

struct Meta: Decodable {
    let model: String
    let V: Int
    let D: Int
    let N: Int
    let full_bytes: Int
    let high_bytes: Int
    let low_bytes: Int?
}

func readData(_ path: String) throws -> Data { try Data(contentsOf: URL(fileURLWithPath: path)) }
func median(_ x: [Double]) -> Double { let s=x.sorted(); return s[s.count/2] }
func percentile(_ x:[Double],_ p:Double)->Double { let s=x.sorted(); return s[min(s.count-1,max(0,Int((Double(s.count-1)*p).rounded()))) ] }
func stats(_ x:[Double])->[String:Any] { let y=x.filter{$0.isFinite}; return y.isEmpty ? ["n":0] : ["n":y.count,"median_ms":median(y),"p10_ms":percentile(y,0.1),"p90_ms":percentile(y,0.9),"mean_ms":y.reduce(0,+)/Double(y.count)] }

if CommandLine.arguments.count < 3 { fputs("usage: mpsdirect DATA_DIR ARGMAX_METALLIB [reps] [queries]\n",stderr); exit(2) }
let dir=CommandLine.arguments[1], libPath=CommandLine.arguments[2]
let reps=CommandLine.arguments.count>3 ? Int(CommandLine.arguments[3])! : 20
let requested=CommandLine.arguments.count>4 ? Int(CommandLine.arguments[4])! : 3
let meta=try JSONDecoder().decode(Meta.self,from:readData(dir+"/meta.json"))
let fullData=try readData(dir+"/full_u16.bin"), hiddenData=try readData(dir+"/hidden_f32.bin")
guard let dev=MTLCreateSystemDefaultDevice() else { fatalError("No Metal") }
let queue=dev.makeCommandQueue()!, lib=try dev.makeLibrary(URL:URL(fileURLWithPath:libPath))
let a1=try dev.makeComputePipelineState(function:lib.makeFunction(name:"argmax_half_stage1")!)
let af=try dev.makeComputePipelineState(function:lib.makeFunction(name:"argmax_float_final_mps")!)

// Use Apple's recommended MPS row stride. If it differs from the checkpoint's
// tight row-major stride, repack once outside timing so the baseline is not
// handicapped by a suboptimal MPS layout.
let tightRowBytes=meta.D*2
let recommendedRowBytes=MPSMatrixDescriptor.rowBytes(forColumns:meta.D,dataType:.float16)
let matrixRowBytes=max(tightRowBytes,recommendedRowBytes)
let matrixBuffer=dev.makeBuffer(length:matrixRowBytes*meta.V,options:[.storageModeShared])!
fullData.withUnsafeBytes { srcRaw in
    let src=srcRaw.bindMemory(to:UInt8.self).baseAddress!
    let dst=matrixBuffer.contents().bindMemory(to:UInt8.self,capacity:matrixRowBytes*meta.V)
    for r in 0..<meta.V { memcpy(dst+r*matrixRowBytes,src+r*tightRowBytes,tightRowBytes) }
}
let matrixDesc=MPSMatrixDescriptor(dimensions:meta.V,columns:meta.D,rowBytes:matrixRowBytes,dataType:.float16)
let matrix=MPSMatrix(buffer:matrixBuffer,descriptor:matrixDesc)
let inputDesc=MPSVectorDescriptor(length:meta.D,dataType:.float16)
let resultDesc=MPSVectorDescriptor(length:meta.V,dataType:.float16)
let resultBuffer=dev.makeBuffer(length:resultDesc.vectorBytes,options:[.storageModeShared])!
let result=MPSVector(buffer:resultBuffer,descriptor:resultDesc)
let gemv=MPSMatrixVectorMultiplication(device:dev,transpose:false,rows:meta.V,columns:meta.D,alpha:1.0,beta:0.0)

let nBlocks=(meta.V+4095)/4096
let blockVals=dev.makeBuffer(length:nBlocks*4,options:[.storageModeShared])!
let blockIdx=dev.makeBuffer(length:nBlocks*4,options:[.storageModeShared])!
let winner=dev.makeBuffer(length:4,options:[.storageModeShared])!
let argGroups=MTLSize(width:nBlocks,height:1,depth:1), lanes=MTLSize(width:32,height:1,depth:1), one=MTLSize(width:1,height:1,depth:1)

// Prebuild one FP16 MPSVector per real hidden state outside timing.
let hiddenF32=hiddenData.withUnsafeBytes { raw in Array(raw.bindMemory(to:Float.self)) }
let nq=min(requested,meta.N)
var inputVectors:[MPSVector]=[]
for q in 0..<nq {
    let b=dev.makeBuffer(length:inputDesc.vectorBytes,options:[.storageModeShared])!
    let p=b.contents().bindMemory(to:Float16.self,capacity:inputDesc.vectorBytes/2)
    for j in 0..<meta.D { p[j]=Float16(hiddenF32[q*meta.D+j]) }
    inputVectors.append(MPSVector(buffer:b,descriptor:inputDesc))
}

func encodeArgmaxHalf(_ cb:MTLCommandBuffer) {
    var N=UInt32(meta.V)
    var e=cb.makeComputeCommandEncoder()!; e.setComputePipelineState(a1); e.setBuffer(resultBuffer,offset:0,index:0); e.setBuffer(blockVals,offset:0,index:1); e.setBuffer(blockIdx,offset:0,index:2); e.setBytes(&N,length:4,index:3); e.dispatchThreadgroups(argGroups,threadsPerThreadgroup:lanes); e.endEncoding()
    var NB=UInt32(nBlocks)
    e=cb.makeComputeCommandEncoder()!; e.setComputePipelineState(af); e.setBuffer(blockVals,offset:0,index:0); e.setBuffer(blockIdx,offset:0,index:1); e.setBuffer(winner,offset:0,index:2); e.setBytes(&NB,length:4,index:3); e.dispatchThreadgroups(one,threadsPerThreadgroup:lanes); e.endEncoding()
}
func run(_ q:Int)->(Double,Double,UInt32) {
    let cb=queue.makeCommandBuffer()!
    gemv.encode(commandBuffer:cb,inputMatrix:matrix,inputVector:inputVectors[q],resultVector:result)
    encodeArgmaxHalf(cb)
    let t0=DispatchTime.now().uptimeNanoseconds; cb.commit(); cb.waitUntilCompleted(); if cb.status != .completed { fatalError("MPS failure \(String(describing:cb.error))") }
    let wall=Double(DispatchTime.now().uptimeNanoseconds-t0)/1e6
    let gpu=(cb.gpuEndTime>cb.gpuStartTime) ? (cb.gpuEndTime-cb.gpuStartTime)*1000.0 : Double.nan
    return(gpu,wall,winner.contents().bindMemory(to:UInt32.self,capacity:1)[0])
}

var rows:[[String:Any]]=[]
for q in 0..<nq { let r=run(q); rows.append(["query":q,"winner":Int(r.2),"one_run_gpu_ms":r.0,"one_run_wall_ms":r.1]) }
for q in 0..<nq { for _ in 0..<6 { _=run(q) } }
var gpu:[Double]=[],wall:[Double]=[]
for r in 0..<reps { for q in 0..<nq { let x=run(q);gpu.append(x.0);wall.append(x.1) } }
let resultJSON:[String:Any]=[
    "kind":"proofbits_direct_mps_matrix_vector_dense_decision_baseline",
    "device":dev.name,"model":meta.model,"V":meta.V,"D":meta.D,"queries":nq,"reps_per_query":reps,
    "matrix_row_bytes_tight":tightRowBytes,"matrix_row_bytes_mps_recommended":recommendedRowBytes,"matrix_row_bytes_used":matrixRowBytes,
    "input_vector_bytes":inputDesc.vectorBytes,"result_vector_bytes":resultDesc.vectorBytes,
    "winners":rows,"gpu":stats(gpu),"wall":stats(wall),
    "operation":"MPSMatrixVectorMultiplication FP16 + custom GPU FP16 argmax reduction in one command buffer",
    "caveat":"Strong Apple MPS matrix-vector baseline using recommended matrix stride. Numerical accumulation/output semantics may differ from ProofBits FP32-accumulating custom Metal reference; winner IDs are reported."
]
let data=try JSONSerialization.data(withJSONObject:resultJSON,options:[.prettyPrinted,.sortedKeys]);print(String(data:data,encoding:.utf8)!);try data.write(to:URL(fileURLWithPath:dir+"/proofbits_mps_direct.json"))
