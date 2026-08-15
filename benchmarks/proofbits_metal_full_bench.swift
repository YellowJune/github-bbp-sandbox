import Foundation
import Metal

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

if CommandLine.arguments.count < 3 { fputs("usage: fullbench DATA_DIR METALLIB [reps] [queries]\n",stderr); exit(2) }
let dir=CommandLine.arguments[1], libPath=CommandLine.arguments[2]
let reps=CommandLine.arguments.count>3 ? Int(CommandLine.arguments[3])! : 12
let requested=CommandLine.arguments.count>4 ? Int(CommandLine.arguments[4])! : 3
let meta=try JSONDecoder().decode(Meta.self,from:readData(dir+"/meta.json"))
let fullData=try readData(dir+"/full_u16.bin"), highData=try readData(dir+"/high_u8.bin"), lowData=try readData(dir+"/low_u8.bin"), hiddenData=try readData(dir+"/hidden_f32.bin")
precondition(fullData.count==meta.V*meta.D*2 && highData.count==meta.V*meta.D && lowData.count==meta.V*meta.D)
guard let dev=MTLCreateSystemDefaultDevice() else { fatalError("No Metal") }
let queue=dev.makeCommandQueue()!, lib=try dev.makeLibrary(URL:URL(fileURLWithPath:libPath))
func pso(_ name:String)throws->MTLComputePipelineState { try dev.makeComputePipelineState(function:lib.makeFunction(name:name)!) }
let densePSO=try pso("dense_fp16_row"), highPSO=try pso("proofbits_high_upper_row"), a1PSO=try pso("argmax_stage1"), afPSO=try pso("argmax_final"), pilotPSO=try pso("exact_pilot_row"), refinePSO=try pso("conditional_refine_row")
func buffer(_ d:Data)->MTLBuffer { d.withUnsafeBytes{dev.makeBuffer(bytes:$0.baseAddress!,length:d.count,options:[.storageModeShared])!} }
let full=buffer(fullData), high=buffer(highData), low=buffer(lowData), hidden=buffer(hiddenData)
let denseOut=dev.makeBuffer(length:meta.V*4,options:[.storageModeShared])!, U=dev.makeBuffer(length:meta.V*4,options:[.storageModeShared])!, exactOut=dev.makeBuffer(length:meta.V*4,options:[.storageModeShared])!
let nBlocks=(meta.V+4095)/4096
let blockVals=dev.makeBuffer(length:nBlocks*4,options:[.storageModeShared])!, blockIdx=dev.makeBuffer(length:nBlocks*4,options:[.storageModeShared])!
let pilotIdx=dev.makeBuffer(length:4,options:[.storageModeShared])!, winnerIdx=dev.makeBuffer(length:4,options:[.storageModeShared])!, Bbuf=dev.makeBuffer(length:4,options:[.storageModeShared])!
let rowGroups=MTLSize(width:meta.V,height:1,depth:1), oneGroup=MTLSize(width:1,height:1,depth:1), argGroups=MTLSize(width:nBlocks,height:1,depth:1), lanes=MTLSize(width:32,height:1,depth:1)

func encoder(_ cb:MTLCommandBuffer,_ p:MTLComputePipelineState)->MTLComputeCommandEncoder { let e=cb.makeComputeCommandEncoder()!; e.setComputePipelineState(p); return e }
func encodeArgmax(_ cb:MTLCommandBuffer, values:MTLBuffer, out:MTLBuffer) {
    var N=UInt32(meta.V)
    var e=encoder(cb,a1PSO); e.setBuffer(values,offset:0,index:0); e.setBuffer(blockVals,offset:0,index:1); e.setBuffer(blockIdx,offset:0,index:2); e.setBytes(&N,length:4,index:3); e.dispatchThreadgroups(argGroups,threadsPerThreadgroup:lanes); e.endEncoding()
    var NB=UInt32(nBlocks)
    e=encoder(cb,afPSO); e.setBuffer(blockVals,offset:0,index:0); e.setBuffer(blockIdx,offset:0,index:1); e.setBuffer(out,offset:0,index:2); e.setBytes(&NB,length:4,index:3); e.dispatchThreadgroups(oneGroup,threadsPerThreadgroup:lanes); e.endEncoding()
}
func finish(_ cb:MTLCommandBuffer,_ t0:UInt64)->(Double,Double) { cb.commit(); cb.waitUntilCompleted(); if cb.status != .completed { fatalError("GPU failure \(String(describing:cb.error))") }; let wall=Double(DispatchTime.now().uptimeNanoseconds-t0)/1e6; let gpu=(cb.gpuEndTime>cb.gpuStartTime) ? (cb.gpuEndTime-cb.gpuStartTime)*1000.0 : Double.nan; return(gpu,wall) }
func runDenseDecision(_ q:Int)->(Double,Double) {
    let cb=queue.makeCommandBuffer()!; var D=UInt32(meta.D)
    let e=encoder(cb,densePSO); e.setBuffer(full,offset:0,index:0); e.setBuffer(hidden,offset:q*meta.D*4,index:1); e.setBuffer(denseOut,offset:0,index:2); e.setBytes(&D,length:4,index:3); e.dispatchThreadgroups(rowGroups,threadsPerThreadgroup:lanes); e.endEncoding()
    encodeArgmax(cb,values:denseOut,out:winnerIdx); let t0=DispatchTime.now().uptimeNanoseconds; return finish(cb,t0)
}
func runProofBits(_ q:Int)->(Double,Double) {
    let cb=queue.makeCommandBuffer()!; var D=UInt32(meta.D)
    var e=encoder(cb,highPSO); e.setBuffer(high,offset:0,index:0); e.setBuffer(hidden,offset:q*meta.D*4,index:1); e.setBuffer(U,offset:0,index:2); e.setBytes(&D,length:4,index:3); e.dispatchThreadgroups(rowGroups,threadsPerThreadgroup:lanes); e.endEncoding()
    encodeArgmax(cb,values:U,out:pilotIdx)
    e=encoder(cb,pilotPSO); e.setBuffer(high,offset:0,index:0); e.setBuffer(low,offset:0,index:1); e.setBuffer(hidden,offset:q*meta.D*4,index:2); e.setBuffer(pilotIdx,offset:0,index:3); e.setBuffer(Bbuf,offset:0,index:4); e.setBytes(&D,length:4,index:5); e.dispatchThreadgroups(oneGroup,threadsPerThreadgroup:lanes); e.endEncoding()
    e=encoder(cb,refinePSO); e.setBuffer(high,offset:0,index:0); e.setBuffer(low,offset:0,index:1); e.setBuffer(hidden,offset:q*meta.D*4,index:2); e.setBuffer(U,offset:0,index:3); e.setBuffer(Bbuf,offset:0,index:4); e.setBuffer(exactOut,offset:0,index:5); e.setBytes(&D,length:4,index:6); e.dispatchThreadgroups(rowGroups,threadsPerThreadgroup:lanes); e.endEncoding()
    encodeArgmax(cb,values:exactOut,out:winnerIdx); let t0=DispatchTime.now().uptimeNanoseconds; return finish(cb,t0)
}

let nq=min(requested,meta.N)
var correctness:[[String:Any]]=[]
for q in 0..<nq {
    _=runDenseDecision(q); let denseWin=winnerIdx.contents().bindMemory(to:UInt32.self,capacity:1)[0]
    _=runProofBits(q); let pbWin=winnerIdx.contents().bindMemory(to:UInt32.self,capacity:1)[0]; let pilot=pilotIdx.contents().bindMemory(to:UInt32.self,capacity:1)[0]; let B=Bbuf.contents().bindMemory(to:Float.self,capacity:1)[0]; let up=U.contents().bindMemory(to:Float.self,capacity:meta.V)
    var survivors=0; for i in 0..<meta.V { if up[i]>=B { survivors += 1 } }
    correctness.append(["query":q,"dense_winner":Int(denseWin),"proofbits_winner":Int(pbWin),"exact":denseWin==pbWin,"pilot":Int(pilot),"pilot_is_winner":pilot==denseWin,"survivors":survivors,"survivor_fraction":Double(survivors)/Double(meta.V)])
}
for q in 0..<nq { for _ in 0..<4 { _=runDenseDecision(q); _=runProofBits(q) } }
var dGPU:[Double]=[],pGPU:[Double]=[],dWall:[Double]=[],pWall:[Double]=[]
for r in 0..<reps { for q in 0..<nq {
    if ((r+q)&1)==0 { let d=runDenseDecision(q),p=runProofBits(q); dGPU.append(d.0);dWall.append(d.1);pGPU.append(p.0);pWall.append(p.1) }
    else { let p=runProofBits(q),d=runDenseDecision(q); dGPU.append(d.0);dWall.append(d.1);pGPU.append(p.0);pWall.append(p.1) }
} }
let dg=median(dGPU.filter{$0.isFinite}), pg=median(pGPU.filter{$0.isFinite}), dw=median(dWall), pw=median(pWall)
let result:[String:Any]=["kind":"proofbits_metal_full_decision_benchmark","device":dev.name,"model":meta.model,"V":meta.V,"D":meta.D,"queries":nq,"reps_per_query":reps,"all_exact":correctness.allSatisfy{($0["exact"] as? Bool)==true},"correctness":correctness,"dense_decision_gpu":stats(dGPU),"proofbits_decision_gpu":stats(pGPU),"dense_decision_wall":stats(dWall),"proofbits_decision_wall":stats(pWall),"full_decision_gpu_speedup":dg/pg,"full_decision_wall_speedup":dw/pw,"caveat":"Full custom Metal decision-head pipeline with GPU reductions and conditional survivor refinement. Rejected rows still launch one SIMDgroup and read U/B but do not read low-byte weights. This is matched against the same custom dense Metal row-scoring plus GPU argmax, not yet against an optimized MPS/native GEMV library kernel or an end-to-end transformer decode."]
let data=try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]); print(String(data:data,encoding:.utf8)!); try data.write(to:URL(fileURLWithPath:dir+"/proofbits_metal_full.json"))
