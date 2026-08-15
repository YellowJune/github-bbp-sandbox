import Foundation
import Metal

struct Meta: Decodable {
    let model: String
    let V: Int
    let D: Int
    let N: Int
    let full_bytes: Int
    let high_bytes: Int
}

func readData(_ path: String) throws -> Data { try Data(contentsOf: URL(fileURLWithPath: path)) }

func median(_ x: [Double]) -> Double {
    let s = x.sorted()
    return s[s.count / 2]
}

func percentile(_ x: [Double], _ p: Double) -> Double {
    let s = x.sorted()
    let idx = min(s.count - 1, max(0, Int((Double(s.count - 1) * p).rounded())))
    return s[idx]
}

func argmax(_ ptr: UnsafePointer<Float>, _ n: Int) -> Int {
    var k = 0
    var best = ptr[0]
    if n > 1 {
        for i in 1..<n where ptr[i] > best { best = ptr[i]; k = i }
    }
    return k
}

if CommandLine.arguments.count < 3 {
    fputs("usage: proofbits_metal_bench DATA_DIR METALLIB [reps] [queries]\n", stderr)
    exit(2)
}
let dataDir = CommandLine.arguments[1]
let libPath = CommandLine.arguments[2]
let reps = CommandLine.arguments.count > 3 ? Int(CommandLine.arguments[3])! : 20
let requestedQueries = CommandLine.arguments.count > 4 ? Int(CommandLine.arguments[4])! : 4

let metaData = try readData(dataDir + "/meta.json")
let meta = try JSONDecoder().decode(Meta.self, from: metaData)
let fullData = try readData(dataDir + "/full_u16.bin")
let highData = try readData(dataDir + "/high_u8.bin")
let hiddenData = try readData(dataDir + "/hidden_f32.bin")
precondition(fullData.count == meta.V * meta.D * 2)
precondition(highData.count == meta.V * meta.D)
precondition(hiddenData.count == meta.N * meta.D * 4)

 guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
let queue = device.makeCommandQueue()!
let library = try device.makeLibrary(URL: URL(fileURLWithPath: libPath))
let densePSO = try device.makeComputePipelineState(function: library.makeFunction(name: "dense_fp16_row")!)
let highPSO = try device.makeComputePipelineState(function: library.makeFunction(name: "proofbits_high_upper_row")!)
precondition(densePSO.threadExecutionWidth == 32 || highPSO.threadExecutionWidth == 32)

func buffer(from data: Data) -> MTLBuffer {
    return data.withUnsafeBytes { raw in
        device.makeBuffer(bytes: raw.baseAddress!, length: data.count, options: [.storageModeShared])!
    }
}
let fullBuffer = buffer(from: fullData)
let highBuffer = buffer(from: highData)
let hiddenBuffer = buffer(from: hiddenData)
let denseOut = device.makeBuffer(length: meta.V * 4, options: [.storageModeShared])!
let highOut = device.makeBuffer(length: meta.V * 4, options: [.storageModeShared])!
var d32 = UInt32(meta.D)
let dBuffer = device.makeBuffer(bytes: &d32, length: MemoryLayout<UInt32>.size, options: [.storageModeShared])!

let groups = MTLSize(width: meta.V, height: 1, depth: 1)
let threads = MTLSize(width: 32, height: 1, depth: 1)

func run(_ pso: MTLComputePipelineState, _ input: MTLBuffer, _ output: MTLBuffer, query: Int) -> (Double, Double) {
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(input, offset: 0, index: 0)
    enc.setBuffer(hiddenBuffer, offset: query * meta.D * 4, index: 1)
    enc.setBuffer(output, offset: 0, index: 2)
    enc.setBuffer(dBuffer, offset: 0, index: 3)
    enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
    enc.endEncoding()
    let t0 = DispatchTime.now().uptimeNanoseconds
    cb.commit()
    cb.waitUntilCompleted()
    let t1 = DispatchTime.now().uptimeNanoseconds
    if cb.status != .completed { fatalError("command buffer failed: \(String(describing: cb.error))") }
    let gpu = (cb.gpuEndTime > cb.gpuStartTime) ? (cb.gpuEndTime - cb.gpuStartTime) * 1000.0 : Double.nan
    let wall = Double(t1 - t0) / 1e6
    return (gpu, wall)
}

let nq = min(requestedQueries, meta.N)
var denseGPU: [Double] = [], highGPU: [Double] = []
var denseWall: [Double] = [], highWall: [Double] = []
var correctness: [[String: Any]] = []

// Correctness and candidate accounting on each selected real hidden state.
for q in 0..<nq {
    _ = run(densePSO, fullBuffer, denseOut, query: q)
    _ = run(highPSO, highBuffer, highOut, query: q)
    let dz = denseOut.contents().bindMemory(to: Float.self, capacity: meta.V)
    let uz = highOut.contents().bindMemory(to: Float.self, capacity: meta.V)
    let ref = argmax(UnsafePointer(dz), meta.V)
    let pilot = argmax(UnsafePointer(uz), meta.V)
    let B = dz[pilot]
    var survivors = 0
    var boundViolations = 0
    var minSlack = Double.greatestFiniteMagnitude
    for i in 0..<meta.V {
        if uz[i] >= B { survivors += 1 }
        let slack = Double(uz[i] - dz[i])
        if slack < minSlack { minSlack = slack }
        if uz[i] + 1e-5 < dz[i] { boundViolations += 1 }
    }
    correctness.append([
        "query": q,
        "dense_argmax": ref,
        "upper_argmax_pilot": pilot,
        "pilot_is_winner": pilot == ref,
        "winner_survives": uz[ref] >= B,
        "survivors": survivors,
        "survivor_fraction": Double(survivors) / Double(meta.V),
        "bound_violations_tolerance_1e-5": boundViolations,
        "minimum_observed_upper_minus_dense": minSlack
    ])
}

// Warmup both paths in alternating order.
for q in 0..<nq {
    for _ in 0..<5 {
        _ = run(densePSO, fullBuffer, denseOut, query: q)
        _ = run(highPSO, highBuffer, highOut, query: q)
    }
}

// Counterbalanced measurements to reduce thermal/order bias.
for r in 0..<reps {
    for q in 0..<nq {
        if ((r + q) & 1) == 0 {
            let d = run(densePSO, fullBuffer, denseOut, query: q)
            let h = run(highPSO, highBuffer, highOut, query: q)
            denseGPU.append(d.0); denseWall.append(d.1); highGPU.append(h.0); highWall.append(h.1)
        } else {
            let h = run(highPSO, highBuffer, highOut, query: q)
            let d = run(densePSO, fullBuffer, denseOut, query: q)
            denseGPU.append(d.0); denseWall.append(d.1); highGPU.append(h.0); highWall.append(h.1)
        }
    }
}

func stats(_ x: [Double]) -> [String: Any] {
    let y = x.filter { $0.isFinite }
    if y.isEmpty { return ["n": 0] }
    return ["n": y.count, "median_ms": median(y), "p10_ms": percentile(y, 0.10), "p90_ms": percentile(y, 0.90), "mean_ms": y.reduce(0,+)/Double(y.count)]
}
let denseGpuMed = median(denseGPU.filter {$0.isFinite})
let highGpuMed = median(highGPU.filter {$0.isFinite})
let denseWallMed = median(denseWall)
let highWallMed = median(highWall)
let result: [String: Any] = [
    "kind": "proofbits_metal_matched_stage_benchmark",
    "device": device.name,
    "model": meta.model,
    "V": meta.V,
    "D": meta.D,
    "queries": nq,
    "repetitions_per_query": reps,
    "dense_weight_bytes": meta.full_bytes,
    "high_plane_bytes": meta.high_bytes,
    "logical_weight_byte_ratio": Double(meta.full_bytes) / Double(meta.high_bytes),
    "dense_pipeline_thread_width": densePSO.threadExecutionWidth,
    "high_pipeline_thread_width": highPSO.threadExecutionWidth,
    "correctness": correctness,
    "dense_gpu": stats(denseGPU),
    "high_gpu": stats(highGPU),
    "dense_wall": stats(denseWall),
    "high_wall": stats(highWall),
    "matched_high_vs_dense_gpu_speedup": denseGpuMed / highGpuMed,
    "matched_high_vs_dense_wall_speedup": denseWallMed / highWallMed,
    "caveat": "Matched custom Metal stage benchmark only. It compares dense FP16-row scoring with the ProofBits high-byte certified-upper pass using identical one-SIMDgroup-per-row decomposition. It is not yet the full pilot/threshold/refine pipeline and not a comparison against the platform's optimized native GEMV/MPS kernel."
]
let json = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
print(String(data: json, encoding: .utf8)!)
try json.write(to: URL(fileURLWithPath: dataDir + "/proofbits_metal_matched.json"))
