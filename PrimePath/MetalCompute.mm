#import "MetalCompute.h"
#import <Metal/Metal.h>

// Ring buffer size — with async dispatch, allows N-1 batches in flight
static const int RING_SIZE = 3;
static const uint32_t MAX_BATCH = 262144; // 256K max batch

@implementation MetalCompute {
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;
    id<MTLLibrary> _library;

    id<MTLComputePipelineState> _wieferichPSO;
    id<MTLComputePipelineState> _wallsunsunPSO;
    id<MTLComputePipelineState> _twinPSO;
    id<MTLComputePipelineState> _sophiePSO;
    id<MTLComputePipelineState> _cousinPSO;
    id<MTLComputePipelineState> _sexyPSO;
    id<MTLComputePipelineState> _primalityPSO;
    id<MTLComputePipelineState> _wilsonPSO;
    id<MTLComputePipelineState> _wilsonSegPSO;
    id<MTLComputePipelineState> _mersenneTrialPSO;
    id<MTLComputePipelineState> _fermatFactorPSO;

    // Pre-allocated ring buffers
    id<MTLBuffer> _ringInput[RING_SIZE];
    id<MTLBuffer> _ringOutput[RING_SIZE];
    id<MTLBuffer> _ringCount[RING_SIZE];
    id<MTLCommandBuffer> _ringCmdBuf[RING_SIZE]; // in-flight command buffers
    uint32_t _ringBatchCount[RING_SIZE];          // count for each in-flight batch
    int _ringIdx;

    // GPU timing stats
    uint64_t _totalThreads;
    uint64_t _totalBatches;
    double _totalGpuTimeSec;
    double _totalWallTimeSec;
    NSDate *_statsStart;

    // Rolling window for utilization (last 10 seconds)
    double _windowGpuSec;
    NSDate *_windowStart;
    double _windowWallSec;
}

- (instancetype)init {
    self = [super init];
    if (!self) return nil;

    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        NSLog(@"Metal: no GPU device found");
        return self;
    }

    _queue = [_device newCommandQueue];
    NSError *error = nil;
    _library = [_device newDefaultLibrary];
    if (!_library) {
        NSLog(@"Metal: failed to load shader library: %@", error);
        return self;
    }

    _wieferichPSO  = [self psoForFunction:@"wieferich_batch"];
    _wallsunsunPSO = [self psoForFunction:@"wallsunsun_batch"];
    _twinPSO       = [self psoForFunction:@"twin_batch"];
    _sophiePSO     = [self psoForFunction:@"sophie_batch"];
    _cousinPSO     = [self psoForFunction:@"cousin_batch"];
    _sexyPSO       = [self psoForFunction:@"sexy_batch"];
    _primalityPSO  = [self psoForFunction:@"primality_batch"];
    _wilsonPSO     = [self psoForFunction:@"wilson_batch"];
    _wilsonSegPSO  = [self psoForFunction:@"wilson_segments"];
    _mersenneTrialPSO = [self psoForFunction:@"mersenne_trial_batch"];
    _fermatFactorPSO  = [self psoForFunction:@"fermat_factor_batch"];

    // Pre-allocate ring buffers for max batch size
    for (int i = 0; i < RING_SIZE; i++) {
        _ringInput[i]  = [_device newBufferWithLength:MAX_BATCH * sizeof(uint64_t)
                                              options:MTLResourceStorageModeShared];
        _ringOutput[i] = [_device newBufferWithLength:MAX_BATCH * sizeof(uint8_t)
                                              options:MTLResourceStorageModeShared];
        _ringCount[i]  = [_device newBufferWithLength:sizeof(uint32_t)
                                              options:MTLResourceStorageModeShared];
        _ringCmdBuf[i] = nil;
        _ringBatchCount[i] = 0;
    }
    _ringIdx = 0;

    _totalThreads = 0;
    _totalBatches = 0;
    _totalGpuTimeSec = 0;
    _totalWallTimeSec = 0;
    _statsStart = [NSDate date];
    _windowGpuSec = 0;
    _windowStart = [NSDate date];
    _windowWallSec = 0;

    NSLog(@"Metal: GPU ready — %@ | max threads/group: %lu | ring: %d x %uK (async pipeline)",
          _device.name, (unsigned long)_wieferichPSO.maxTotalThreadsPerThreadgroup,
          RING_SIZE, MAX_BATCH / 1024);
    return self;
}

- (id<MTLComputePipelineState>)psoForFunction:(NSString *)name {
    id<MTLFunction> fn = [_library newFunctionWithName:name];
    if (!fn) {
        NSLog(@"Metal: function '%@' not found", name);
        return nil;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> pso = [_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pso) NSLog(@"Metal: PSO error for '%@': %@", name, error);
    return pso;
}

- (BOOL)available {
    return _device != nil && _library != nil;
}

// ── Async pipelined dispatch ────────────────────────────────────────
//
// Instead of commit + waitUntilCompleted per batch, we:
// 1. Wait for the ring slot we're about to use (if it has an in-flight batch)
// 2. Read results from that completed batch → return them
// 3. Submit new work into the now-free slot → GPU starts immediately
//
// This keeps GPU working on slot N while CPU processes results from slot N-1.
// With RING_SIZE=3, up to 2 batches can be in flight.

- (void)waitForSlot:(int)idx {
    if (_ringCmdBuf[idx] != nil) {
        [_ringCmdBuf[idx] waitUntilCompleted];

        // Collect timing from completed batch
        double gpuTimeSec = 0;
        if (@available(macOS 10.15, *)) {
            CFTimeInterval gpuStart = _ringCmdBuf[idx].GPUStartTime;
            CFTimeInterval gpuEnd = _ringCmdBuf[idx].GPUEndTime;
            if (gpuEnd > gpuStart) gpuTimeSec = gpuEnd - gpuStart;
        }
        _totalGpuTimeSec += gpuTimeSec;
        _windowGpuSec += gpuTimeSec;
        _ringCmdBuf[idx] = nil;
    }
}

// Submit work to GPU asynchronously, return results from the PREVIOUS batch
// that was using this ring slot. On first call for a slot, returns nil.
- (NSData *)dispatchKernel:(id<MTLComputePipelineState>)pso
                     input:(const uint64_t *)data
                     count:(uint32_t)count {
    if (!pso || count == 0) return nil;
    if (count > MAX_BATCH) count = MAX_BATCH;

    int idx = _ringIdx;
    _ringIdx = (_ringIdx + 1) % RING_SIZE;

    // Wait for any in-flight work on this slot to complete
    [self waitForSlot:idx];

    // Capture results from previous batch in this slot (if any)
    NSData *prevResults = nil;
    uint32_t prevCount = _ringBatchCount[idx];
    if (prevCount > 0) {
        prevResults = [NSData dataWithBytes:_ringOutput[idx].contents
                                     length:prevCount * sizeof(uint8_t)];
    }

    // Fill input buffer for new batch
    memcpy(_ringInput[idx].contents, data, count * sizeof(uint64_t));
    memset(_ringOutput[idx].contents, 0, count * sizeof(uint8_t));
    *((uint32_t *)_ringCount[idx].contents) = count;

    // Encode and submit
    id<MTLCommandBuffer> cmdBuf = [_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:pso];
    [encoder setBuffer:_ringInput[idx]  offset:0 atIndex:0];
    [encoder setBuffer:_ringOutput[idx] offset:0 atIndex:1];
    [encoder setBuffer:_ringCount[idx]  offset:0 atIndex:2];

    NSUInteger threadGroupSize = pso.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > 256) threadGroupSize = 256;
    MTLSize gridSize = MTLSizeMake(count, 1, 1);
    MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
    [encoder endEncoding];

    [cmdBuf commit]; // GPU starts immediately — no wait!

    _ringCmdBuf[idx] = cmdBuf;
    _ringBatchCount[idx] = count;
    _totalThreads += count;
    _totalBatches++;

    return prevResults; // may be nil on first call
}

// Flush: wait for ALL in-flight batches and return the last one's results
- (NSData *)flushAndGetLast {
    // Find the most recently submitted slot
    int lastIdx = (_ringIdx + RING_SIZE - 1) % RING_SIZE;
    [self waitForSlot:lastIdx];

    uint32_t count = _ringBatchCount[lastIdx];
    if (count == 0) return nil;

    NSData *result = [NSData dataWithBytes:_ringOutput[lastIdx].contents
                                    length:count * sizeof(uint8_t)];
    _ringBatchCount[lastIdx] = 0;

    // Also wait for any other in-flight slots
    for (int i = 0; i < RING_SIZE; i++) {
        [self waitForSlot:i];
    }

    return result;
}

// ── Synchronous dispatch (simpler path for one-off calls) ───────────
// Used by Wilson segmented and when we need guaranteed results immediately.
- (NSData *)dispatchKernelSync:(id<MTLComputePipelineState>)pso
                         input:(const uint64_t *)data
                         count:(uint32_t)count {
    if (!pso || count == 0) return nil;
    if (count > MAX_BATCH) count = MAX_BATCH;

    int idx = _ringIdx;
    _ringIdx = (_ringIdx + 1) % RING_SIZE;

    [self waitForSlot:idx];

    memcpy(_ringInput[idx].contents, data, count * sizeof(uint64_t));
    memset(_ringOutput[idx].contents, 0, count * sizeof(uint8_t));
    *((uint32_t *)_ringCount[idx].contents) = count;

    id<MTLCommandBuffer> cmdBuf = [_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:pso];
    [encoder setBuffer:_ringInput[idx]  offset:0 atIndex:0];
    [encoder setBuffer:_ringOutput[idx] offset:0 atIndex:1];
    [encoder setBuffer:_ringCount[idx]  offset:0 atIndex:2];

    NSUInteger threadGroupSize = pso.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > 256) threadGroupSize = 256;
    MTLSize gridSize = MTLSizeMake(count, 1, 1);
    MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
    [encoder endEncoding];

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    double gpuTimeSec = 0;
    if (@available(macOS 10.15, *)) {
        CFTimeInterval gpuStart = cmdBuf.GPUStartTime;
        CFTimeInterval gpuEnd = cmdBuf.GPUEndTime;
        if (gpuEnd > gpuStart) gpuTimeSec = gpuEnd - gpuStart;
    }

    _totalThreads += count;
    _totalBatches++;
    _totalGpuTimeSec += gpuTimeSec;
    _windowGpuSec += gpuTimeSec;
    _ringBatchCount[idx] = 0;
    _ringCmdBuf[idx] = nil;

    return [NSData dataWithBytes:_ringOutput[idx].contents length:count * sizeof(uint8_t)];
}

// ── Public batch methods ────────────────────────────────────────────
// Use sync dispatch so callers get results immediately (async pipeline
// is used at the TaskManager level where we can manage the overlap).

- (NSData *)runWieferichBatch:(const uint64_t *)primes count:(uint32_t)count {
    return [self dispatchKernelSync:_wieferichPSO input:primes count:count];
}
- (NSData *)runWallSunSunBatch:(const uint64_t *)primes count:(uint32_t)count {
    return [self dispatchKernelSync:_wallsunsunPSO input:primes count:count];
}
- (NSData *)runTwinBatch:(const uint64_t *)candidates count:(uint32_t)count {
    return [self dispatchKernelSync:_twinPSO input:candidates count:count];
}
- (NSData *)runSophieBatch:(const uint64_t *)candidates count:(uint32_t)count {
    return [self dispatchKernelSync:_sophiePSO input:candidates count:count];
}
- (NSData *)runCousinBatch:(const uint64_t *)candidates count:(uint32_t)count {
    return [self dispatchKernelSync:_cousinPSO input:candidates count:count];
}
- (NSData *)runSexyBatch:(const uint64_t *)candidates count:(uint32_t)count {
    return [self dispatchKernelSync:_sexyPSO input:candidates count:count];
}
- (NSData *)runPrimalityBatch:(const uint64_t *)candidates count:(uint32_t)count {
    return [self dispatchKernelSync:_primalityPSO input:candidates count:count];
}

- (NSData *)runWilsonBatch:(const uint64_t *)primes count:(uint32_t)count {
    return [self dispatchKernelSync:_wilsonPSO input:primes count:count];
}

// ── Mersenne trial factoring dispatch ────────────────────────────────
// Candidates are packed as 6× uint32 per entry (q.lo, q.mid, q.hi, mu.lo, mu.mid, mu.hi).
// Params buffer: [0]=exponent p, [1]=count.
- (NSData *)runMersenneTrialBatch:(const uint32_t *)candidates
                            count:(uint32_t)count
                         exponent:(uint64_t)p {
    if (!_mersenneTrialPSO || count == 0) return nil;
    if (count > MAX_BATCH) count = MAX_BATCH;

    for (int i = 0; i < RING_SIZE; i++) [self waitForSlot:i];

    size_t inputSize = (size_t)count * 6 * sizeof(uint32_t);
    id<MTLBuffer> inputBuf = [_device newBufferWithBytes:candidates length:inputSize
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuf = [_device newBufferWithLength:count * sizeof(uint8_t)
                                                   options:MTLResourceStorageModeShared];
    memset(outputBuf.contents, 0, count);

    uint64_t paramsData[2] = {p, (uint64_t)count};
    id<MTLBuffer> paramsBuf = [_device newBufferWithBytes:paramsData length:sizeof(paramsData)
                                                  options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmdBuf = [_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:_mersenneTrialPSO];
    [encoder setBuffer:inputBuf  offset:0 atIndex:0];
    [encoder setBuffer:outputBuf offset:0 atIndex:1];
    [encoder setBuffer:paramsBuf offset:0 atIndex:2];

    NSUInteger threadGroupSize = _mersenneTrialPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > 256) threadGroupSize = 256;
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    double gpuTimeSec = 0;
    if (@available(macOS 10.15, *)) {
        CFTimeInterval gpuStart = cmdBuf.GPUStartTime;
        CFTimeInterval gpuEnd = cmdBuf.GPUEndTime;
        if (gpuEnd > gpuStart) gpuTimeSec = gpuEnd - gpuStart;
    }
    _totalThreads += count;
    _totalBatches++;
    _totalGpuTimeSec += gpuTimeSec;
    _windowGpuSec += gpuTimeSec;

    return [NSData dataWithBytes:outputBuf.contents length:count * sizeof(uint8_t)];
}

// ── Fermat factor search dispatch ────────────────────────────────────
- (NSData *)runFermatFactorBatch:(const uint32_t *)candidates
                           count:(uint32_t)count
                     fermatIndex:(uint64_t)m {
    if (!_fermatFactorPSO || count == 0) return nil;
    if (count > MAX_BATCH) count = MAX_BATCH;

    for (int i = 0; i < RING_SIZE; i++) [self waitForSlot:i];

    size_t inputSize = (size_t)count * 6 * sizeof(uint32_t);
    id<MTLBuffer> inputBuf = [_device newBufferWithBytes:candidates length:inputSize
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuf = [_device newBufferWithLength:count * sizeof(uint8_t)
                                                   options:MTLResourceStorageModeShared];
    memset(outputBuf.contents, 0, count);

    uint64_t paramsData[2] = {m, (uint64_t)count};
    id<MTLBuffer> paramsBuf = [_device newBufferWithBytes:paramsData length:sizeof(paramsData)
                                                  options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmdBuf = [_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:_fermatFactorPSO];
    [encoder setBuffer:inputBuf  offset:0 atIndex:0];
    [encoder setBuffer:outputBuf offset:0 atIndex:1];
    [encoder setBuffer:paramsBuf offset:0 atIndex:2];

    NSUInteger threadGroupSize = _fermatFactorPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > 256) threadGroupSize = 256;
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    double gpuTimeSec = 0;
    if (@available(macOS 10.15, *)) {
        CFTimeInterval gpuStart = cmdBuf.GPUStartTime;
        CFTimeInterval gpuEnd = cmdBuf.GPUEndTime;
        if (gpuEnd > gpuStart) gpuTimeSec = gpuEnd - gpuStart;
    }
    _totalThreads += count;
    _totalBatches++;
    _totalGpuTimeSec += gpuTimeSec;
    _windowGpuSec += gpuTimeSec;

    return [NSData dataWithBytes:outputBuf.contents length:count * sizeof(uint8_t)];
}

- (BOOL)runWilsonSegments:(uint64_t)prime
              numSegments:(uint32_t)numSegments
                partialLo:(uint64_t *)outLo
                partialHi:(uint64_t *)outHi {
    if (!_wilsonSegPSO || numSegments == 0) return NO;

    // Flush any async work first
    for (int i = 0; i < RING_SIZE; i++) [self waitForSlot:i];

    uint64_t params[2] = {prime, numSegments};
    id<MTLBuffer> paramsBuf = [_device newBufferWithBytes:params length:sizeof(params)
                                                  options:MTLResourceStorageModeShared];
    size_t outSize = numSegments * sizeof(uint64_t);
    id<MTLBuffer> loBuf = [_device newBufferWithLength:outSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> hiBuf = [_device newBufferWithLength:outSize options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmdBuf = [_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:_wilsonSegPSO];
    [encoder setBuffer:paramsBuf offset:0 atIndex:0];
    [encoder setBuffer:loBuf     offset:0 atIndex:1];
    [encoder setBuffer:hiBuf     offset:0 atIndex:2];

    NSUInteger threadGroupSize = _wilsonSegPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > 256) threadGroupSize = 256;
    MTLSize gridSize = MTLSizeMake(numSegments, 1, 1);
    MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    memcpy(outLo, loBuf.contents, outSize);
    memcpy(outHi, hiBuf.contents, outSize);
    return YES;
}

// ── GPU performance stats ────────────────────────────────────────────

- (double)gpuUtilization {
    // Use a 10-second rolling window for responsive utilization display
    static const double WINDOW_SEC = 10.0;
    double windowAge = -[_windowStart timeIntervalSinceNow];
    if (windowAge >= WINDOW_SEC) {
        // Roll the window: capture current window's utilization, reset
        _windowGpuSec = 0;
        _windowStart = [NSDate date];
        windowAge = 0;
    }
    // During a window, show utilization so far within this window
    if (windowAge < 0.5) {
        // Window just started — use all-time to avoid showing 0
        double totalWall = -[_statsStart timeIntervalSinceNow];
        if (totalWall < 0.1) return 0.0;
        return fmin(1.0, _totalGpuTimeSec / totalWall);
    }
    return fmin(1.0, _windowGpuSec / windowAge);
}

- (uint64_t)totalThreadsDispatched { return _totalThreads; }
- (uint64_t)totalBatchesDispatched { return _totalBatches; }

- (double)avgGpuTimeMs {
    if (_totalBatches == 0) return 0.0;
    return (_totalGpuTimeSec / _totalBatches) * 1000.0;
}

- (void)resetStats {
    _totalThreads = 0;
    _totalBatches = 0;
    _totalGpuTimeSec = 0;
    _totalWallTimeSec = 0;
    _statsStart = [NSDate date];
    _windowGpuSec = 0;
    _windowStart = [NSDate date];
}

@end
