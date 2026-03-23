// MetalModArith.mm
// Implementation of the MetalModArith Objective-C wrapper.

#import "MetalModArith.h"
#import <Metal/Metal.h>

NSString * const MetalModArithErrorDomain = @"MetalModArithErrorDomain";

static const uint32_t MAX_BATCH = 262144; // 256K elements per dispatch

@implementation MetalModArith {
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;
    id<MTLLibrary> _library;

    id<MTLComputePipelineState> _modpowPSO;
    id<MTLComputePipelineState> _primalityPSO;
    id<MTLComputePipelineState> _mulmodPSO;
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

- (nullable instancetype)initWithError:(NSError **)error {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        NSLog(@"MetalModArith: no Metal-capable GPU device found");
        if (error) {
            *error = [NSError errorWithDomain:MetalModArithErrorDomain
                                         code:MetalModArithErrorNoDevice
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         @"No Metal-capable GPU device found."}];
        }
        return nil;
    }
    return [self initWithDevice:device metallibPath:nil error:error];
}

- (nullable instancetype)initWithDevice:(id)device
                           metallibPath:(nullable NSString *)path
                                  error:(NSError **)error {
    self = [super init];
    if (!self) return nil;

    _device = (id<MTLDevice>)device;

    _queue = [_device newCommandQueue];
    if (!_queue) {
        NSLog(@"MetalModArith: failed to create command queue");
        if (error) {
            *error = [NSError errorWithDomain:MetalModArithErrorDomain
                                         code:MetalModArithErrorNoDevice
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         @"Failed to create Metal command queue."}];
        }
        return nil;
    }

    // Load shader library
    NSError *libError = nil;
    if (path) {
        NSURL *url = [NSURL fileURLWithPath:path];
        _library = [_device newLibraryWithURL:url error:&libError];
    } else {
        // Try the main bundle first, then fall back to default library
        NSString *bundlePath = [[NSBundle mainBundle] pathForResource:@"MetalModArith"
                                                               ofType:@"metallib"];
        if (bundlePath) {
            NSURL *url = [NSURL fileURLWithPath:bundlePath];
            _library = [_device newLibraryWithURL:url error:&libError];
        } else {
            _library = [_device newDefaultLibrary];
        }
    }

    if (!_library) {
        NSLog(@"MetalModArith: failed to load shader library: %@",
              libError.localizedDescription ?: @"unknown error");
        if (error) {
            *error = [NSError errorWithDomain:MetalModArithErrorDomain
                                         code:MetalModArithErrorLibraryLoad
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         [NSString stringWithFormat:
                                             @"Failed to load Metal shader library: %@",
                                             libError.localizedDescription ?: @"unknown error"]}];
        }
        return nil;
    }

    // Create compute pipelines
    _modpowPSO    = [self _createPipeline:@"modpow_batch" error:error];
    _primalityPSO = [self _createPipeline:@"primality_batch" error:error];
    _mulmodPSO    = [self _createPipeline:@"mulmod_batch" error:error];

    if (!_modpowPSO || !_primalityPSO || !_mulmodPSO) {
        return nil;
    }

    NSLog(@"MetalModArith: ready on %@ | max threads/group: %lu",
          _device.name,
          (unsigned long)_modpowPSO.maxTotalThreadsPerThreadgroup);

    return self;
}

- (BOOL)available {
    return _device != nil && _library != nil &&
           _modpowPSO != nil && _primalityPSO != nil && _mulmodPSO != nil;
}

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

- (nullable id<MTLComputePipelineState>)_createPipeline:(NSString *)name
                                                  error:(NSError **)error {
    id<MTLFunction> fn = [_library newFunctionWithName:name];
    if (!fn) {
        NSLog(@"MetalModArith: kernel function '%@' not found in library", name);
        if (error) {
            *error = [NSError errorWithDomain:MetalModArithErrorDomain
                                         code:MetalModArithErrorPipelineCreation
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         [NSString stringWithFormat:
                                             @"Shader function '%@' not found in library.", name]}];
        }
        return nil;
    }

    NSError *psoError = nil;
    id<MTLComputePipelineState> pso = [_device newComputePipelineStateWithFunction:fn
                                                                            error:&psoError];
    if (!pso) {
        NSLog(@"MetalModArith: pipeline creation failed for '%@': %@",
              name, psoError.localizedDescription);
        if (error) {
            *error = [NSError errorWithDomain:MetalModArithErrorDomain
                                         code:MetalModArithErrorPipelineCreation
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         [NSString stringWithFormat:
                                             @"Failed to create pipeline for '%@': %@",
                                             name, psoError.localizedDescription]}];
        }
        return nil;
    }

    return pso;
}

// ---------------------------------------------------------------------------
// Internal dispatch helper
// ---------------------------------------------------------------------------

/// Encode and synchronously execute a compute kernel.
///
/// @param pso           The pipeline state to use.
/// @param inputBuffer   The input data buffer.
/// @param outputBuffer  The output data buffer (pre-allocated).
/// @param count         Number of work items.
/// @param error         On failure, describes what went wrong.
/// @return YES on success, NO on failure.
- (BOOL)_dispatchPSO:(id<MTLComputePipelineState>)pso
          inputBuffer:(id<MTLBuffer>)inputBuffer
         outputBuffer:(id<MTLBuffer>)outputBuffer
                count:(uint32_t)count
                error:(NSError **)error {

    // Count buffer
    id<MTLBuffer> countBuffer = [_device newBufferWithLength:sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];
    if (!countBuffer) {
        NSLog(@"MetalModArith: failed to allocate count buffer");
        if (error) {
            *error = [NSError errorWithDomain:MetalModArithErrorDomain
                                         code:MetalModArithErrorBufferAllocation
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         @"Failed to allocate count buffer."}];
        }
        return NO;
    }
    *((uint32_t *)countBuffer.contents) = count;

    id<MTLCommandBuffer> cmdBuf = [_queue commandBuffer];
    if (!cmdBuf) {
        NSLog(@"MetalModArith: failed to create command buffer");
        if (error) {
            *error = [NSError errorWithDomain:MetalModArithErrorDomain
                                         code:MetalModArithErrorDispatch
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         @"Failed to create command buffer."}];
        }
        return NO;
    }

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:inputBuffer  offset:0 atIndex:0];
    [encoder setBuffer:outputBuffer offset:0 atIndex:1];
    [encoder setBuffer:countBuffer  offset:0 atIndex:2];

    NSUInteger threadGroupSize = pso.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > 256) threadGroupSize = 256;

    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
    [encoder endEncoding];

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    if (cmdBuf.error) {
        NSLog(@"MetalModArith: GPU command buffer error: %@",
              cmdBuf.error.localizedDescription);
        if (error) {
            *error = [NSError errorWithDomain:MetalModArithErrorDomain
                                         code:MetalModArithErrorDispatch
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         [NSString stringWithFormat:
                                             @"GPU command buffer error: %@",
                                             cmdBuf.error.localizedDescription],
                                                NSUnderlyingErrorKey: cmdBuf.error}];
        }
        return NO;
    }

    return YES;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

- (nullable NSData *)modpowBatchWithBases:(const uint64_t *)bases
                                exponents:(const uint64_t *)exponents
                                  moduli:(const uint64_t *)moduli
                                   count:(uint32_t)count
                                   error:(NSError **)error {
    if (count == 0) return [NSData data];
    uint32_t batchCount = (count > MAX_BATCH) ? MAX_BATCH : count;

    // Pack input: [base, exp, mod] x count
    size_t inputSize = (size_t)batchCount * 3 * sizeof(uint64_t);
    id<MTLBuffer> inputBuf = [_device newBufferWithLength:inputSize
                                                  options:MTLResourceStorageModeShared];
    if (!inputBuf) {
        NSLog(@"MetalModArith: failed to allocate input buffer (%zu bytes)", inputSize);
        if (error) {
            *error = [NSError errorWithDomain:MetalModArithErrorDomain
                                         code:MetalModArithErrorBufferAllocation
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         @"Failed to allocate input buffer for modpow."}];
        }
        return nil;
    }

    uint64_t *packed = (uint64_t *)inputBuf.contents;
    for (uint32_t i = 0; i < batchCount; i++) {
        packed[i * 3 + 0] = bases[i];
        packed[i * 3 + 1] = exponents[i];
        packed[i * 3 + 2] = moduli[i];
    }

    // Output: [result_lo, result_hi] x count -- we return only lo for 64-bit moduli
    size_t outputSize = (size_t)batchCount * 2 * sizeof(uint64_t);
    id<MTLBuffer> outputBuf = [_device newBufferWithLength:outputSize
                                                   options:MTLResourceStorageModeShared];
    if (!outputBuf) {
        NSLog(@"MetalModArith: failed to allocate output buffer (%zu bytes)", outputSize);
        if (error) {
            *error = [NSError errorWithDomain:MetalModArithErrorDomain
                                         code:MetalModArithErrorBufferAllocation
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         @"Failed to allocate output buffer for modpow."}];
        }
        return nil;
    }
    memset(outputBuf.contents, 0, outputSize);

    if (![self _dispatchPSO:_modpowPSO inputBuffer:inputBuf
               outputBuffer:outputBuf count:batchCount error:error]) {
        return nil;
    }

    // Extract lo words (the full result for 64-bit moduli)
    NSMutableData *result = [NSMutableData dataWithLength:batchCount * sizeof(uint64_t)];
    uint64_t *out = (uint64_t *)outputBuf.contents;
    uint64_t *dst = (uint64_t *)result.mutableBytes;
    for (uint32_t i = 0; i < batchCount; i++) {
        dst[i] = out[i * 2]; // lo word
    }

    return result;
}

- (nullable NSData *)primalityBatchWithCandidates:(const uint64_t *)candidates
                                            count:(uint32_t)count
                                            error:(NSError **)error {
    if (count == 0) return [NSData data];
    uint32_t batchCount = (count > MAX_BATCH) ? MAX_BATCH : count;

    size_t inputSize = (size_t)batchCount * sizeof(uint64_t);
    id<MTLBuffer> inputBuf = [_device newBufferWithBytes:candidates
                                                  length:inputSize
                                                 options:MTLResourceStorageModeShared];
    if (!inputBuf) {
        NSLog(@"MetalModArith: failed to allocate input buffer (%zu bytes)", inputSize);
        if (error) {
            *error = [NSError errorWithDomain:MetalModArithErrorDomain
                                         code:MetalModArithErrorBufferAllocation
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         @"Failed to allocate input buffer for primality."}];
        }
        return nil;
    }

    size_t outputSize = (size_t)batchCount * sizeof(uint8_t);
    id<MTLBuffer> outputBuf = [_device newBufferWithLength:outputSize
                                                   options:MTLResourceStorageModeShared];
    if (!outputBuf) {
        NSLog(@"MetalModArith: failed to allocate output buffer (%zu bytes)", outputSize);
        if (error) {
            *error = [NSError errorWithDomain:MetalModArithErrorDomain
                                         code:MetalModArithErrorBufferAllocation
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         @"Failed to allocate output buffer for primality."}];
        }
        return nil;
    }
    memset(outputBuf.contents, 0, outputSize);

    if (![self _dispatchPSO:_primalityPSO inputBuffer:inputBuf
               outputBuffer:outputBuf count:batchCount error:error]) {
        return nil;
    }

    return [NSData dataWithBytes:outputBuf.contents length:outputSize];
}

- (nullable NSData *)mulmodBatchWithA:(const uint64_t *)aValues
                                    b:(const uint64_t *)bValues
                               moduli:(const uint64_t *)moduli
                                count:(uint32_t)count
                                error:(NSError **)error {
    if (count == 0) return [NSData data];
    uint32_t batchCount = (count > MAX_BATCH) ? MAX_BATCH : count;

    // Pack input: [a, b, mod] x count
    size_t inputSize = (size_t)batchCount * 3 * sizeof(uint64_t);
    id<MTLBuffer> inputBuf = [_device newBufferWithLength:inputSize
                                                  options:MTLResourceStorageModeShared];
    if (!inputBuf) {
        NSLog(@"MetalModArith: failed to allocate input buffer (%zu bytes)", inputSize);
        if (error) {
            *error = [NSError errorWithDomain:MetalModArithErrorDomain
                                         code:MetalModArithErrorBufferAllocation
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         @"Failed to allocate input buffer for mulmod."}];
        }
        return nil;
    }

    uint64_t *packed = (uint64_t *)inputBuf.contents;
    for (uint32_t i = 0; i < batchCount; i++) {
        packed[i * 3 + 0] = aValues[i];
        packed[i * 3 + 1] = bValues[i];
        packed[i * 3 + 2] = moduli[i];
    }

    size_t outputSize = (size_t)batchCount * sizeof(uint64_t);
    id<MTLBuffer> outputBuf = [_device newBufferWithLength:outputSize
                                                   options:MTLResourceStorageModeShared];
    if (!outputBuf) {
        NSLog(@"MetalModArith: failed to allocate output buffer (%zu bytes)", outputSize);
        if (error) {
            *error = [NSError errorWithDomain:MetalModArithErrorDomain
                                         code:MetalModArithErrorBufferAllocation
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         @"Failed to allocate output buffer for mulmod."}];
        }
        return nil;
    }
    memset(outputBuf.contents, 0, outputSize);

    if (![self _dispatchPSO:_mulmodPSO inputBuffer:inputBuf
               outputBuffer:outputBuf count:batchCount error:error]) {
        return nil;
    }

    return [NSData dataWithBytes:outputBuf.contents length:outputSize];
}

@end
