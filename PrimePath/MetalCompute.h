#pragma once
#import <Foundation/Foundation.h>

// ═══════════════════════════════════════════════════════════════════════
// Metal GPU compute wrapper — dispatches batch primality tests to GPU
// ═══════════════════════════════════════════════════════════════════════

@interface MetalCompute : NSObject

- (instancetype)init;
- (BOOL)available;

// GPU performance stats
- (double)gpuUtilization;       // 0.0-1.0 fraction of time GPU was busy
- (uint64_t)totalThreadsDispatched;
- (uint64_t)totalBatchesDispatched;
- (double)avgGpuTimeMs;         // average GPU execution time per batch (ms)
- (void)resetStats;

// Batch tests — input: array of primes/candidates, output: array of booleans (1=hit)
- (NSData *)runWieferichBatch:(const uint64_t *)primes count:(uint32_t)count;
- (NSData *)runWallSunSunBatch:(const uint64_t *)primes count:(uint32_t)count;
- (NSData *)runTwinBatch:(const uint64_t *)candidates count:(uint32_t)count;
- (NSData *)runSophieBatch:(const uint64_t *)candidates count:(uint32_t)count;
- (NSData *)runCousinBatch:(const uint64_t *)candidates count:(uint32_t)count;
- (NSData *)runSexyBatch:(const uint64_t *)candidates count:(uint32_t)count;
- (NSData *)runPrimalityBatch:(const uint64_t *)candidates count:(uint32_t)count;

// Wilson
- (NSData *)runWilsonBatch:(const uint64_t *)primes count:(uint32_t)count;
- (BOOL)runWilsonSegments:(uint64_t)prime
              numSegments:(uint32_t)numSegments
                partialLo:(uint64_t *)outLo
                partialHi:(uint64_t *)outHi;

// Mersenne trial factoring
- (NSData *)runMersenneTrialBatch:(const uint32_t *)candidates
                            count:(uint32_t)count
                         exponent:(uint64_t)p;

// Fermat factor search
- (NSData *)runFermatFactorBatch:(const uint32_t *)candidates
                           count:(uint32_t)count
                     fermatIndex:(uint64_t)m;

@end
