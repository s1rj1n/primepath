#include "GPUBackend.hpp"

#ifdef __APPLE__
#import "MetalCompute.h"
#include <cstring>

namespace prime {

MetalBackend::MetalBackend() {
    MetalCompute *mc = [[MetalCompute alloc] init];
    _impl = (__bridge_retained void *)mc;
}

MetalBackend::~MetalBackend() {
    if (_impl) {
        MetalCompute *mc = (__bridge_transfer MetalCompute *)_impl;
        mc = nil;  // release
        _impl = nullptr;
    }
}

bool MetalBackend::available() const {
    if (!_impl) return false;
    MetalCompute *mc = (__bridge MetalCompute *)_impl;
    return [mc available];
}

std::string MetalBackend::name() const {
    return "Metal GPU (Apple Silicon)";
}

static int run_metal(void *impl, NSData *(^block)(MetalCompute *), uint8_t *results, uint32_t count) {
    if (!impl) return -1;
    @autoreleasepool {
        MetalCompute *mc = (__bridge MetalCompute *)impl;
        NSData *data = block(mc);
        if (!data) return -1;
        memcpy(results, data.bytes, count);
        int hits = 0;
        for (uint32_t i = 0; i < count; i++) if (results[i]) hits++;
        return hits;
    }
}

int MetalBackend::wieferich_batch(const uint64_t *primes, uint8_t *results, uint32_t count) {
    return run_metal(_impl, ^(MetalCompute *mc) {
        return [mc runWieferichBatch:primes count:count];
    }, results, count);
}

int MetalBackend::wallsunsun_batch(const uint64_t *primes, uint8_t *results, uint32_t count) {
    return run_metal(_impl, ^(MetalCompute *mc) {
        return [mc runWallSunSunBatch:primes count:count];
    }, results, count);
}

int MetalBackend::twin_batch(const uint64_t *cands, uint8_t *results, uint32_t count) {
    return run_metal(_impl, ^(MetalCompute *mc) {
        return [mc runTwinBatch:cands count:count];
    }, results, count);
}

int MetalBackend::sophie_batch(const uint64_t *cands, uint8_t *results, uint32_t count) {
    return run_metal(_impl, ^(MetalCompute *mc) {
        return [mc runSophieBatch:cands count:count];
    }, results, count);
}

int MetalBackend::cousin_batch(const uint64_t *cands, uint8_t *results, uint32_t count) {
    return run_metal(_impl, ^(MetalCompute *mc) {
        return [mc runCousinBatch:cands count:count];
    }, results, count);
}

int MetalBackend::sexy_batch(const uint64_t *cands, uint8_t *results, uint32_t count) {
    return run_metal(_impl, ^(MetalCompute *mc) {
        return [mc runSexyBatch:cands count:count];
    }, results, count);
}

int MetalBackend::primality_batch(const uint64_t *cands, uint8_t *results, uint32_t count) {
    return run_metal(_impl, ^(MetalCompute *mc) {
        return [mc runPrimalityBatch:cands count:count];
    }, results, count);
}

int MetalBackend::wilson_batch(const uint64_t *primes, uint8_t *results, uint32_t count) {
    return run_metal(_impl, ^(MetalCompute *mc) {
        return [mc runWilsonBatch:primes count:count];
    }, results, count);
}

int MetalBackend::wilson_segmented(uint64_t prime, uint32_t num_segments,
                                    uint64_t *partial_lo, uint64_t *partial_hi) {
    if (!_impl) return -1;
    @autoreleasepool {
        MetalCompute *mc = (__bridge MetalCompute *)_impl;
        BOOL ok = [mc runWilsonSegments:prime numSegments:num_segments
                              partialLo:partial_lo partialHi:partial_hi];
        return ok ? 0 : -1;
    }
}

int MetalBackend::mersenne_trial_batch(const uint32_t *candidates, uint8_t *results,
                                        uint32_t count, uint64_t exponent) {
    if (!_impl) return -1;
    @autoreleasepool {
        MetalCompute *mc = (__bridge MetalCompute *)_impl;
        NSData *data = [mc runMersenneTrialBatch:candidates count:count exponent:exponent];
        if (!data) return -1;
        memcpy(results, data.bytes, count);
        int hits = 0;
        for (uint32_t i = 0; i < count; i++) if (results[i]) hits++;
        return hits;
    }
}

int MetalBackend::fermat_factor_batch(const uint32_t *candidates, uint8_t *results,
                                       uint32_t count, uint64_t fermat_index) {
    if (!_impl) return -1;
    @autoreleasepool {
        MetalCompute *mc = (__bridge MetalCompute *)_impl;
        NSData *data = [mc runFermatFactorBatch:candidates count:count fermatIndex:fermat_index];
        if (!data) return -1;
        memcpy(results, data.bytes, count);
        int hits = 0;
        for (uint32_t i = 0; i < count; i++) if (results[i]) hits++;
        return hits;
    }
}

std::vector<GPUBackend::FusedHit> MetalBackend::mersenne_fused_sieve(
    uint64_t exponent, uint64_t k_start, uint64_t k_count) {
    if (!_impl) return {};
    @autoreleasepool {
        MetalCompute *mc = (__bridge MetalCompute *)_impl;
        NSArray *hits = [mc runMersenneFusedSieve:exponent kStart:k_start kCount:k_count];
        std::vector<FusedHit> result;
        for (NSArray *hit in hits) {
            FusedHit fh;
            fh.q_lo = [hit[0] unsignedLongLongValue];
            fh.q_hi_and_k = [hit[1] unsignedLongLongValue];
            result.push_back(fh);
        }
        return result;
    }
}

std::vector<uint32_t> MetalBackend::gpu_sieve(uint64_t lo, uint64_t odd_count,
                                                const uint64_t *sieve_primes, uint32_t num_primes) {
    if (!_impl) return {};
    @autoreleasepool {
        MetalCompute *mc = (__bridge MetalCompute *)_impl;
        NSData *data = [mc runGPUSieve:lo count:odd_count sievePrimes:sieve_primes numPrimes:num_primes];
        if (!data) return {};
        uint32_t words = (uint32_t)(data.length / sizeof(uint32_t));
        std::vector<uint32_t> bitmap(words);
        memcpy(bitmap.data(), data.bytes, data.length);
        return bitmap;
    }
}

double MetalBackend::gpu_utilization() const {
    if (!_impl) return 0.0;
    MetalCompute *mc = (__bridge MetalCompute *)_impl;
    return [mc gpuUtilization];
}

uint64_t MetalBackend::total_threads_dispatched() const {
    if (!_impl) return 0;
    MetalCompute *mc = (__bridge MetalCompute *)_impl;
    return [mc totalThreadsDispatched];
}

uint64_t MetalBackend::total_batches_dispatched() const {
    if (!_impl) return 0;
    MetalCompute *mc = (__bridge MetalCompute *)_impl;
    return [mc totalBatchesDispatched];
}

double MetalBackend::avg_gpu_time_ms() const {
    if (!_impl) return 0.0;
    MetalCompute *mc = (__bridge MetalCompute *)_impl;
    return [mc avgGpuTimeMs];
}

} // namespace prime
#endif
