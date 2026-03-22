#import "MetalCompute.h"
#include <cstdint>
#include <cstring>

// C bridge functions so TaskManager.cpp (pure C++) can call Metal
extern "C" {

int metal_wieferich_batch(void *mc, const uint64_t *primes, uint8_t *results, uint32_t count) {
    MetalCompute *compute = (__bridge MetalCompute *)mc;
    NSData *data = [compute runWieferichBatch:primes count:count];
    if (!data) return -1;
    memcpy(results, data.bytes, count);
    return 0;
}

int metal_wallsunsun_batch(void *mc, const uint64_t *primes, uint8_t *results, uint32_t count) {
    MetalCompute *compute = (__bridge MetalCompute *)mc;
    NSData *data = [compute runWallSunSunBatch:primes count:count];
    if (!data) return -1;
    memcpy(results, data.bytes, count);
    return 0;
}

int metal_twin_batch(void *mc, const uint64_t *cands, uint8_t *results, uint32_t count) {
    MetalCompute *compute = (__bridge MetalCompute *)mc;
    NSData *data = [compute runTwinBatch:cands count:count];
    if (!data) return -1;
    memcpy(results, data.bytes, count);
    return 0;
}

int metal_sophie_batch(void *mc, const uint64_t *cands, uint8_t *results, uint32_t count) {
    MetalCompute *compute = (__bridge MetalCompute *)mc;
    NSData *data = [compute runSophieBatch:cands count:count];
    if (!data) return -1;
    memcpy(results, data.bytes, count);
    return 0;
}

int metal_cousin_batch(void *mc, const uint64_t *cands, uint8_t *results, uint32_t count) {
    MetalCompute *compute = (__bridge MetalCompute *)mc;
    NSData *data = [compute runCousinBatch:cands count:count];
    if (!data) return -1;
    memcpy(results, data.bytes, count);
    return 0;
}

int metal_sexy_batch(void *mc, const uint64_t *cands, uint8_t *results, uint32_t count) {
    MetalCompute *compute = (__bridge MetalCompute *)mc;
    NSData *data = [compute runSexyBatch:cands count:count];
    if (!data) return -1;
    memcpy(results, data.bytes, count);
    return 0;
}

int metal_primality_batch(void *mc, const uint64_t *cands, uint8_t *results, uint32_t count) {
    MetalCompute *compute = (__bridge MetalCompute *)mc;
    NSData *data = [compute runPrimalityBatch:cands count:count];
    if (!data) return -1;
    memcpy(results, data.bytes, count);
    return 0;
}

}
