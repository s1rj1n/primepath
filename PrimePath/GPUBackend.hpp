#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace prime {

// ═══════════════════════════════════════════════════════════════════════
// GPUBackend — abstract interface for GPU compute
//
// Implementations:
//   MetalBackend (macOS)    — PrimeShaders.metal
//   VulkanBackend (Windows) — future: PrimeShaders.comp (SPIR-V)
//   CUDABackend (NVIDIA)    — future: PrimeShaders.cu
//   CPUBackend (fallback)   — pure C++ fallback on any platform
// ═══════════════════════════════════════════════════════════════════════

class GPUBackend {
public:
    virtual ~GPUBackend() = default;

    // Is this backend available and ready?
    virtual bool available() const = 0;

    // Backend name for display
    virtual std::string name() const = 0;

    // ── Batch tests ──────────────────────────────────────────────────
    // Input: array of u64 primes/candidates
    // Output: array of u8 results (1=hit, 0=miss)
    // Returns number of hits found, or -1 on error

    virtual int wieferich_batch(const uint64_t *primes, uint8_t *results, uint32_t count) = 0;
    virtual int wallsunsun_batch(const uint64_t *primes, uint8_t *results, uint32_t count) = 0;
    virtual int twin_batch(const uint64_t *cands, uint8_t *results, uint32_t count) = 0;
    virtual int sophie_batch(const uint64_t *cands, uint8_t *results, uint32_t count) = 0;
    virtual int cousin_batch(const uint64_t *cands, uint8_t *results, uint32_t count) = 0;
    virtual int sexy_batch(const uint64_t *cands, uint8_t *results, uint32_t count) = 0;
    virtual int primality_batch(const uint64_t *cands, uint8_t *results, uint32_t count) = 0;

    // Wilson: batch (small primes, one thread each)
    virtual int wilson_batch(const uint64_t *primes, uint8_t *results, uint32_t count) = 0;

    // Wilson: segmented factorial for one large prime.
    // Returns num_segments partial products (lo/hi pairs) in partial_lo/partial_hi.
    // CPU combines them. Returns 0 on success, -1 on error.
    virtual int wilson_segmented(uint64_t prime, uint32_t num_segments,
                                 uint64_t *partial_lo, uint64_t *partial_hi) = 0;

    // Mersenne trial factoring: test if candidates divide 2^exponent - 1.
    // candidates: packed array of 6× uint32 per entry (q + Barrett constant).
    // results: 1 if factor found, 0 otherwise. Returns hit count or -1.
    virtual int mersenne_trial_batch(const uint32_t *candidates, uint8_t *results,
                                     uint32_t count, uint64_t exponent) = 0;

    // Fermat factor search: test if candidates divide F_m = 2^(2^m) + 1.
    // Same packed format as mersenne_trial_batch.
    virtual int fermat_factor_batch(const uint32_t *candidates, uint8_t *results,
                                    uint32_t count, uint64_t fermat_index) = 0;

    // GPU-accelerated segmented sieve: returns bitmap of composites for odd numbers
    // in [lo, lo + odd_count*2). Bit i set = lo + 2*i + 1 is composite.
    // sieve_primes: array of primes from 3..sqrt(hi). Returns bitmap data.
    virtual std::vector<uint32_t> gpu_sieve(uint64_t lo, uint64_t odd_count,
                                             const uint64_t *sieve_primes, uint32_t num_primes) {
        return {}; // default: not supported
    }

    // Fused Mersenne sieve+test: entire pipeline on GPU.
    // Returns vector of (q_lo, q_hi_and_k) pairs for any factors found.
    struct FusedHit { uint64_t q_lo; uint64_t q_hi_and_k; };
    virtual std::vector<FusedHit> mersenne_fused_sieve(uint64_t exponent,
                                                        uint64_t k_start,
                                                        uint64_t k_count) {
        return {}; // default: not supported
    }

    // ── Performance stats ────────────────────────────────────────────
    virtual double gpu_utilization() const { return 0.0; }     // 0.0-1.0
    virtual uint64_t total_threads_dispatched() const { return 0; }
    virtual uint64_t total_batches_dispatched() const { return 0; }
    virtual double avg_gpu_time_ms() const { return 0.0; }
};

// ── CPU fallback (always available, any platform) ───────────────────

class CPUBackend : public GPUBackend {
public:
    bool available() const override { return true; }
    std::string name() const override { return "CPU (fallback)"; }

    int wieferich_batch(const uint64_t *primes, uint8_t *results, uint32_t count) override;
    int wallsunsun_batch(const uint64_t *primes, uint8_t *results, uint32_t count) override;
    int twin_batch(const uint64_t *cands, uint8_t *results, uint32_t count) override;
    int sophie_batch(const uint64_t *cands, uint8_t *results, uint32_t count) override;
    int cousin_batch(const uint64_t *cands, uint8_t *results, uint32_t count) override;
    int sexy_batch(const uint64_t *cands, uint8_t *results, uint32_t count) override;
    int primality_batch(const uint64_t *cands, uint8_t *results, uint32_t count) override;
    int wilson_batch(const uint64_t *primes, uint8_t *results, uint32_t count) override;
    int wilson_segmented(uint64_t prime, uint32_t num_segments,
                         uint64_t *partial_lo, uint64_t *partial_hi) override;
    int mersenne_trial_batch(const uint32_t *candidates, uint8_t *results,
                             uint32_t count, uint64_t exponent) override;
    int fermat_factor_batch(const uint32_t *candidates, uint8_t *results,
                            uint32_t count, uint64_t fermat_index) override;
};

#ifdef __APPLE__
// ── Metal backend (macOS / Apple Silicon) ───────────────────────────

class MetalBackend : public GPUBackend {
public:
    MetalBackend();
    ~MetalBackend() override;

    bool available() const override;
    std::string name() const override;

    int wieferich_batch(const uint64_t *primes, uint8_t *results, uint32_t count) override;
    int wallsunsun_batch(const uint64_t *primes, uint8_t *results, uint32_t count) override;
    int twin_batch(const uint64_t *cands, uint8_t *results, uint32_t count) override;
    int sophie_batch(const uint64_t *cands, uint8_t *results, uint32_t count) override;
    int cousin_batch(const uint64_t *cands, uint8_t *results, uint32_t count) override;
    int sexy_batch(const uint64_t *cands, uint8_t *results, uint32_t count) override;
    int primality_batch(const uint64_t *cands, uint8_t *results, uint32_t count) override;
    int wilson_batch(const uint64_t *primes, uint8_t *results, uint32_t count) override;
    int wilson_segmented(uint64_t prime, uint32_t num_segments,
                         uint64_t *partial_lo, uint64_t *partial_hi) override;
    int mersenne_trial_batch(const uint32_t *candidates, uint8_t *results,
                             uint32_t count, uint64_t exponent) override;
    int fermat_factor_batch(const uint32_t *candidates, uint8_t *results,
                            uint32_t count, uint64_t fermat_index) override;
    std::vector<FusedHit> mersenne_fused_sieve(uint64_t exponent,
                                                uint64_t k_start,
                                                uint64_t k_count) override;
    std::vector<uint32_t> gpu_sieve(uint64_t lo, uint64_t odd_count,
                                     const uint64_t *sieve_primes, uint32_t num_primes) override;

    // Performance stats from Metal command buffer timing
    double gpu_utilization() const override;
    uint64_t total_threads_dispatched() const override;
    uint64_t total_batches_dispatched() const override;
    double avg_gpu_time_ms() const override;

private:
    void *_impl; // MetalCompute* (opaque for C++)
};
#endif

// ── Factory: create best available backend ──────────────────────────

inline GPUBackend* create_best_backend() {
#ifdef __APPLE__
    auto *metal = new MetalBackend();
    if (metal->available()) return metal;
    delete metal;
#endif
    // Future: #ifdef _WIN32 → try VulkanBackend
    // Future: #ifdef __CUDA__ → try CUDABackend
    return new CPUBackend();
}

} // namespace prime
