# PrimePath

A prime number search engine that runs on Apple Silicon GPUs using Metal compute shaders.

This is the first Metal implementation for Mersenne trial factoring and Fermat factor searching. Every other GPU prime search tool out there (mfaktc, mfakto, GpuOwl, Genefer) targets CUDA or OpenCL. None of them run on Apple GPUs. PrimePath does.

## Download

Grab the latest signed and notarized DMG from [Releases](https://github.com/s1rj1n/primepath/releases). Requires macOS 13+ on Apple Silicon (M1/M2/M3/M4).

## What it does

### GPU-accelerated searches

**Mersenne trial factoring** tests candidate factors q = 2kp + 1 against Mersenne numbers 2^p - 1. This is the same computation that GIMPS uses to eliminate Mersenne candidates before running expensive Lucas-Lehmer tests. The GPU kernel does 96-bit Barrett modular exponentiation, computing 2^p mod q for thousands of candidates in parallel.

**Fermat factor search** looks for factors of Fermat numbers F_m = 2^(2^m) + 1. Any factor must have the form k * 2^(m+2) + 1, so we enumerate k values, sieve out composites on CPU, and test the survivors on GPU by computing 2^(2^m) mod q via repeated squaring. People found new Fermat factors in 2024 and 2025 doing exactly this kind of search.

### Other searches

- **Wieferich primes** - 2^(p-1) = 1 (mod p^2), only two known (1093 and 3511)
- **Wall-Sun-Sun primes** - p^2 divides F(p-(p/5)), none known
- **Wilson primes** - (p-1)! = -1 (mod p^2), only three known (5, 13, 563)
- **Twin, Sophie Germain, cousin, sexy primes** - pair searches
- **Emirps** - primes that are also prime when reversed

All searches use a sieve pipeline that keeps CPU cores busy generating candidates while the GPU processes the current batch.

## How it works

```
CPU sieve pipeline          Metal GPU
==================          =========

  Wheel-210 sieve    --->   96-bit Barrett
  CRT rejection             modular squaring
  Small prime filter         (per thread, independent)
  Pack candidates
  Compute Barrett mu        2^p mod q
                            or
                            2^(2^m) mod q
```

The 96-bit arithmetic uses three uint32 limbs with hardware `mulhi` for the multiply-and-reduce step. Each GPU thread tests one candidate completely independently, so it scales across all GPU cores. Apple Silicon's unified memory means there's zero copy cost between CPU sieve output and GPU input.

### Architecture

| Layer | File | What it does |
|-------|------|-------------|
| Metal shaders | `PrimeShaders.metal` | u96 Barrett arithmetic, trial factoring kernels |
| GPU dispatch | `MetalCompute.mm` | Buffer management, command encoding, timing |
| Backend abstraction | `GPUBackend.hpp` | Metal / CPU fallback interface |
| Search engine | `TaskManager.cpp` | CPU sieve, GPU batch dispatch, persistence |
| Prime engine | `PrimeEngine.hpp` | Miller-Rabin, CRT, wheel sieve, factoring |
| App | `AppDelegate.mm` | macOS UI, controls, stats display |

## Build from source

```bash
git clone https://github.com/s1rj1n/primepath.git
cd primepath
open PrimePath.xcodeproj
```

Build and run in Xcode. Requires Xcode 15+ with Metal support.

To run the engine tests without the app:

```bash
clang++ -std=c++17 -O2 -I. test_engine.cpp PrimePath/PrimeEngine.cpp -o test_engine -lpthread
./test_engine
```

## Build a signed DMG

If you have an Apple Developer account:

```bash
# One-time setup for notarization
xcrun notarytool store-credentials "PrimePath" \
    --apple-id YOUR_APPLE_ID \
    --team-id YOUR_TEAM_ID

# Build, sign, notarize, package
./scripts/build-dmg.sh
```

## Performance

On an M1 Pro, Mersenne trial factoring processes roughly 500K-1M candidate factors per second depending on exponent size. The Barrett kernel does about 9 hardware `mulhi` calls per modular squaring, and each thread runs independently with no shared memory or synchronization.

## Contributing

The biggest opportunities right now:

- **Fermat factor search tuning** - optimizing the sieve to generate better candidates
- **Larger arithmetic** - extending beyond 96-bit to handle bigger factors (128-bit or 192-bit Barrett)
- **GFN primality testing** - Generalized Fermat Number primes would need NTT-based big integer multiplication on Metal
- **ECM factoring** - Elliptic Curve Method on Metal would need a multi-precision Montgomery arithmetic library

Issues and PRs welcome.

## License

MIT
