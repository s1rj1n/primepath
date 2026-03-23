# PrimePath

[![Build & Test](https://github.com/s1rj1n/primepath/actions/workflows/build.yml/badge.svg)](https://github.com/s1rj1n/primepath/actions/workflows/build.yml)

A prime number search engine, analysis toolkit, and GPU modular arithmetic library for Apple Silicon, using Metal compute shaders.

This is the first Metal implementation for Mersenne trial factoring and Fermat factor searching. Every other GPU prime search tool (mfaktc, mfakto, GpuOwl, Genefer) targets CUDA or OpenCL. None run on Apple GPUs. PrimePath does.

## Download

Grab the latest signed and notarized DMG from [Releases](https://github.com/s1rj1n/primepath/releases). Requires macOS 13+ on Apple Silicon (M1/M2/M3/M4).

## What it does

### GPU-accelerated searches

**Mersenne trial factoring** tests candidate factors q = 2kp + 1 against Mersenne numbers 2^p - 1. The GPU kernel does 96-bit Barrett modular exponentiation, computing 2^p mod q for thousands of candidates in parallel. This is the same computation GIMPS uses to eliminate candidates before expensive Lucas-Lehmer tests.

**Fermat factor search** looks for factors of Fermat numbers F_m = 2^(2^m) + 1. Any factor must have the form k * 2^(m+2) + 1, so we enumerate k values, sieve out composites on CPU, and test survivors on GPU. New Fermat factors were found in 2024 and 2025 doing exactly this.

### Other searches

| Search | What it tests | Known results |
|--------|--------------|---------------|
| Wieferich | 2^(p-1) = 1 (mod p^2) | Only 2 known: 1093, 3511 |
| Wall-Sun-Sun | p^2 divides F(p-(p/5)) | NONE known |
| Wilson | (p-1)! = -1 (mod p^2) | Only 3 known: 5, 13, 563 |
| Twin primes | p, p+2 both prime | Infinite (conjectured) |
| Sophie Germain | p, 2p+1 both prime | Infinite (conjectured) |
| Cousin primes | p, p+4 both prime | Infinite (conjectured) |
| Sexy primes | p, p+6 both prime | Infinite (conjectured) |
| Emirps | p and reverse(p) both prime | ~10% of primes |

All searches use a CPU sieve pipeline that keeps cores busy generating candidates while the GPU processes the current batch.

### Test Catalog

PrimePath includes a comprehensive test catalog with 30+ tests organized by category. The catalog is loaded from an external `TestCatalog.txt` file, making it easy to add or customize tests without recompiling. Each test has default parameters, a detailed description, and algorithm documentation. Categories include:

- **Primality** -- Miller-Rabin, Fermat, Lucas, Baillie-PSW, AKS
- **Number Theory** -- Euler's totient, Mobius function, Goldbach conjecture, quadratic residues, Collatz
- **Factoring** -- Trial division, Pollard rho, ECM
- **Cryptography** -- RSA key generation with random prime selection
- **Special Primes** -- Twin, Sophie Germain, Mersenne, safe primes, palindromic
- **Sequences** -- Prime gaps, prime counting (pi(x)), prime constellations
- **Factor Seed Prediction** -- Ring Beacon, Topography, Factor Web, Audio Harmonic, Twisting Tree

### Factor Seed Prediction

A set of experimental analysis tools that classify prime factors into seed types (TinyPrime, SmallPrime, TwinFactor, SophieFactor, MersenneFactor, PowerFactor, DigitFactor, LargePrime) and look for predictive patterns in their distribution:

| Analysis | What it does |
|----------|-------------|
| Ring Beacon | Mod-210 wheel beacon density scoring -- measures how factor seeds cluster at specific residue positions |
| Topography | Factor count elevation mapping and depth-gap correlation analysis |
| Factor Web | Connectivity analysis between factor seeds, gap prediction via shared factor structure |
| Audio Harmonic | Harmonic resonance scoring of factor frequencies, gap detection through spectral analysis |
| Twisting Tree | Factor tree shape classification (Linear, Bushy, Deep, Balanced) and shape transition analysis between consecutive primes |

### Search Pipeline Builder

A configurable pipeline builder window lets you construct custom search and analysis pipelines by combining stages from five categories:

- **Sieve** -- Wheel-210, MatrixSieve (NEON), CRT filter, pseudoprime filter
- **Score** -- Convergence scoring, EvenShadow p+/-1 analysis
- **Test** -- Miller-Rabin, GPU primality, Wieferich, Wilson, twin/cousin/sexy pair, Sophie Germain, emirp
- **Post** -- Full factoring, PinchFactor, Lucky7s, DivisorWeb
- **Analysis** -- Ring Beacon, Topography, Factor Web, Audio Harmonic, Twisting Tree, Quadratic Residue, Goldbach Split, Euler Totient

Each stage shows estimated cost (ms per 1M candidates) and rejection percentage. Stages can be reordered and enabled/disabled individually. The stage list is scrollable to accommodate all available stages.

### GIMPS Integration

PrimePath connects directly to [mersenne.org](https://www.mersenne.org) (GIMPS -- Great Internet Mersenne Prime Search) via the PrimeNet v5 API. From the GIMPS panel you can:

- **Register** your machine with your mersenne.org account
- **Get work** -- fetch trial factoring assignments from the server
- **Run assignments** -- automatically configure the Metal GPU Mersenne TF engine for the assigned exponent and bit range
- **Submit results** -- report factors found (or no-factor) back to mersenne.org
- **Local results file** -- all results are also written to `results.json.txt` in mfaktc-compatible JSON format

This makes PrimePath the first Metal GPU client that can contribute to GIMPS trial factoring.

### Distributed Search (Needs Testing)

PrimePath includes distributed search capabilities that allow splitting large search ranges across multiple machines. This feature is implemented but **needs further testing** before it can be considered production-ready. If you have access to multiple Apple Silicon machines and want to help test distributed search, please open an issue or reach out.

## Metal Modular Arithmetic Library

The GPU arithmetic primitives are available as a standalone library in [`lib/MetalModArith/`](lib/MetalModArith/). This is the first open-source Metal implementation of:

- **96-bit and 128-bit modular exponentiation** using Barrett reduction
- **Hardware-accelerated `mulhi`** for multi-precision multiply
- **GPU Miller-Rabin primality testing** with 12 deterministic witnesses
- **Modular squaring chains** for repeated-squaring algorithms

These are the same operations used in RSA, Diffie-Hellman, post-quantum lattice cryptography (CRYSTALS-Kyber, Dilithium), and zero-knowledge proof systems. If you're building anything on Mac that needs fast modular arithmetic on GPU, this is the only Metal library that does it.

See [`lib/MetalModArith/README.md`](lib/MetalModArith/README.md) for usage and API docs.

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

The 96-bit arithmetic uses three uint32 limbs with hardware `mulhi` for the multiply-and-reduce step. Each GPU thread tests one candidate independently -- no shared memory or synchronization. Apple Silicon's unified memory means zero copy cost between CPU sieve output and GPU input.

### Architecture

| Layer | File | What it does |
|-------|------|-------------|
| Metal shaders | `PrimeShaders.metal` | u96/u128 Barrett arithmetic, all GPU kernels |
| Standalone library | `lib/MetalModArith/` | Reusable modular arithmetic for any project |
| GPU dispatch | `MetalCompute.mm` | Ring-buffered command encoding, async pipelining |
| Backend abstraction | `GPUBackend.hpp` | Metal / CPU fallback interface |
| Search engine | `TaskManager.cpp` | CPU sieve, GPU batch dispatch, persistence |
| Prime engine | `PrimeEngine.hpp` | Miller-Rabin, CRT, wheel sieve, Pollard rho |
| Data tools | `DataTools.hpp` | Factor seed prediction, Tonelli-Shanks, analysis algorithms |
| App | `AppDelegate.mm` | macOS UI, test catalog, pipeline builder, stats display |

## Performance

Benchmarks on Apple Silicon (single GPU, no CPU fallback):

| Machine | Mersenne TF | Primality (Miller-Rabin) | Sieve throughput |
|---------|-------------|-------------------------|-----------------|
| M1 Pro | ~800K candidates/s | ~1.2M tests/s | ~50M/s |
| M2 Max | ~1.4M candidates/s | ~2.1M tests/s | ~80M/s |
| M3 Max | ~1.8M candidates/s | ~2.6M tests/s | ~95M/s |

The Barrett kernel does about 9 hardware `mulhi` calls per modular squaring. Each GPU thread runs a complete modular exponentiation independently, so throughput scales linearly with GPU core count.

### CPU engine performance

The CPU sieve + Miller-Rabin pipeline processes 1-5M primes/second depending on range, using all available cores with a wheel-210 sieve and CRT pre-filter.

## Build from source

```bash
git clone https://github.com/s1rj1n/primepath.git
cd primepath
open PrimePath.xcodeproj
```

Build and run in Xcode. Requires Xcode 15+ with Metal support.

### Run the test suite

```bash
clang++ -std=c++17 -O2 -I. test_engine.cpp PrimePath/PrimeEngine.cpp -o test_engine -lpthread
./test_engine
```

Tests cover primality, modular arithmetic, sieving, factoring, and verification of all known Wieferich/Wilson primes.

### Build a signed DMG

If you have an Apple Developer account:

```bash
# One-time setup for notarization
xcrun notarytool store-credentials "PrimePath" \
    --apple-id YOUR_APPLE_ID \
    --team-id YOUR_TEAM_ID

# Build, sign, notarize, package
./scripts/build-dmg.sh
```

## Contributing

The biggest opportunities:

- **Distributed search testing** -- help validate multi-machine search coordination and result merging
- **Fermat factor search tuning** -- optimizing the sieve to generate better candidates
- **Larger arithmetic** -- extending beyond 96-bit to handle bigger factors (128-bit or 192-bit Barrett)
- **GFN primality testing** -- Generalized Fermat Number primes via NTT-based big integer multiplication on Metal
- **ECM factoring** -- Elliptic Curve Method on Metal with multi-precision Montgomery arithmetic
- **Montgomery multiplication** -- alternative to Barrett for modular reduction, potentially faster for fixed moduli
- **Factor seed research** -- analyzing whether factor seed classification can improve prime prediction

Issues and PRs welcome.

## License

MIT

## Author

Sergei Nester -- snester@viewbuild.com
