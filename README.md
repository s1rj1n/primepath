# PrimePath

[![Build & Test](https://github.com/s1rj1n/primepath/actions/workflows/build.yml/badge.svg)](https://github.com/s1rj1n/primepath/actions/workflows/build.yml)

Metal GPU prime search engine for Apple Silicon. Mersenne trial factoring, Fermat factor search, and a bunch of number theory tools.

Every other GPU prime tool (mfaktc, mfakto, GpuOwl, Genefer) is CUDA or OpenCL. None of them run on Apple GPUs. This one does.

## Download

Grab the latest build from [Releases](https://github.com/s1rj1n/primepath/releases). macOS 13+, Apple Silicon only. Extract the zip and run `PrimePath.app`.

## Searches

**Mersenne trial factoring** -- tests candidate factors q = 2kp + 1 against 2^p - 1 using 96-bit Barrett reduction on GPU. Same computation GIMPS runs before Lucas-Lehmer.

**Fermat factor search** -- factors of F_m = 2^(2^m) + 1. Candidates must be k * 2^(m+2) + 1, so we sieve k on CPU and test on GPU. New Fermat factors were found in 2024-2025 this way.

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

CPU sieve keeps all cores busy generating candidates while GPU crunches the current batch. CPU tests use Nester Carry Chain hardware mulmod for 4-7x faster 128-bit modular arithmetic (toggle via NesterCarryChain checkbox). Auto-falls back to binary doubling for moduli > 96 bits.

## Test Catalog

30+ tests in an external `TestCatalog.txt` -- edit it without recompiling. Primality tests (Miller-Rabin, Fermat, Lucas, Baillie-PSW, AKS), number theory (totient, Mobius, Goldbach, quadratic residues, Collatz), factoring (trial division, Pollard rho, ECM), crypto (RSA key gen), special primes (twins, Sophie Germain, Mersenne, safe primes, palindromic), sequences (gaps, pi(x), constellations), and factor seed prediction.

## Factor Seed Prediction

Experimental. Classifies prime factors into seed types (TinyPrime, SmallPrime, TwinFactor, SophieFactor, MersenneFactor, PowerFactor, DigitFactor, LargePrime) and looks for patterns:

- **Ring Beacon** -- mod-210 wheel beacon density, where factor seeds cluster at specific residues
- **Topography** -- factor count elevation mapping, depth-gap correlation
- **Factor Web** -- connectivity between factor seeds, gap prediction from shared structure
- **Audio Harmonic** -- harmonic resonance of factor frequencies, spectral gap detection
- **Twisting Tree** -- factor tree shapes (Linear/Bushy/Deep/Balanced) and how they change between consecutive primes

## Pipeline Builder

Build custom search pipelines by picking stages from five categories -- Sieve (Wheel-210, MatrixSieve, CRT, pseudoprime filter), Score (convergence, EvenShadow), Test (Miller-Rabin, GPU primality, Wieferich, Wilson, pair tests), Post (factoring, PinchFactor, Lucky7s, DivisorWeb), and Analysis (all the factor seed tools plus quadratic residue, Goldbach split, Euler totient). Each stage shows cost and rejection rate. Reorder and toggle as needed.

## Nester Carry Chain

Two division-free methods for modular arithmetic (S. Nester, 2026):

**Carry-chain modular multiplication** -- computes (a * b) mod m for operands up to 128 bits using ARM64 MUL+UMULH partial products assembled through a 192-bit carry chain. 4-7x faster than binary doubling. Falls back automatically for moduli > 96 bits.

**Streaming divisibility** -- tests whether an arbitrarily large number N is divisible by candidate divisors without division. Streams through N segment by segment, accumulating via Barrett reduction (precomputed reciprocal multiply). N-wide template batching processes up to 8 divisors per pass through the number. Up to 8x faster than scalar division on 2048-bit numbers. Over 10 million divisor tests per second on Apple Silicon.

| Method | Speedup | Where used |
|--------|---------|------------|
| Carry-chain mulmod | 4-7x vs binary doubling | Wieferich, Wall-Sun-Sun, GIMPS CPU verify, Miller-Rabin |
| Streaming 1x | 3x vs scalar divide | Single divisor trial division |
| Streaming 8-wide | 8x vs scalar divide | Bulk trial division (RSA-2048, big numbers) |
| Mersenne modpow | O(log p) | 2^p-1 divisibility without building the number |

Benchmark accessible from the main toolbar ("Bench") and Markov Predict window ("Nester-CarryChain Test").

## GIMPS

Talks directly to [mersenne.org](https://www.mersenne.org) over the PrimeNet v5 API. Register your machine, pull trial factoring assignments, run them on Metal GPU, and submit results back. GPU-found factors are independently verified on CPU using carry-chain modular exponentiation before submission. Reports system specs (chip, CPU/GPU cores, RAM) in results.

Result submissions use the PrimeNet JSON format (single-line JSON in the `&m=` parameter) with fields for timestamp (UTC), exponent, worktype, status, bit range, factors, program info, OS info, user, computer, AID, and hardware details. Local `results.json.txt` matches the submitted JSON exactly. First Metal GPU client for GIMPS.

### AutoPrimeNet

PrimePath also interoperates with [AutoPrimeNet](https://github.com/tdulcet/AutoPrimeNet), the recommended assignment handler used by all major third-party GIMPS clients (Mlucas, GpuOwl, PRPLL, mfaktc, mfakto, etc.). AutoPrimeNet handles the full PrimeNet v5 API (10,000+ lines of it) plus email notifications, log rotation, proxy support, stall monitoring, and automatic version checking.

To use AutoPrimeNet with PrimePath:

1. Install AutoPrimeNet and configure it as if running `mfaktc` or `mfakto` for TF assignments.
2. Point AutoPrimeNet at PrimePath's data directory (`~/Library/Application Support/PrimePath/`). It will populate `worktodo.txt` with `Factor=<AID>,<exponent>,<bitlo>,<bithi>` lines.
3. Run PrimePath. It will read assignments from `worktodo.txt`, run them on the Metal GPU, append JSON results to `results.json.txt`, and atomically remove completed lines from `worktodo.txt`.
4. AutoPrimeNet picks up `results.json.txt` and submits it to mersenne.org.

No PrimeNet registration on the PrimePath side is required in this mode. If you also register directly, both the in-memory assignment state and the `worktodo.txt` entry are cleaned up on successful submit (hybrid mode).

### JSON Result Editor

Built-in GUI editor for building, validating, and testing PrimeNet JSON results. Accessible from the GIMPS panel.

- Editable fields for every JSON result field (exponent, status, bit range, factors, user, computer, AID, program, kernel)
- **Validate** – checks JSON syntax, required fields, status/factor consistency, timestamp format, 2000-char limit
- **Load / Save Template** – reusable JSON configurations for different assignment types
- **Simulate Test** – loads a known result (M67 has factor 193707721) to verify format without a real assignment
- **Auto-Fill** – populates fields from system info, current assignment, and discoveries
- **Send to Server** – configurable server URL and credentials string, sends the JSON in the `&m=` parameter with full confirmation before submission
- Dual-pane output with editable JSON text and validation/server log

## Distributed Search

Splits work across multiple machines using a Conductor/Carriage architecture with Bonjour discovery. **Still being tested** -- if you have a couple of Macs and want to try it, open an issue.

## Metal Modular Arithmetic Library

Standalone in [`lib/MetalModArith/`](lib/MetalModArith/). First open-source Metal implementation of:

- 96-bit and 128-bit modular exponentiation (Barrett reduction)
- Hardware `mulhi` multi-precision multiply
- GPU Miller-Rabin with 12 deterministic witnesses
- Modular squaring chains

Same operations behind RSA, Diffie-Hellman, lattice crypto (CRYSTALS-Kyber, Dilithium), and ZK proofs. If you need fast modular arithmetic on a Mac GPU, this is it.

See [`lib/MetalModArith/README.md`](lib/MetalModArith/README.md) for API docs.

## How it works

```
CPU sieve pipeline          Metal GPU
==================          =========

  Wheel-210 sieve    --->   96-bit Barrett
  CRT rejection             modular squaring
  Small prime filter         (per thread, independent)
  Pack candidates
  Compute Barrett mu        2^p mod q  or  2^(2^m) mod q
```

Three uint32 limbs with hardware `mulhi` for the multiply-and-reduce step. Each GPU thread does a complete modular exponentiation independently -- no shared memory, no sync. Unified memory means zero copy between CPU sieve output and GPU input.

CPU-side 128-bit modular arithmetic uses the Nester Carry Chain: ARM64 `MUL`+`UMULH` for full 64x64->128-bit products, decomposed into a 192-bit intermediate, reduced via hardware `__int128` division in two 32-bit shifts. 4-7x faster than binary shift-and-add. Falls back to binary doubling automatically for moduli > 96 bits.

For big-number trial division, the streaming divisibility engine tests candidates via Barrett reduction (precomputed reciprocal multiply, no UDIV) with N-wide template batching. 8x faster than scalar division at 2048 bits.

| Layer | File | What it does |
|-------|------|-------------|
| Metal shaders | `PrimeShaders.metal` | u96/u128 Barrett arithmetic, all GPU kernels |
| Standalone library | `lib/MetalModArith/` | Reusable modular arithmetic |
| GPU dispatch | `MetalCompute.mm` | Ring-buffered command encoding, async pipelining |
| Backend abstraction | `GPUBackend.hpp` | Metal / CPU fallback |
| Search engine | `TaskManager.cpp` | CPU sieve, GPU batch dispatch, persistence |
| Prime engine | `PrimeEngine.hpp` | Miller-Rabin, CRT, wheel sieve, Pollard rho, Nester Carry Chain streaming |
| Analysis | `DataTools.hpp` | Factor seed prediction, Tonelli-Shanks |
| App | `AppDelegate.mm` | UI, test catalog, pipeline builder, GIMPS panel |

## Performance

| Machine | Mersenne TF | Primality (Miller-Rabin) | Sieve throughput |
|---------|-------------|-------------------------|-----------------|
| M1 Pro | ~800K candidates/s | ~1.2M tests/s | ~50M/s |
| M2 Max | ~1.4M candidates/s | ~2.1M tests/s | ~80M/s |
| M3 Max | ~1.8M candidates/s | ~2.6M tests/s | ~95M/s |

About 9 `mulhi` calls per squaring. Throughput scales linearly with GPU cores. CPU sieve does 1-5M primes/sec across all cores.

## Build

```bash
git clone https://github.com/s1rj1n/primepath.git
cd primepath
open PrimePath.xcodeproj
```

Xcode 15+, Metal support required.

### Tests

```bash
clang++ -std=c++17 -O2 -I. test_engine.cpp PrimePath/PrimeEngine.cpp -o test_engine -lpthread
./test_engine
```

### Signed DMG

```bash
xcrun notarytool store-credentials "PrimePath" \
    --apple-id YOUR_APPLE_ID \
    --team-id YOUR_TEAM_ID

./scripts/build-dmg.sh
```

## Contributing

Where help is needed most:

- **Distributed search testing** -- multi-machine coordination, result merging
- **Fermat sieve tuning** -- better candidate generation
- **128-bit / 192-bit Barrett on GPU** -- bigger factors via Metal shaders
- **GFN primality** -- NTT big integer multiplication on Metal
- **ECM on Metal** -- elliptic curve method with Montgomery arithmetic
- **Factor seed research** -- does seed classification actually predict anything?

Issues and PRs welcome.

## License

MIT

## Author

Sergei Nester -- snester@viewbuild.com
