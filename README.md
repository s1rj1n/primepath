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

Two division-free methods for modular arithmetic (S. Nester, 2026), with a three-gear auto-shifting engine that picks the fastest execution path.

**Carry-chain modular multiplication** -- computes (a * b) mod m for operands up to 128 bits using ARM64 MUL+UMULH partial products assembled through a 192-bit carry chain. 4-7x faster than binary doubling. Falls back automatically for moduli > 96 bits.

**Streaming divisibility** -- tests whether an arbitrarily large number N is divisible by candidate divisors without division. Streams through N segment by segment, accumulating via Barrett reduction (precomputed reciprocal multiply). N-wide template batching processes up to 8 divisors per pass through the number. Three gears auto-selected by work volume:

| Gear | Engine | Best for | Speedup vs single-thread |
|------|--------|----------|--------------------------|
| 1 | CPU single-thread, 8-wide Barrett | < 5K divisors | baseline (8x vs scalar divide) |
| 2 | CPU 10-thread, each 8-wide Barrett | 5K-50K divisors | up to 4x |
| 3 | GPU Metal, one thread per divisor | 50K+ divisors | up to 7x |

Gear selection is calibrated from a 5x5 benchmark matrix (128-32768 bit numbers, 100-200K divisors). The auto selector matches the actual fastest gear in all 25 test cases. For very large numbers (32768+ bits), GPU wins starting at 1K divisors.

| Method | Speedup | Where used |
|--------|---------|------------|
| Carry-chain mulmod | 4-7x vs binary doubling | Wieferich, Wall-Sun-Sun, GIMPS CPU verify, Miller-Rabin |
| Streaming 8-wide | 8x vs scalar divide | Bulk trial division (RSA-2048, big numbers) |
| Three-gear auto | up to 7x over single CPU | Large-scale divisibility testing |
| Mersenne modpow | O(log p) | 2^p-1 divisibility without building the number |

Benchmark accessible from the main toolbar ("Bench") and Markov Predict window ("Nester-CarryChain Test").

## GIMPS

All server communication with [mersenne.org](https://www.mersenne.org) is routed through [AutoPrimeNet](https://github.com/tdulcet/AutoPrimeNet), the recommended assignment handler used by all major third-party GIMPS clients (Mlucas, GpuOwl, PRPLL, mfaktc, mfakto, etc.). PrimePath does not talk to the PrimeNet API directly. AutoPrimeNet handles assignment management, result submission, email notifications, log rotation, proxy support, stall monitoring, and version checking.

GPU-found factors are independently verified on CPU using carry-chain modular exponentiation before reporting. Composite factors are automatically split via trial division + Pollard rho before submission. Results use the PrimeNet JSON format with CRC32 checksum, and support known-factor lists (composite factor stripping, continue-after-factor).

### Setup

1. Install AutoPrimeNet and configure it for TF assignments.
2. Point AutoPrimeNet at PrimePath's data directory (`~/Library/Application Support/PrimePath/`). It will populate `worktodo.txt` with `Factor=<AID>,<exponent>,<bitlo>,<bithi>` lines.
3. Run PrimePath. It reads assignments from `worktodo.txt`, runs them on the Metal GPU, appends JSON results to `results.json.txt`, and atomically removes completed lines from `worktodo.txt`.
4. AutoPrimeNet picks up `results.json.txt` and submits to mersenne.org.

Known-factor support: `Factor=AID,exp,lo,hi,"factor1,factor2"`. If a discovered factor is composed entirely of known primes it is discarded. Composite factors have known components stripped before reporting. The default is to always complete the full bitlevel (configurable).

### JSON Result Editor

Built-in GUI editor for building, validating, and testing PrimeNet JSON results. Accessible from the GIMPS panel.

- Editable fields for every JSON result field (exponent, status, bit range, factors, user, computer, AID, program, kernel)
- **Validate** – checks JSON syntax, required fields, status/factor consistency, timestamp format, 2000-char limit
- **Load / Save Template** – reusable JSON configurations for different assignment types
- **Simulate Test** – loads a known result (M67 has factor 193707721) to verify format without a real assignment
- **Auto-Fill** – populates fields from system info, current assignment, and discoveries
- **Send to Server** – configurable server URL and credentials string, sends the JSON in the `&m=` parameter with full confirmation before submission
- Dual-pane output with editable JSON text and validation/server log
- **JSON sample button** -- generates real JSON from `build_result_json` for format verification

### JSON Format

Hardware fields adapt to the system architecture:

| System | chip field | core fields | RAM field |
|--------|-----------|-------------|-----------|
| Unified SoC (Apple Silicon) | `chip` | `cpu_p_cores`, `cpu_e_cores`, `gpu_cores` | `ram_gb` |
| Discrete GPU | `cpu_chip`, `gpu_chip` | `cpu_cores`, `gpu_cores` | `cpu_ram_gb`, `gpu_ram_gb` |

Example (factor found, Apple M5):
```json
{"timestamp":"2026-04-13 23:45:29","exponent":67,"worktype":"TF","status":"F",
 "bitlo":27,"bithi":28,"rangecomplete":false,"factors":["193707721"],
 "program":{"name":"PrimePath","version":"1.3.0","kernel":"Metal96bit"},
 "os":{"os":"macOS","version":"25.3.0","architecture":"ARM_64"},
 "user":"s1rj1n","computer":"Sergeis-MacBook-Pro.local",
 "aid":"A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4",
 "hardware":{"chip":"Apple M5","cpu_p_cores":4,"cpu_e_cores":6,
  "gpu_cores":10,"ram_gb":24},
 "checksum":{"version":1,"checksum":"5EF5EF6C"}}
```

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

For big-number trial division, the three-gear streaming engine auto-selects CPU single-thread (Gear 1), CPU multi-thread (Gear 2), or GPU Metal (Gear 3) based on divisor count and number size. All gears use Barrett reduction (precomputed reciprocal multiply, no UDIV). Up to 8x over scalar on CPU, up to 7x more on GPU for large workloads.

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
