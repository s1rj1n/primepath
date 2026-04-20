---
name: distributed-compute
description: Work with PrimePath's distributed computation engine across Metal GPU, CPU NEON, and multi-device networking. Use when modifying GPU shaders, load balancing, Conductor/Carriage networking, Nester Carry Chain math, or the three-gear auto-shifting engine.
user-invocable: true
argument-hint: [component] [action]
effort: high
---

# PrimePath Distributed Computation Engine

You are working on PrimePath's distributed prime search engine. This system coordinates computation across Metal GPU, CPU with ARM64 NEON, and multiple networked Macs.

## Architecture Overview

```
CONDUCTOR (master, port 9807, Bonjour)
  |-- Splits work via split_range() into WorkChunks
  |-- Sends WorkAssignment JSON to Carriages
  |-- Heartbeat 5s, timeout 10s, auto-reassign on disconnect
  |
CARRIAGE (worker, auto-discovers via Bonjour)
  |-- Receives WorkAssignment, runs local TaskManager
  |-- Reports Progress at 1Hz, DiscoveryReport immediately
  |-- Sends WorkDone on completion

LOCAL MACHINE (runs both Conductor + TaskManager)
  |
  |-- CPU Sieve Pipeline: wheel-210 + CRT + matrix filter + small prime test
  |-- GPU Metal: ring-buffered async dispatch (3 slots, 262K candidates/batch)
  |-- NEON Pre-filter: trial div by primes {3..47}, eliminates ~75% composites
  |-- Load Balancer: 50/50 GPU/CPU split, pool nudge, GPU throttle at 85%
  |-- Three-Gear Engine (Nester Carry Chain divisibility):
      Gear 1: CPU single-thread, 8-wide Barrett (<5K divisors)
      Gear 2: CPU 10-thread, 8-wide Barrett (5K-50K)
      Gear 3: GPU Metal, one thread per divisor (50K+)
```

## Key Files

| Component | Files |
|-----------|-------|
| **Metal GPU dispatch** | `PrimePath/MetalCompute.mm`, `MetalCompute.h` |
| **Metal shaders** | `PrimePath/PrimeShaders.metal` |
| **GPU backend abstraction** | `PrimePath/GPUBackend.hpp`, `GPUBackend.cpp` |
| **Load balancer + NEON** | `PrimePath/LoadBalancer.hpp`, `LoadBalancer.cpp` |
| **Nester Carry Chain math** | `PrimePath/PrimeEngine.hpp` (Barrett reduction, streaming divisibility) |
| **Task orchestration** | `PrimePath/TaskManager.hpp`, `TaskManager.mm` |
| **Conductor (master)** | `PrimePath/Network/ConductorServer.mm` |
| **Carriage (worker)** | `PrimePath/Network/CarriageClient.mm` |
| **Network protocol** | `PrimePath/Network/NetworkProtocol.hpp`, `WorkSplitter.hpp` |
| **PrimeNet/GIMPS** | `PrimePath/Network/PrimeNetClient.hpp`, `PrimeNetClient.mm` |
| **Shaders** | `PrimePath/PrimeShaders.metal` |
| **UI + orchestration** | `PrimePath/AppDelegate.mm` |

## GPU Specifics

- Ring buffer size 3: up to 2 batches in-flight, CPU processes batch N-1 while GPU runs batch N
- All buffers `MTLResourceStorageModeShared` (Apple Silicon unified memory, zero-copy)
- u128 arithmetic in Metal: custom `u128` struct with lo/hi u64 limbs, `mulhi` for 64x64->128
- Barrett reduction replaces division (O(8 muls) vs O(128 iterations))
- GPU pacing: 2ms min gap between dispatches to avoid starving CPU
- Mersenne TF kernel: fused sieve+modexp, 96-bit Barrett, one thread per candidate

## CPU/NEON Specifics

- Nester Carry Chain: modular multiplication without UDIV instruction
  - `q = floor(x * inv_d / 2^64)` via ARM64 UMULH
  - Precomputed reciprocal: `inv = UINT64_MAX / d`
  - Streaming divisibility: processes number segment-by-segment MSB to LSB
- NEON pre-filter: `uint64x2_t` lanes, 2 candidates at a time, ~30% survival rate
- Template batching: 1/2/4/8/16 divisors per pass, compile-time unroll at -O2

## Network Protocol

- Framing: 4-byte big-endian length prefix + JSON body
- Message types: AssignWork (0x01), Progress (0x10), DiscoveryReport (0x11), WorkDone (0x12), Ping/Pong (0x03/0x13), Hello (0x04)
- Work splitting: `split_range(type, start, end, num_workers)` creates equal WorkChunks
- Failover: dead carriages have work reassigned from last known position

## Task Types

Wieferich, Wall-Sun-Sun, Wilson, Twin, Sophie Germain, Cousin, Sexy, General, Emirp, Mersenne TF, Fermat Factor (11 types)

## Build Commands

```bash
# Debug build
xcodebuild -scheme PrimePath -configuration Debug build

# Release build
xcodebuild -scheme PrimePath -configuration Release build

# Release with notarization
./scripts/build-dmg.sh

# Unit tests
clang++ -std=c++17 -O2 -I. test_engine.cpp PrimePath/PrimeEngine.cpp -o test_engine -lpthread
./test_engine
```

## Guidelines

- All GPU work goes through GPUBackend abstraction. Never call Metal APIs directly from TaskManager.
- Load balancer is decoupled from search implementations. Query `advise()` for routing decisions.
- u128 math in Metal shaders must use the custom u128 struct, not compiler extensions.
- Barrett reduction must be used for all modular arithmetic in hot loops (no division).
- Ring buffer slot management is critical: always release previous slot before submitting new work.
- NEON pre-filter runs on every candidate batch regardless of gear selection.
- Conductor/Carriage messages are JSON over TCP with length-prefix framing.
- Test on both GPU and CPU-fallback paths when modifying compute kernels.
- Checkpoint files are written every 30s. Mersenne TF checkpoints use per-exponent filenames: `mersenne_tf_checkpoint_M{exponent}.txt`

When asked to work on: $ARGUMENTS
