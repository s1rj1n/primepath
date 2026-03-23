# MetalModArith

A standalone Metal GPU modular arithmetic library for Apple Silicon.

MetalModArith offloads batch modular arithmetic to the GPU via Metal compute
shaders. It is designed to be embedded in any macOS or iOS project that needs
high-throughput modular math on large batches of 64-bit integers.

## What it provides

- **u96/u128 Barrett arithmetic** -- full-width multiplication and Barrett
  reduction on the GPU, avoiding the precision loss of floating-point
  approaches. u96 uses 3x uint32 limbs; u128 uses 2x uint64 limbs.
- **Modular exponentiation** -- batch `base^exp mod m` using a binary
  square-and-multiply algorithm over u128 intermediates.
- **Miller-Rabin primality testing** -- deterministic primality for all 64-bit
  integers using 12 known-good witnesses, dispatched in GPU-parallel batches.
- **Modular multiplication** -- batch `(a * b) mod m` for 64-bit operands
  with exact u128 intermediate products via hardware `mulhi`.

An Objective-C wrapper (`MetalModArith`) handles device setup, shader loading,
pipeline creation, and synchronous batch dispatch.

## Use cases

- **Cryptographic implementations** -- RSA key generation, Diffie-Hellman
  parameter validation, and other protocols that require millions of modular
  exponentiations or primality tests. The GPU parallelism turns batch
  operations that take seconds on CPU into milliseconds.
- **Number theory research** -- sieving for primes with special properties
  (Wieferich, Wall-Sun-Sun, twin, Sophie Germain) or running large-scale
  computational experiments where GPU throughput matters.
- **Zero-knowledge proof systems** -- field arithmetic over large primes for
  protocols like Bulletproofs, STARKs, and Groth16 where modular multiply
  and exponentiation are core primitives.
- **Post-quantum lattice crypto prototyping** -- NTT (Number Theoretic
  Transform) building blocks for lattice-based schemes like Kyber and
  Dilithium, where batch modular multiply is a bottleneck worth GPU
  acceleration during parameter exploration.

## Limitations

- The Metal shader functions are not constant-time and should not be used for
  production cryptographic implementations where timing side-channels matter.
- Maximum operand sizes: u96 (96-bit), u128 (128-bit). For larger sizes
  (256-bit+), you would need to extend the library.
- Batch size is capped at 256K elements per dispatch.

## Quick start

```objc
#import "MetalModArith.h"

// Create the library (loads shaders, builds pipeline states).
NSError *error = nil;
MetalModArith *gpu = [[MetalModArith alloc] initWithError:&error];
if (!gpu) {
    NSLog(@"GPU init failed: %@", error);
    return;
}

// Batch modular exponentiation: base[i]^exp[i] mod mod[i]
uint64_t bases[]     = {2, 3, 5, 7};
uint64_t exponents[] = {100, 200, 300, 400};
uint64_t moduli[]    = {1000000007, 1000000007, 1000000007, 1000000007};

NSData *results = [gpu modpowBatchWithBases:bases
                                  exponents:exponents
                                    moduli:moduli
                                     count:4
                                     error:&error];
if (!results) {
    NSLog(@"modpow failed: %@", error);
    return;
}

const uint64_t *out = (const uint64_t *)results.bytes;
for (int i = 0; i < 4; i++) {
    NSLog(@"result[%d] = %llu", i, out[i]);
}

// Batch primality testing
uint64_t candidates[] = {2, 15, 17, 1000000007, 1000000009};
NSData *primeResults = [gpu primalityBatchWithCandidates:candidates
                                                  count:5
                                                  error:&error];
const uint8_t *flags = (const uint8_t *)primeResults.bytes;
for (int i = 0; i < 5; i++) {
    NSLog(@"%llu is %s", candidates[i], flags[i] ? "prime" : "composite");
}

// Batch modular multiplication: (a[i] * b[i]) mod m[i]
uint64_t aVals[] = {123456789, 987654321};
uint64_t bVals[] = {111111111, 222222222};
uint64_t mVals[] = {1000000007, 1000000007};

NSData *mulResults = [gpu mulmodBatchWithA:aVals
                                         b:bVals
                                    moduli:mVals
                                     count:2
                                     error:&error];
const uint64_t *mulOut = (const uint64_t *)mulResults.bytes;
for (int i = 0; i < 2; i++) {
    NSLog(@"mulmod[%d] = %llu", i, mulOut[i]);
}
```

## Xcode integration

1. **Add the source files** to your Xcode target:
   - `MetalModArith.h` -- public header
   - `MetalModArith.mm` -- Objective-C++ implementation
   - `MetalModArith.metal` -- GPU compute kernels

2. **Link the Metal framework.** In your target's Build Phases, add
   `Metal.framework` and `Foundation.framework` to "Link Binary With
   Libraries."

3. **Compile the Metal shaders.** Xcode compiles `.metal` files automatically
   when they are part of the target. The kernels will be available via
   `[device newDefaultLibrary]` at runtime.

4. **Import and use.** In any `.m` or `.mm` file:
   ```objc
   #import "MetalModArith.h"
   ```

5. **Alternative: precompiled metallib.** If you prefer to ship a precompiled
   shader library, compile it manually:
   ```sh
   xcrun -sdk macosx metal -c MetalModArith.metal -o MetalModArith.air
   xcrun -sdk macosx metallib MetalModArith.air -o MetalModArith.metallib
   ```
   Add the `.metallib` to your app bundle's resources, and the library will
   find it automatically. You can also pass an explicit path:
   ```objc
   MetalModArith *gpu = [[MetalModArith alloc]
       initWithDevice:MTLCreateSystemDefaultDevice()
         metallibPath:@"/path/to/MetalModArith.metallib"
                error:&error];
   ```

## Thread safety

`MetalModArith` instances are **not** thread-safe. Create one instance per
thread, or synchronize access externally.

## Error handling

All dispatch methods return `nil` on failure and populate the `NSError **`
parameter. Errors are also logged via `NSLog` for debugging. Error codes are
defined in the `MetalModArithError` enum:

| Code | Meaning |
|------|---------|
| `MetalModArithErrorNoDevice` | No Metal GPU available on this machine. |
| `MetalModArithErrorLibraryLoad` | Shader library (.metallib) failed to load. |
| `MetalModArithErrorPipelineCreation` | A kernel function was missing or the PSO failed to compile. |
| `MetalModArithErrorBufferAllocation` | Metal buffer allocation returned nil (out of GPU memory). |
| `MetalModArithErrorDispatch` | Command buffer execution failed on the GPU. |

## Files

| File | Description |
|------|-------------|
| `MetalModArith.metal` | GPU shader library with all arithmetic primitives and compute kernels |
| `MetalModArith.h` | Objective-C header for the wrapper class |
| `MetalModArith.mm` | Wrapper implementation: device setup, pipeline creation, batch dispatch |

## Using the shader primitives directly

If you need custom GPU kernels, include or copy the primitives from
`MetalModArith.metal` into your own `.metal` file and write kernels on top of
them. The existing kernels (`modpow_batch`, `primality_batch`, `mulmod_batch`)
show the pattern.
