// MetalModArith.h
// Standalone GPU modular arithmetic library for Apple Silicon.
//
// Provides batch modular exponentiation, primality testing, and modular
// multiplication using Metal compute shaders.

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Error domain for MetalModArith operations.
extern NSString * const MetalModArithErrorDomain;

/// Error codes.
typedef NS_ENUM(NSInteger, MetalModArithError) {
    MetalModArithErrorNoDevice = 1,
    MetalModArithErrorLibraryLoad,
    MetalModArithErrorPipelineCreation,
    MetalModArithErrorBufferAllocation,
    MetalModArithErrorDispatch,
};

/// A lightweight wrapper around Metal compute pipelines for modular arithmetic.
///
/// Thread safety: instances are NOT thread-safe. Create one per thread or
/// synchronize externally.
@interface MetalModArith : NSObject

/// Initialize with the default Metal device.
/// Returns nil and sets *error if Metal is unavailable or shaders fail to load.
- (nullable instancetype)initWithError:(NSError **)error;

/// Initialize with a specific Metal device and path to the compiled .metallib.
/// Pass nil for metallibPath to look for MetalModArith.metallib in the main bundle.
- (nullable instancetype)initWithDevice:(id)device
                           metallibPath:(nullable NSString *)path
                                  error:(NSError **)error;

/// Whether the library is ready to dispatch work.
@property (nonatomic, readonly) BOOL available;

// -------------------------------------------------------------------------
// Batch modular exponentiation
// -------------------------------------------------------------------------

/// Compute base[i]^exp[i] mod mod[i] for count elements.
///
/// @param bases      Array of uint64 base values (count elements).
/// @param exponents  Array of uint64 exponent values (count elements).
/// @param moduli     Array of uint64 modulus values (count elements).
/// @param count      Number of elements.
/// @param error      On failure, describes what went wrong.
/// @return Array of uint64 results (count elements), or nil on error.
- (nullable NSData *)modpowBatchWithBases:(const uint64_t *)bases
                                exponents:(const uint64_t *)exponents
                                  moduli:(const uint64_t *)moduli
                                   count:(uint32_t)count
                                   error:(NSError **)error;

// -------------------------------------------------------------------------
// Batch primality testing (deterministic Miller-Rabin)
// -------------------------------------------------------------------------

/// Test primality of count candidates.
///
/// @param candidates  Array of uint64 values to test.
/// @param count       Number of elements.
/// @param error       On failure, describes what went wrong.
/// @return NSData of uint8 results (1 = prime, 0 = composite), or nil on error.
- (nullable NSData *)primalityBatchWithCandidates:(const uint64_t *)candidates
                                            count:(uint32_t)count
                                            error:(NSError **)error;

// -------------------------------------------------------------------------
// Batch modular multiplication
// -------------------------------------------------------------------------

/// Compute (a[i] * b[i]) mod m[i] for count elements (64-bit operands).
///
/// @param aValues  Array of uint64 first operands (count elements).
/// @param bValues  Array of uint64 second operands (count elements).
/// @param moduli   Array of uint64 modulus values (count elements).
/// @param count    Number of elements.
/// @param error    On failure, describes what went wrong.
/// @return Array of uint64 results (count elements), or nil on error.
- (nullable NSData *)mulmodBatchWithA:(const uint64_t *)aValues
                                    b:(const uint64_t *)bValues
                               moduli:(const uint64_t *)moduli
                                count:(uint32_t)count
                                error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
