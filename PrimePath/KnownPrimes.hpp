#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include "PrimeEngine.hpp"

// ═══════════════════════════════════════════════════════════════════════
// KnownPrimes — built-in database of notable primes and pseudoprimes
// so we never waste cycles recalculating what's already established.
// ═══════════════════════════════════════════════════════════════════════

namespace prime {

enum class KnownClass {
    MersennePrime,     // 2^p - 1
    WieferichPrime,    // 2^(p-1) ≡ 1 (mod p²)
    WilsonPrime,       // (p-1)! ≡ -1 (mod p²)
    TwinPrime,         // p and p+2 both prime
    SophieGermain,     // p and 2p+1 both prime
    FermatPrime,       // 2^(2^n) + 1
    WagstaffPrime,     // (2^p + 1) / 3
    Factorial,         // n! ± 1
    Primorial,         // p# ± 1
    Palindromic,       // reads same forwards/backwards
    Repunit,           // (10^n - 1) / 9
    NotablePrime,      // other mathematically notable primes
    CarmichaelNumber,  // pseudoprime to all coprime bases (Fermat)
    StrongPseudoprime, // passes strong probable-prime test for specific bases
    EulerPseudoprime,  // Euler pseudoprime
};

struct KnownEntry {
    uint64_t value;
    KnownClass kclass;
    const char* description;
};

inline const char* known_class_name(KnownClass c) {
    switch (c) {
        case KnownClass::MersennePrime:     return "Mersenne Prime";
        case KnownClass::WieferichPrime:    return "Wieferich Prime";
        case KnownClass::WilsonPrime:       return "Wilson Prime";
        case KnownClass::TwinPrime:         return "Twin Prime";
        case KnownClass::SophieGermain:     return "Sophie Germain Prime";
        case KnownClass::FermatPrime:       return "Fermat Prime";
        case KnownClass::WagstaffPrime:     return "Wagstaff Prime";
        case KnownClass::Factorial:         return "Factorial Prime";
        case KnownClass::Primorial:         return "Primorial Prime";
        case KnownClass::Palindromic:       return "Palindromic Prime";
        case KnownClass::Repunit:           return "Repunit Prime";
        case KnownClass::NotablePrime:      return "Notable Prime";
        case KnownClass::CarmichaelNumber:  return "Carmichael Number";
        case KnownClass::StrongPseudoprime: return "Strong Pseudoprime";
        case KnownClass::EulerPseudoprime:  return "Euler Pseudoprime";
    }
    return "Unknown";
}

// ── The database ─────────────────────────────────────────────────────

// All Mersenne primes that fit in uint64_t: 2^p - 1
// M2=3, M3=7, M5=31, M7=127, M13=8191, M17=131071, M19=524287,
// M31=2147483647, M61=2305843009213693951
static const KnownEntry KNOWN_PRIMES_DB[] = {
    // ── Mersenne Primes (2^p - 1) ──
    {3,                       KnownClass::MersennePrime, "M2 = 2^2 - 1"},
    {7,                       KnownClass::MersennePrime, "M3 = 2^3 - 1"},
    {31,                      KnownClass::MersennePrime, "M5 = 2^5 - 1"},
    {127,                     KnownClass::MersennePrime, "M7 = 2^7 - 1"},
    {8191,                    KnownClass::MersennePrime, "M13 = 2^13 - 1"},
    {131071,                  KnownClass::MersennePrime, "M17 = 2^17 - 1"},
    {524287,                  KnownClass::MersennePrime, "M19 = 2^19 - 1"},
    {2147483647ULL,           KnownClass::MersennePrime, "M31 = 2^31 - 1"},
    {2305843009213693951ULL,  KnownClass::MersennePrime, "M61 = 2^61 - 1"},

    // ── Fermat Primes (2^(2^n) + 1) ──
    {5,       KnownClass::FermatPrime, "F1 = 2^(2^1) + 1"},
    {17,      KnownClass::FermatPrime, "F2 = 2^(2^2) + 1"},
    {257,     KnownClass::FermatPrime, "F3 = 2^(2^3) + 1"},
    {65537,   KnownClass::FermatPrime, "F4 = 2^(2^4) + 1"},

    // ── Wieferich Primes ── (only 2 known!)
    {1093,  KnownClass::WieferichPrime, "1st Wieferich: 2^1092 mod 1093^2 = 1"},
    {3511,  KnownClass::WieferichPrime, "2nd Wieferich: 2^3510 mod 3511^2 = 1"},

    // ── Wilson Primes ── (only 3 known!)
    // 5 is also Fermat, listed above
    {13,   KnownClass::WilsonPrime, "2nd Wilson: 12! mod 169 = 168"},
    {563,  KnownClass::WilsonPrime, "3rd Wilson: 562! mod 563^2 ≡ -1"},

    // ── Wagstaff Primes ((2^p + 1) / 3) ──
    {2796203,             KnownClass::WagstaffPrime, "(2^23 + 1)/3"},
    {178956971,           KnownClass::WagstaffPrime, "(2^29 + 1)/3"},
    {2932031007403ULL,    KnownClass::WagstaffPrime, "(2^43 + 1)/3"},
    {768614336404564651ULL, KnownClass::WagstaffPrime, "(2^61 + 1)/3"},

    // ── Factorial Primes (n! ± 1) ──
    {719,     KnownClass::Factorial, "6! - 1 = 719"},
    {5039,    KnownClass::Factorial, "7! - 1 = 5039"},
    {39916801ULL, KnownClass::Factorial, "11! + 1"},
    {479001599ULL, KnownClass::Factorial, "12! - 1"},
    {87178291199ULL, KnownClass::Factorial, "14! - 1"},

    // ── Primorial Primes (p# ± 1) ──
    {2311,     KnownClass::Primorial, "11# + 1 = 2311"},
    {30029,    KnownClass::Primorial, "13# - 1 = 30029"},

    // ── Notable Twin Primes (both p and p+2 prime) ──
    {2027,    KnownClass::TwinPrime, "twin: (2027, 2029)"},
    {3119,    KnownClass::TwinPrime, "twin: (3119, 3121)"},
    {5009,    KnownClass::TwinPrime, "twin: (5009, 5011)"},
    {10007,   KnownClass::TwinPrime, "twin: (10007, 10009)"},
    {1000037, KnownClass::TwinPrime, "twin: (1000037, 1000039)"},
    {10000139ULL, KnownClass::TwinPrime, "twin: (10000139, 10000141)"},
    {1000000007ULL, KnownClass::TwinPrime, "twin: (1000000007, 1000000009)"},

    // ── Sophie Germain Primes (p and 2p+1 both prime, from OEIS A005384) ──
    {2381,    KnownClass::SophieGermain, "SG: 2381, safe 4763"},
    {2399,    KnownClass::SophieGermain, "SG: 2399, safe 4799"},
    {2819,    KnownClass::SophieGermain, "SG: 2819, safe 5639"},
    {2963,    KnownClass::SophieGermain, "SG: 2963, safe 5927"},
    {3023,    KnownClass::SophieGermain, "SG: 3023, safe 6047"},

    // ── Palindromic Primes ──
    {10301,      KnownClass::Palindromic, "5-digit palindromic prime"},
    {1003001,    KnownClass::Palindromic, "7-digit palindromic prime"},
    {100030001,  KnownClass::Palindromic, "9-digit palindromic prime"},
    {10000500001ULL, KnownClass::Palindromic, "11-digit palindromic prime"},

    // ── Repunit Primes ((10^n - 1)/9) ──
    {1111111111111111111ULL, KnownClass::Repunit, "R19 = (10^19 - 1)/9"},

    // ── Other Notable Primes ──
    {2053,          KnownClass::NotablePrime, "first prime > 2048"},
    {4099,          KnownClass::NotablePrime, "first prime > 4096"},
    {8209,          KnownClass::NotablePrime, "first prime > 8192"},
    {65537,         KnownClass::NotablePrime, "2^16 + 1, largest known Fermat prime"},
    {104729,        KnownClass::NotablePrime, "10000th prime"},
    {1299709,       KnownClass::NotablePrime, "100000th prime"},
    {15485863,      KnownClass::NotablePrime, "1000000th prime"},
    {179424673ULL,  KnownClass::NotablePrime, "10000000th prime"},
    {2038074743ULL, KnownClass::NotablePrime, "100000000th prime"},
    {4294967291ULL, KnownClass::NotablePrime, "largest prime < 2^32"},
    {4294967311ULL, KnownClass::NotablePrime, "first prime > 2^32"},
    {1099511627791ULL,            KnownClass::NotablePrime, "first prime > 2^40"},
    {281474976710677ULL,          KnownClass::NotablePrime, "first prime > 2^48"},
    {72057594037927931ULL,        KnownClass::NotablePrime, "first prime > 2^56"},
    {18446744073709551557ULL,     KnownClass::NotablePrime, "largest prime < 2^64"},

    // ═══════════════════════════════════════════════════════════════
    // PSEUDOPRIMES — Carmichael numbers (composite but pass Fermat test)
    // ═══════════════════════════════════════════════════════════════
    {561,       KnownClass::CarmichaelNumber, "3 x 11 x 17 — 1st Carmichael"},
    {1105,      KnownClass::CarmichaelNumber, "5 x 13 x 17"},
    {1729,      KnownClass::CarmichaelNumber, "7 x 13 x 19 — Hardy-Ramanujan / taxicab"},
    {2465,      KnownClass::CarmichaelNumber, "5 x 17 x 29"},
    {2821,      KnownClass::CarmichaelNumber, "7 x 13 x 31"},
    {6601,      KnownClass::CarmichaelNumber, "7 x 23 x 41"},
    {8911,      KnownClass::CarmichaelNumber, "7 x 19 x 67"},
    {10585,     KnownClass::CarmichaelNumber, "5 x 29 x 73"},
    {15841,     KnownClass::CarmichaelNumber, "7 x 31 x 73"},
    {29341,     KnownClass::CarmichaelNumber, "13 x 37 x 61"},
    {41041,     KnownClass::CarmichaelNumber, "7 x 11 x 13 x 41"},
    {46657,     KnownClass::CarmichaelNumber, "13 x 37 x 97"},
    {52633,     KnownClass::CarmichaelNumber, "7 x 73 x 103"},
    {62745,     KnownClass::CarmichaelNumber, "3 x 5 x 47 x 89"},
    {63973,     KnownClass::CarmichaelNumber, "7 x 13 x 19 x 37"},
    {75361,     KnownClass::CarmichaelNumber, "11 x 13 x 17 x 31"},
    {101101,    KnownClass::CarmichaelNumber, "7 x 11 x 13 x 101"},
    {115921,    KnownClass::CarmichaelNumber, "13 x 37 x 241"},
    {126217,    KnownClass::CarmichaelNumber, "7 x 13 x 19 x 73"},
    {162401,    KnownClass::CarmichaelNumber, "17 x 41 x 233"},
    {172081,    KnownClass::CarmichaelNumber, "7 x 13 x 31 x 61"},
    {188461,    KnownClass::CarmichaelNumber, "7 x 13 x 19 x 109"},
    {252601,    KnownClass::CarmichaelNumber, "41 x 61 x 101"},
    {278545,    KnownClass::CarmichaelNumber, "5 x 17 x 29 x 113"},
    {294409,    KnownClass::CarmichaelNumber, "37 x 73 x 109"},
    {314821,    KnownClass::CarmichaelNumber, "13 x 61 x 397"},
    {334153,    KnownClass::CarmichaelNumber, "19 x 43 x 409"},
    {340561,    KnownClass::CarmichaelNumber, "13 x 17 x 23 x 67"},
    {399001,    KnownClass::CarmichaelNumber, "31 x 61 x 211"},
    {410041,    KnownClass::CarmichaelNumber, "41 x 73 x 137"},
    {488881,    KnownClass::CarmichaelNumber, "37 x 73 x 181"},
    {512461,    KnownClass::CarmichaelNumber, "31 x 61 x 271"},
    {530881,    KnownClass::CarmichaelNumber, "13 x 97 x 421"},
    {552721,    KnownClass::CarmichaelNumber, "13 x 17 x 41 x 61"},
    {825265,    KnownClass::CarmichaelNumber, "5 x 7 x 17 x 19 x 73"},
    {1024651,   KnownClass::CarmichaelNumber, "19 x 199 x 271"},
    {1152271,   KnownClass::CarmichaelNumber, "43 x 127 x 211"},
    {1193221,   KnownClass::CarmichaelNumber, "31 x 61 x 631"},
    {1461241,   KnownClass::CarmichaelNumber, "37 x 73 x 541"},
    {1615681,   KnownClass::CarmichaelNumber, "13 x 37 x 3361"},

    // ── Strong pseudoprimes to base 2 (SPRP-2, OEIS A001262) ──
    {2047,     KnownClass::StrongPseudoprime, "sprp-2: 23 x 89"},
    {3277,     KnownClass::StrongPseudoprime, "sprp-2: 29 x 113"},
    {4033,     KnownClass::StrongPseudoprime, "sprp-2: 37 x 109"},
    {4681,     KnownClass::StrongPseudoprime, "sprp-2: 31 x 151"},
    {8321,     KnownClass::StrongPseudoprime, "sprp-2: 53 x 157"},
    {15841,    KnownClass::StrongPseudoprime, "sprp-2: 7 x 31 x 73"},
    {29341,    KnownClass::StrongPseudoprime, "sprp-2: 13 x 37 x 61"},
    {42799,    KnownClass::StrongPseudoprime, "sprp-2: 127 x 337"},
    {49141,    KnownClass::StrongPseudoprime, "sprp-2: 157 x 313"},
    {52633,    KnownClass::StrongPseudoprime, "sprp-2: 7 x 73 x 103"},
    {65281,    KnownClass::StrongPseudoprime, "sprp-2: 97 x 673"},
    {74665,    KnownClass::StrongPseudoprime, "sprp-2: 5 x 109 x 137"},
    {80581,    KnownClass::StrongPseudoprime, "sprp-2: 61 x 1321"},
    {85489,    KnownClass::StrongPseudoprime, "sprp-2: 71 x 1204"},
    {88357,    KnownClass::StrongPseudoprime, "sprp-2: 149 x 593"},
    {90751,    KnownClass::StrongPseudoprime, "sprp-2"},

    // ── Large Carmichael numbers ──
    {1105,       KnownClass::CarmichaelNumber, "5 x 13 x 17"},
    {10024561ULL, KnownClass::CarmichaelNumber, "large Carmichael"},
    {118901521ULL, KnownClass::CarmichaelNumber, "large Carmichael"},
    {172947529ULL, KnownClass::CarmichaelNumber, "large Carmichael"},
    {216821881ULL, KnownClass::CarmichaelNumber, "large Carmichael"},
    {228842209ULL, KnownClass::CarmichaelNumber, "large Carmichael"},
    {1299963601ULL, KnownClass::CarmichaelNumber, "large Carmichael"},
    {2301745249ULL, KnownClass::CarmichaelNumber, "large Carmichael"},
    {9746347772161ULL, KnownClass::CarmichaelNumber, "13-digit Carmichael"},
    {1436697831295441ULL, KnownClass::CarmichaelNumber, "16-digit Carmichael"},
};

static const size_t KNOWN_DB_SIZE = sizeof(KNOWN_PRIMES_DB) / sizeof(KNOWN_PRIMES_DB[0]);

// ── Lookup class ─────────────────────────────────────────────────────

class KnownPrimesDB {
public:
    KnownPrimesDB() {
        for (size_t i = 0; i < KNOWN_DB_SIZE; i++) {
            const auto& e = KNOWN_PRIMES_DB[i];
            if (e.value == 0) continue; // skip invalid
            _lookup[e.value] = i;
            if (e.kclass == KnownClass::CarmichaelNumber ||
                e.kclass == KnownClass::StrongPseudoprime ||
                e.kclass == KnownClass::EulerPseudoprime) {
                _pseudoprimes.insert(e.value);
            } else {
                _primes.insert(e.value);
            }
        }
    }

    // Check if a number is in the known database
    bool is_known(uint64_t n) const {
        return _lookup.find(n) != _lookup.end();
    }

    // Check if it's a known prime (not pseudoprime)
    bool is_known_prime(uint64_t n) const {
        return _primes.find(n) != _primes.end();
    }

    // Check if it's a known pseudoprime
    bool is_known_pseudoprime(uint64_t n) const {
        return _pseudoprimes.find(n) != _pseudoprimes.end();
    }

    // Get all entries for a value (can have multiple classifications)
    std::vector<const KnownEntry*> get_entries(uint64_t n) const {
        std::vector<const KnownEntry*> result;
        for (size_t i = 0; i < KNOWN_DB_SIZE; i++) {
            if (KNOWN_PRIMES_DB[i].value == n) {
                result.push_back(&KNOWN_PRIMES_DB[i]);
            }
        }
        return result;
    }

    // Get the primary entry
    const KnownEntry* get_entry(uint64_t n) const {
        auto it = _lookup.find(n);
        if (it == _lookup.end()) return nullptr;
        return &KNOWN_PRIMES_DB[it->second];
    }

    // Get all known primes as a sorted vector
    std::vector<uint64_t> all_known_primes() const {
        std::vector<uint64_t> v(_primes.begin(), _primes.end());
        std::sort(v.begin(), v.end());
        return v;
    }

    // Get all known pseudoprimes as a sorted vector
    std::vector<uint64_t> all_known_pseudoprimes() const {
        std::vector<uint64_t> v(_pseudoprimes.begin(), _pseudoprimes.end());
        std::sort(v.begin(), v.end());
        return v;
    }

    size_t prime_count() const { return _primes.size(); }
    size_t pseudoprime_count() const { return _pseudoprimes.size(); }

private:
    std::unordered_map<uint64_t, size_t> _lookup;  // value -> first index in DB
    std::unordered_set<uint64_t> _primes;
    std::unordered_set<uint64_t> _pseudoprimes;
};

// Singleton accessor
inline const KnownPrimesDB& known_db() {
    static KnownPrimesDB db;
    return db;
}

// Format factors as comma-delimited string: "3, 11, 17"
inline std::string factors_comma_string(uint64_t n) {
    auto factors = factor_u64(n);  // from PrimeEngine.hpp — includes Lucky7s + PinchFactor
    std::string s;
    for (size_t i = 0; i < factors.size(); i++) {
        if (i > 0) s += ", ";
        s += std::to_string(factors[i]);
    }
    return s;
}

} // namespace prime
