# Changelog

## v1.1 — Carry-Chain Mulmod (2026-03-26)

### New
- **Carry-chain hardware mulmod** — 4-7x faster 128-bit modular multiplication for CPU tests (Wieferich, Wall-Sun-Sun). Uses ARM64 MUL+UMULH with __int128 reduction instead of binary shift-and-add. Works for moduli up to 95 bits. Toggle via CarryChain checkbox in main window.
- **CPU factor verification** — GPU-found Mersenne factors are independently verified on CPU using carry-chain `2^p mod q == 1` before logging or submission. Mismatches are flagged but still saved for investigation.
- **System metadata in GIMPS** — PrimeNet registration and result submissions now include dynamic system specs: chip name (e.g. Apple M5), CPU core breakdown (P+E), GPU cores, RAM, macOS version, and source repo (github.com/s1rj1n/primepath).
- **Carry-chain benchmark** — CarryChain checkbox runs a comparison test showing binary vs carry-chain performance across multiple bit widths with the current GIMPS exponent.

### Fixed
- **Discoveries file bloat** — `discoveries.txt` was growing to 100s of MB from pseudoprime/composite filtering artifacts (false WSS positives from the earlier overflow bug). Now only confirmed primes and Mersenne/Fermat factors are persisted. Cleared old bogus data.
- **Atomic file writes** — `discoveries.txt` now writes to a temp file then renames, preventing Spotlight/Finder indexing from blocking the write.
- **Removed pseudoprimes.txt / composites.txt** — replaced with single `primes.txt` for confirmed discoveries only.

### Changed
- PrimeNet `&a=` field now includes GitHub repo reference
- PrimeNet `&c=` field uses dynamic chip name instead of hardcoded string
- PrimeNet result `&m=` message includes full system description
- Local `results.json.txt` includes `source` and `machine` fields

## v1.0 — Initial Release

- Metal GPU prime search engine for Apple Silicon
- Mersenne trial factoring with 96-bit Barrett reduction on GPU
- Fermat factor search
- Wieferich, Wall-Sun-Sun, Wilson, twin, Sophie Germain, cousin, sexy, emirp searches
- GIMPS / PrimeNet v5 API integration
- Fused GPU sieve+modexp kernel
- CPU/GPU load balancer with last-digit work splitting
- Distributed search (Conductor/Carriage with Bonjour)
- Test catalog with 30+ number theory tests
- Pipeline builder for custom search configurations
- Factor seed prediction (Ring Beacon, Topography, Factor Web, Audio Harmonic, Twisting Tree)
- Metal modular arithmetic library (standalone)
