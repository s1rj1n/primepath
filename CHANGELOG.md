# Changelog

## v1.2.1 – AutoPrimeNet Interop & GPU Core Fix (2026-04-10)

### New
- **`worktodo.txt` reader** – PrimePath now reads the standard GIMPS `worktodo.txt` file used by AutoPrimeNet and mfaktc/mfakto. Supported line format: `Factor=<AID>,<exponent>,<bitlo>,<bithi>` (AID optional). Blank lines, `#`, `;`, and `//` comments are ignored.
- **AutoPrimeNet integration mode** – when `worktodo.txt` is present, PrimePath reads assignments from there instead of calling the PrimeNet v5 server. After a TF run finishes, the JSON result is appended to `results.json.txt` and the matching line is atomically removed from `worktodo.txt`. This lets users run PrimePath through [AutoPrimeNet](https://github.com/tdulcet/AutoPrimeNet) for full assignment management, email notifications, log rotation, proxy support, and stall monitoring, without needing to register PrimePath with PrimeNet directly. Hybrid mode (registered and using `worktodo.txt`) also works and cleans up both state locations on successful submit.

### Fixed
- **GPU core count detection** – `sysctlbyname("gpu.core_count", ...)` does not exist on macOS, so the `gpu_cores` field was silently dropped from the JSON output. Replaced with an IORegistry lookup on `AGXAccelerator` for the `gpu-core-count` property. M5 now correctly reports 10 GPU cores.
- **Stale version string** – JSON editor and `PrimeNetClient` now emit `version":"1.2.0"` consistently. Previous builds could produce output showing `1.0.0` if a cached binary was run.
- **All-zero AID handling** – JSON output no longer emits `"aid":"00000000000000000000000000000000"` for simulated or manual runs; the field is omitted entirely unless a real assignment key is present.

## v1.2 – JSON Result Editor & PrimeNet JSON Format (2026-04-10)

### New
- **JSON Result Editor panel** – full GUI editor for PrimeNet JSON results with:
  - Editable fields for every result field (exponent, status, bit range, factors, user, computer, AID, program, kernel)
  - Status and range-complete popups with inline validation hints
  - Auto-detected hardware and OS info (chip, CPU P/E cores, GPU cores, RAM, Darwin version)
  - **Validate** – checks JSON syntax, required fields, status/factor consistency, timestamp format, 2000-char limit
  - **Load Template / Save Template** – reusable JSON configurations (`.json` files)
  - **Simulate Test** – loads a known result (M67 has factor 193707721) for format verification
  - **Auto-Fill** – populates fields from system info, current assignment, and discoveries
  - **Send to Server** – builds the full `t=ar` URL with credentials and JSON as `&m=`, with confirmation dialog
  - Dual-pane output: editable JSON text view + validation/server log
- **PrimeNet JSON result format** – result submissions now send a single-line JSON in the `&m=` parameter instead of the old human-readable string. Local `results.json.txt` writes the same JSON as submitted. Matches the mersenne.org expected format with fields: `timestamp` (UTC), `exponent`, `worktype`, `status`, `bitlo`, `bithi`, `rangecomplete`, `factors` (if F), `program` (name/version/kernel), `os` (os/version/architecture), `user`, `computer`, `aid`, `hardware` (chip/cores/RAM).

### Changed
- Version bumped to 1.2.0 across User-Agent header, PrimeNet registration, machine description, and JSON output
- `CFBundleShortVersionString` / `CFBundleVersion` updated to 1.2.0 in Info.plist
- PrimeNet result submission no longer writes the old `"M<exp> has a factor..."` string format

## v1.1 – Carry-Chain Mulmod (2026-03-26)

### New
- **Carry-chain hardware mulmod** – 4-7x faster 128-bit modular multiplication for CPU tests (Wieferich, Wall-Sun-Sun). Uses ARM64 MUL+UMULH with __int128 reduction instead of binary shift-and-add. Works for moduli up to 95 bits. Toggle via CarryChain checkbox in main window.
- **CPU factor verification** – GPU-found Mersenne factors are independently verified on CPU using carry-chain `2^p mod q == 1` before logging or submission. Mismatches are flagged but still saved for investigation.
- **System metadata in GIMPS** – PrimeNet registration and result submissions now include dynamic system specs: chip name (e.g. Apple M5), CPU core breakdown (P+E), GPU cores, RAM, macOS version, and source repo (github.com/s1rj1n/primepath).
- **Carry-chain benchmark** – CarryChain checkbox runs a comparison test showing binary vs carry-chain performance across multiple bit widths with the current GIMPS exponent.

### Fixed
- **Discoveries file bloat** – `discoveries.txt` was growing to 100s of MB from pseudoprime/composite filtering artifacts (false WSS positives from the earlier overflow bug). Now only confirmed primes and Mersenne/Fermat factors are persisted. Cleared old bogus data.
- **Atomic file writes** – `discoveries.txt` now writes to a temp file then renames, preventing Spotlight/Finder indexing from blocking the write.
- **Removed pseudoprimes.txt / composites.txt** – replaced with single `primes.txt` for confirmed discoveries only.

### Changed
- PrimeNet `&a=` field now includes GitHub repo reference
- PrimeNet `&c=` field uses dynamic chip name instead of hardcoded string
- PrimeNet result `&m=` message includes full system description
- Local `results.json.txt` includes `source` and `machine` fields

## v1.0 – Initial Release

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
