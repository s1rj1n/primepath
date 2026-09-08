# PrimePath

macOS (Apple silicon) Metal GPU prime discovery engine. Objective-C++ app around a
C++ engine, with GIMPS/PrimeNet integration and multi-Mac distributed search.

Architecture is documented in `README.md` (layer table) and, for the GPU /
networking / Nester Carry Chain internals, in the `distributed-compute` skill
(`.claude/skills/distributed-compute/`). Read those rather than re-deriving.

## Commands

Engine tests — 108 of them, the fast feedback loop. Runs without Xcode:

```bash
clang++ -std=c++17 -O2 -I. test_engine.cpp PrimePath/PrimeEngine.cpp -o test_engine -lpthread
./test_engine          # expect: 108 passed, 0 failed
```

Syntax-check a single Objective-C++ file without a full build (seconds, not minutes):

```bash
clang -fsyntax-only -x objective-c++ -std=c++17 -fobjc-arc -Wformat \
  -IPrimePath -IPrimePath/Network -I. PrimePath/AppDelegate.mm
```

Build the app:

```bash
xcodebuild -project PrimePath.xcodeproj -scheme PrimePath -configuration Release \
  -derivedDataPath <dir> build
```

`-Wformat` is clean as of v1.4.1 — keep it that way. `stringWithFormat:` takes `%@`
for an `NSString *`; `%s` there is undefined behaviour, not a style nit.

## Versioning

`Info.plist` (`CFBundleShortVersionString`) is the **single source of truth**. Every
display string, `User-Agent`, and PrimeNet report reads it through
`PrimePath/Version.h` (`PrimePathVersion()` / `PrimePathVersionUTF8()`).

Never hard-code a version anywhere else. It used to live in 12 literals across three
files, which is how the window title sat at v0.5 for four releases (issue #2).
Bumping a release is now a one-line `Info.plist` edit.

`Version.h` is header-only on purpose: a new `.mm` would have to be hand-added to
`project.pbxproj` to join the build phase. Prefer inline headers for small shared
helpers here.

## Data directory

All runtime state lives in `~/Library/Application Support/PrimePath/`, resolved by
`PrimePathDataDirectory()` in `AppDelegate.mm` and created at launch.

Never hard-code a developer path — that bug shipped for months and meant the app
silently persisted nothing on anyone else's machine (PR #3, Niles Turner). There is
no migration from the old `~/Documents/primes/primelocations/` location.

## Releasing

Artifacts are **not** committed — `*.zip` and `*.dmg` are gitignored and go to GitHub
Releases. Recent releases ship a notarized `.app.zip`:

```bash
xcodebuild ... build \
  CODE_SIGN_IDENTITY="Developer ID Application: Sergei  Nester (TKYG23Q3ZF)" \
  DEVELOPMENT_TEAM=TKYG23Q3ZF CODE_SIGN_STYLE=Manual \
  CODE_SIGN_INJECT_BASE_ENTITLEMENTS=NO \
  OTHER_CODE_SIGN_FLAGS="--timestamp --options runtime"

ditto -c -k --keepParent <App> notarize.zip
xcrun notarytool submit notarize.zip --keychain-profile PrimePath --wait
xcrun stapler staple <App>          # staple the .app, then re-zip
ditto -c -k --keepParent <App> dist/PrimePath-<version>.app.zip
```

Verify before publishing — `spctl -a -vvv -t install <App>` must report
`source=Notarized Developer ID`. Staple *before* the final zip or the download
needs a network round-trip to validate.

`scripts/build-dmg.sh` does the same for a DMG, but its `BUNDLE_ID` says
`com.sergeinester.PrimePath` while the app is actually `com.primes.PrimePath`.

Update `CHANGELOG.md` in the same change as the code — it drifted badly once and
shipped a release describing files that had been renamed.

## Conventions

- `AppDelegate.mm` is ~9,800 lines. Locate work by symbol, don't read it whole.
- Objective-C++ (`.mm`) with ARC on. C++17.
- Author uses `--` in prose, not em dashes, and lowercase-terse commit style.
- `README.md`'s layer table says `TaskManager.cpp`; the file is `TaskManager.mm`.
