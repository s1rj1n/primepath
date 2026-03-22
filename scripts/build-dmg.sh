#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════
# PrimePath — Build, Sign, Notarize, and Package as DMG
# ═══════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SCHEME="PrimePath"
APP_NAME="PrimePath"
DIST_DIR="$PROJECT_DIR/dist"
TEAM_ID="TKYG23Q3ZF"
SIGNING_IDENTITY="Developer ID Application: Sergei  Nester ($TEAM_ID)"
BUNDLE_ID="com.sergeinester.PrimePath"

# Keychain profile for notarization (set up once with:
#   xcrun notarytool store-credentials "PrimePath" --apple-id YOUR_APPLE_ID --team-id TKYG23Q3ZF
# )
NOTARY_PROFILE="PrimePath"

echo "=== Building PrimePath Release ==="

# Clean and build Release
xcodebuild -project "$PROJECT_DIR/PrimePath.xcodeproj" \
    -scheme "$SCHEME" \
    -configuration Release \
    -derivedDataPath "$DIST_DIR/build" \
    clean build \
    CODE_SIGN_IDENTITY="$SIGNING_IDENTITY" \
    DEVELOPMENT_TEAM="$TEAM_ID" \
    CODE_SIGN_STYLE="Manual" \
    CODE_SIGN_INJECT_BASE_ENTITLEMENTS=NO \
    OTHER_CODE_SIGN_FLAGS="--timestamp --options runtime"

APP_PATH="$DIST_DIR/build/Build/Products/Release/$APP_NAME.app"

if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: Build failed — $APP_PATH not found"
    exit 1
fi

echo "=== App built at $APP_PATH ==="

# Get version from Info.plist
VERSION=$(defaults read "$APP_PATH/Contents/Info" CFBundleShortVersionString 2>/dev/null || echo "1.0")
DMG_NAME="${APP_NAME}-${VERSION}.dmg"
DMG_PATH="$DIST_DIR/$DMG_NAME"

# Remove old DMG
rm -f "$DMG_PATH"

echo "=== Creating DMG ==="

# Create a temporary directory for DMG contents
DMG_STAGING="$DIST_DIR/dmg-staging"
rm -rf "$DMG_STAGING"
mkdir -p "$DMG_STAGING"
cp -R "$APP_PATH" "$DMG_STAGING/"
ln -s /Applications "$DMG_STAGING/Applications"

# Create DMG
hdiutil create -volname "$APP_NAME" \
    -srcfolder "$DMG_STAGING" \
    -ov -format UDZO \
    "$DMG_PATH"

rm -rf "$DMG_STAGING"

echo "=== Signing DMG ==="
codesign --force --sign "$SIGNING_IDENTITY" --timestamp "$DMG_PATH"

echo "=== Notarizing ==="
echo "(This may take a few minutes...)"

xcrun notarytool submit "$DMG_PATH" \
    --keychain-profile "$NOTARY_PROFILE" \
    --wait

echo "=== Stapling notarization ticket ==="
xcrun stapler staple "$DMG_PATH"

echo ""
echo "=== Done! ==="
echo "DMG: $DMG_PATH"
echo "Size: $(du -h "$DMG_PATH" | cut -f1)"
echo ""
echo "Upload to GitHub Releases:"
echo "  gh release create v$VERSION $DMG_PATH --title 'PrimePath v$VERSION' --notes 'Release notes here'"
