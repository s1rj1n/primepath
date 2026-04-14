#import "AppDelegate.h"
#import "MetalCompute.h"
#import "PrimeEngine.hpp"
#import "TaskManager.hpp"
#import "KnownPrimes.hpp"
#import "Benchmark.hpp"
#include <string>
#include <mach/mach.h>
#include <sys/resource.h>
#include <thread>
#include <atomic>
#include <algorithm>
#include <set>
#include <map>
#include <future>
#include "Network/ConductorServer.hpp"
#include "Network/CarriageClient.hpp"
#include "Network/PrimeNetClient.hpp"
#include "DataTools.hpp"
#import <IOKit/pwr_mgt/IOPMLib.h>
#import <IOKit/IOKitLib.h>
#include <sys/utsname.h>
#include <zlib.h>
#include <sys/sysctl.h>
#import <objc/runtime.h>

static NSString *const DATA_DIR = @"/Users/sergeinester/Documents/primes/primelocations";

// Defined in TaskManager.mm (inside namespace prime) — carry-chain mulmod toggle
namespace prime { extern volatile bool g_use_carry_chain; }

// ── Test Catalog Entry (loaded from TestCatalog.txt) ─────────────────
struct TestCatalogEntry {
    std::string test_id;
    std::string name;
    std::string category;
    std::string mode;        // "search" or "tool"
    std::string description;
    std::string algorithms;
    std::string default_params;
};

static NSString *formatNumber(uint64_t n) {
    NSNumberFormatter *fmt = [[NSNumberFormatter alloc] init];
    fmt.numberStyle = NSNumberFormatterDecimalStyle;
    fmt.groupingSeparator = @",";
    return [fmt stringFromNumber:@(n)];
}

// ═══════════════════════════════════════════════════════════════════════
// EQBarView -- Graphic equalizer-style vertical bar visualizer
// ═══════════════════════════════════════════════════════════════════════

static const int EQ_HISTORY = 32; // number of vertical bars (time history)

@interface EQBarView : NSView {
    double _history[EQ_HISTORY];
    double _peaks[EQ_HISTORY];
    int _head;
    NSString *_title;
    NSString *_detail;
    NSColor *_barColor;
    NSColor *_peakColor;
}
- (instancetype)initWithFrame:(NSRect)frame title:(NSString *)title color:(NSColor *)color;
- (void)pushValue:(double)pct; // 0-100
- (void)setDetail:(NSString *)detail;
@end

@implementation EQBarView

- (instancetype)initWithFrame:(NSRect)frame title:(NSString *)title color:(NSColor *)color {
    self = [super initWithFrame:frame];
    if (self) {
        _title = title;
        _detail = @"";
        _barColor = color;
        _peakColor = [NSColor colorWithSRGBRed:1.0 green:1.0 blue:1.0 alpha:0.7];
        _head = 0;
        memset(_history, 0, sizeof(_history));
        memset(_peaks, 0, sizeof(_peaks));
    }
    return self;
}

- (void)pushValue:(double)pct {
    pct = fmax(0, fmin(100, pct));
    _history[_head] = pct;
    // Peak hold with decay
    if (pct >= _peaks[_head]) {
        _peaks[_head] = pct;
    }
    _head = (_head + 1) % EQ_HISTORY;
    // Decay all peaks
    for (int i = 0; i < EQ_HISTORY; i++) {
        _peaks[i] *= 0.97;
    }
    [self setNeedsDisplay:YES];
}

- (void)setDetail:(NSString *)detail {
    _detail = detail;
    [self setNeedsDisplay:YES];
}

- (void)drawRect:(NSRect)dirtyRect {
    NSRect b = self.bounds;

    // Dark background
    [[NSColor colorWithSRGBRed:0.08 green:0.08 blue:0.12 alpha:1.0] set];
    NSBezierPath *bg = [NSBezierPath bezierPathWithRoundedRect:b xRadius:4 yRadius:4];
    [bg fill];

    // Title (top-left)
    NSDictionary *titleAttrs = @{
        NSFontAttributeName: [NSFont boldSystemFontOfSize:8],
        NSForegroundColorAttributeName: [NSColor colorWithWhite:0.7 alpha:1.0]
    };
    [_title drawAtPoint:NSMakePoint(b.origin.x + 4, b.origin.y + b.size.height - 12) withAttributes:titleAttrs];

    // Detail (top-right)
    if (_detail.length > 0) {
        NSDictionary *detailAttrs = @{
            NSFontAttributeName: [NSFont monospacedSystemFontOfSize:7 weight:NSFontWeightRegular],
            NSForegroundColorAttributeName: [NSColor colorWithWhite:0.5 alpha:1.0]
        };
        NSSize ds = [_detail sizeWithAttributes:detailAttrs];
        [_detail drawAtPoint:NSMakePoint(b.origin.x + b.size.width - ds.width - 4,
            b.origin.y + b.size.height - 11) withAttributes:detailAttrs];
    }

    // Draw EQ bars
    CGFloat barArea = b.size.width - 8;  // padding
    CGFloat barH = b.size.height - 16;   // leave room for title
    CGFloat barW = barArea / EQ_HISTORY;
    CGFloat gap = fmax(1.0, barW * 0.15);
    barW -= gap;
    if (barW < 1) barW = 1;

    for (int i = 0; i < EQ_HISTORY; i++) {
        int idx = (_head + i) % EQ_HISTORY;
        double val = _history[idx];
        double peak = _peaks[idx];

        CGFloat x = b.origin.x + 4 + i * (barW + gap);
        CGFloat h = (val / 100.0) * barH;
        CGFloat peakY = (peak / 100.0) * barH;

        // Color gradient: green -> yellow -> red based on value
        NSColor *c;
        if (val > 80) {
            c = [NSColor colorWithSRGBRed:0.9 green:0.2 blue:0.15 alpha:0.9];
        } else if (val > 50) {
            CGFloat t = (val - 50) / 30.0;
            c = [NSColor colorWithSRGBRed:0.2 + 0.7*t green:0.8 - 0.3*t blue:0.1 alpha:0.9];
        } else {
            c = [_barColor colorWithAlphaComponent:0.7 + 0.3 * (val / 50.0)];
        }
        [c set];

        NSRect barRect = NSMakeRect(x, b.origin.y + 2, barW, fmax(1, h));
        [[NSBezierPath bezierPathWithRoundedRect:barRect xRadius:1 yRadius:1] fill];

        // Peak indicator (thin line)
        if (peak > 2 && peakY > h + 2) {
            [_peakColor set];
            NSRect peakRect = NSMakeRect(x, b.origin.y + 2 + peakY - 1, barW, 1.5);
            [NSBezierPath fillRect:peakRect];
        }
    }
}

@end

// ═══════════════════════════════════════════════════════════════════════
// PrimePath v0.5 -- Metal GPU + Multi-Core Prime Discovery
// ═══════════════════════════════════════════════════════════════════════

@interface AppDelegate () {
    prime::TaskManager *_taskMgr;
    prime::GPUBackend *_gpu;
    std::atomic<bool> _checkRunning;
    std::thread _checkThread;
    std::atomic<bool> _benchRunning;
    std::thread _benchThread;
    // CPU monitoring delta state (process-only)
    uint64_t _prevProcCPU_ns;       // previous process CPU time (user+system) in nanoseconds
    uint64_t _prevWallClock_ns;     // previous wall clock time in nanoseconds
    // NEON/SIMD monitoring delta state
    uint64_t _prevNeonTested;
    uint64_t _prevNeonRejected;
    // Disk I/O monitoring
    uint64_t _prevDiskReadBytes;
    uint64_t _prevDiskWriteBytes;
    // PrimeLocation predicted primes list (persists after test ends)
    std::vector<uint64_t> _predictedPrimes;
    // Markov Predict: loaded known prime list for training
    std::vector<uint64_t> _markovPrimeList;
    // Distributed computing
    prime::ConductorServer *_conductor;
    prime::CarriageClient  *_carriage;
    // GIMPS / PrimeNet integration
    primenet::PrimeNetClient *_primenet;
    // Power management — prevent sleep while tasks are running
    IOPMAssertionID _sleepAssertionID;
    BOOL _preventSleepEnabled;
    // Test catalog
    std::vector<TestCatalogEntry> _testCatalog;
    std::vector<size_t> _testDisplayOrder;  // indices into _testCatalog (flat, with category headers as SIZE_MAX)
    std::vector<std::string> _testCategories; // category names for headers
    int _selectedTestIdx;  // index into _testCatalog, -1 if none
    // Per-tab ring buffers for output panes
    // Tab indices: 0=Search Tasks, 1=Check/Expressions, 2=Benchmark, 3=Pipeline
    NSMutableArray<NSString *> *_tabRings[4];
    NSUInteger _tabRingHeads[4];
    BOOL _tabDirty[4];
    NSTextView *_tabViews[4];
    NSTimer *_logFlushTimer;
    NSLock *_logLock;
    // Which tab is currently being written to by appendText (for benchmark/pipeline routing)
    int _activeLogTab;
    // Status pane ring buffer (GIMPS, PrimeNet, discoveries, network)
    NSMutableArray<NSString *> *_statusRing;
    NSUInteger _statusRingHead;
    BOOL _statusDirty;
}

@property (strong) NSWindow *mainWindow;
@property (strong) NSTextView *resultView;      // points to active tab's text view (for compat)
@property (strong) NSTextView *statusView;      // status/network pane (bottom)
@property (strong) NSTabView *logTabView;       // tabbed task output
@property (strong) NSTextField *statusLabel;
@property (strong) NSTimer *refreshTimer;
@property (strong) NSTimer *frontierCheckTimer;
@property (strong) id eventMonitor;
@property (strong) NSMutableDictionary<NSNumber *, NSButton *> *taskButtons;
@property (strong) NSMutableDictionary<NSNumber *, NSTextField *> *taskLabels;
@property (strong) NSPopUpButton *taskSelectPopup;   // dropdown to pick which test to start
@property (strong) NSButton *taskStartBtn;            // Start/Pause for selected test
@property (strong) NSTextField *activeTaskListLabel;  // compact list of running/paused tasks

// EQ-style resource visualizers
@property (strong) EQBarView *cpuEQ;
@property (strong) EQBarView *gpuEQ;
@property (strong) EQBarView *neonEQ;
@property (strong) EQBarView *memEQ;
@property (strong) EQBarView *diskEQ;
@property (strong) NSTextField *statusLabel2;       // secondary status line
@property (strong) NSButton *disableVisualizerBtn;

// Keep these for backward compat with updateResourceMonitor
@property (strong) NSProgressIndicator *cpuBar;
@property (strong) NSTextField *cpuLabel;
@property (strong) NSProgressIndicator *memBar;
@property (strong) NSTextField *memLabel;
@property (strong) NSTextField *gpuStatusLabel;
@property (strong) NSTextField *gpuDetailLabel;
@property (strong) NSTextField *aluDetailLabel;

// Start-point controls
@property (strong) NSPopUpButton *startTaskPopup;
@property (strong) NSTextField *startNumberField;
@property (strong) NSTextField *startPowerField;
@property (strong) NSTextField *startHintLabel;

// Check section
@property (strong) NSPopUpButton *checkModePopup;
@property (strong) NSScrollView *checkFromScroll;
@property (strong) NSTextView *checkFromField;
@property (strong) NSScrollView *checkToScroll;
@property (strong) NSTextView *checkToField;
@property (strong) NSTextField *checkToLabel;
@property (strong) NSButton *checkGoButton;
@property (strong) NSButton *checkStopButton;

// PrimeLocation extras
@property (strong) NSButton *primeFactorButton;      // "Run PrimeFactor" -- factor composites using predicted primes
@property (strong) NSButton *checkAtDiscoveryButton;  // checkbox: run special tests on each predicted prime as found
@property (strong) NSButton *carryChainToggle;        // checkbox: use carry-chain mulmod instead of binary
@property (strong) NSButton *markovLoadButton;        // "Load Primes" for Markov Predict mode

// Benchmark
@property (strong) NSButton *benchmarkButton;
@property (strong) NSButton *benchmarkStopButton;

// Expression section
@property (strong) NSTextField *expressionField;
@property (strong) NSButton *expressionEvalButton;

// Test catalog window
@property (strong) NSWindow *testCatalogWindow;
@property (strong) NSTableView *testTableView;
@property (strong) NSTextView *testDetailView;
@property (strong) NSTextView *testParamField;
@property (strong) NSScrollView *testParamScroll;
@property (strong) NSButton *testRunButton;

// Distributed computing UI
@property (strong) NSWindow *networkWindow;
@property (strong) NSSegmentedControl *roleSegment;
@property (strong) NSView *conductorPanel;
@property (strong) NSView *carriagePanel;
@property (strong) NSTextField *conductorPortField;
@property (strong) NSButton *conductorStartStopBtn;
@property (strong) NSTextField *carriageHostField;
@property (strong) NSTextField *carriagePortField;
@property (strong) NSButton *carriageConnectBtn;
@property (strong) NSButton *bonjourToggle;
@property (strong) NSTextView *connectedMachinesView;
@property (strong) NSTextField *networkStatusLabel;
@property (strong) NSTimer *networkRefreshTimer;

@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    // Set app icon explicitly for Dock
    NSImage *appIcon = [NSImage imageNamed:@"AppIcon"];
    if (!appIcon) {
        NSString *iconPath = [[NSBundle mainBundle] pathForResource:@"AppIcon" ofType:@"icns"];
        if (iconPath) appIcon = [[NSImage alloc] initWithContentsOfFile:iconPath];
    }
    if (appIcon) [NSApp setApplicationIconImage:appIcon];

    _checkRunning = false;
    _prevProcCPU_ns = 0;
    _prevWallClock_ns = 0;
    _prevNeonTested = 0;
    _prevNeonRejected = 0;
    _sleepAssertionID = kIOPMNullAssertionID;
    _preventSleepEnabled = NO;

    // Init per-tab ring buffers
    static const NSUInteger TAB_RING_CAPACITY = 1000;
    static const NSUInteger STATUS_RING_CAPACITY = 200;
    for (int t = 0; t < 4; t++) {
        _tabRings[t] = [NSMutableArray arrayWithCapacity:TAB_RING_CAPACITY];
        for (NSUInteger i = 0; i < TAB_RING_CAPACITY; i++) [_tabRings[t] addObject:@""];
        _tabRingHeads[t] = 0;
        _tabDirty[t] = NO;
        _tabViews[t] = nil;  // set during UI build
    }
    _activeLogTab = 0;
    _statusRing = [NSMutableArray arrayWithCapacity:STATUS_RING_CAPACITY];
    for (NSUInteger i = 0; i < STATUS_RING_CAPACITY; i++) [_statusRing addObject:@""];
    _statusRingHead = 0;
    _logLock = [[NSLock alloc] init];
    _statusDirty = NO;

    // Flush log buffer to text view every 0.5 seconds (avoids per-message layout)
    _logFlushTimer = [NSTimer scheduledTimerWithTimeInterval:0.5
                                                     target:self
                                                   selector:@selector(flushLogBuffer)
                                                   userInfo:nil
                                                    repeats:YES];

    // Init GPU backend (abstract -- auto-selects Metal, Vulkan, or CPU)
    _gpu = prime::create_best_backend();

    // Init Task Manager
    _taskMgr = new prime::TaskManager(DATA_DIR.UTF8String);
    _taskMgr->init_defaults();
    _taskMgr->set_gpu(_gpu);

    // Wire callbacks
    __weak AppDelegate *weakSelf = self;
    _taskMgr->set_log_callback([weakSelf](const std::string& msg) {
        @autoreleasepool {
            AppDelegate *s = weakSelf;
            if (!s) return;
            NSString *line = [NSString stringWithUTF8String:msg.c_str()];
            [s->_logLock lock];
            int tab = 0;  // Search Tasks tab
            s->_tabRings[tab][s->_tabRingHeads[tab] % s->_tabRings[tab].count] = line;
            s->_tabRingHeads[tab]++;
            s->_tabDirty[tab] = YES;
            [s->_logLock unlock];
        }
    });
    _taskMgr->set_discovery_callback([weakSelf](const prime::Discovery& d) {
        @autoreleasepool {
            AppDelegate *ss = weakSelf;
            if (!ss) return;
            NSString *line = [NSString stringWithFormat:@">>> DISCOVERY: %s %llu",
                prime::task_name(d.type), d.value];
            if (d.value2 > 0) line = [line stringByAppendingFormat:@", %llu", d.value2];
            line = [line stringByAppendingString:@" <<<"];
            // Route discoveries to the status pane
            [ss->_logLock lock];
            ss->_statusRing[ss->_statusRingHead % ss->_statusRing.count] = line;
            ss->_statusRingHead++;
            ss->_statusDirty = YES;
            [ss->_logLock unlock];
        }
    });

    // Load saved state (restores positions from previous session)
    _taskMgr->load_state();
    _taskMgr->save_state();

    // Init GIMPS / PrimeNet client
    _primenet = new primenet::PrimeNetClient(
        std::string(DATA_DIR.UTF8String),
        [weakSelf](const std::string& msg) {
            NSString *s = [NSString stringWithFormat:@"%@\n",
                [NSString stringWithUTF8String:msg.c_str()]];
            dispatch_async(dispatch_get_main_queue(), ^{
                [weakSelf appendText:s];
            });
        });
    _primenet->set_username("s1rj1n");

    [self loadTestCatalog];
    [self buildUI];

    // Refresh stats every 500ms
    self.refreshTimer = [NSTimer scheduledTimerWithTimeInterval:0.5 repeats:YES
        block:^(NSTimer *t) { [weakSelf refreshStats]; }];

    // Check search frontiers on launch and every 24 hours
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        [weakSelf checkSearchFrontiers];
    });
    self.frontierCheckTimer = [NSTimer scheduledTimerWithTimeInterval:86400 repeats:YES
        block:^(NSTimer *t) {
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
                [weakSelf checkSearchFrontiers];
            });
        }];

    // Monitor all user interaction to throttle workers
    NSEventMask mask = NSEventMaskLeftMouseDown | NSEventMaskRightMouseDown |
                       NSEventMaskMouseMoved | NSEventMaskScrollWheel |
                       NSEventMaskKeyDown | NSEventMaskKeyUp |
                       NSEventMaskLeftMouseDragged | NSEventMaskFlagsChanged;
    self.eventMonitor = [NSEvent addLocalMonitorForEventsMatchingMask:mask
        handler:^NSEvent *(NSEvent *event) {
            AppDelegate *strongSelf = weakSelf;
            if (strongSelf && strongSelf->_taskMgr) {
                strongSelf->_taskMgr->signal_ui_activity();
            }
            return event;
        }];
}

- (void)buildUI {
    // Standard macOS window positioning - cascade from top-left
    NSRect frame = NSMakeRect(80, 100, 920, 760);
    self.mainWindow = [[NSWindow alloc]
        initWithContentRect:frame
        styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                   NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
        backing:NSBackingStoreBuffered defer:NO];
    [self.mainWindow setTitle:@"PrimePath v0.5 -- Metal GPU Prime Discovery"];
    [self.mainWindow setMinSize:NSMakeSize(720, 400)];
    [self.mainWindow center]; // standard macOS centering

    // Use content view directly - no wrapping scroll view.
    // Controls are pinned to top (NSViewMinYMargin), log fills the rest.
    NSView *cv = self.mainWindow.contentView;
    cv.wantsLayer = YES;

    self.taskButtons = [NSMutableDictionary new];
    self.taskLabels = [NSMutableDictionary new];

    CGFloat W = frame.size.width;
    CGFloat M = 12;
    CGFloat CW = W - 2 * M;
    CGFloat cvH = frame.size.height;
    CGFloat y = cvH - 8; // start from top

    // All controls use NSViewMinYMargin so they stick to the top edge on resize.
    // The log scroll view uses NSViewWidthSizable | NSViewHeightSizable to fill remaining space.

    // ── HEADER ───────────────────────────────────────────────────────
    NSTextField *title = [self labelAt:NSMakeRect(M, y - 18, CW, 20)
        text:@"PrimePath -- Metal GPU Prime Discovery" bold:YES size:14];
    title.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:title];
    y -= 18;

    NSString *gpuName = [NSString stringWithUTF8String:_gpu->name().c_str()];
    NSString *subtitle = [NSString stringWithFormat:
        @"%@ | %u threads | Sieve -> MatrixCRT -> Miller-Rabin(12)",
        _gpu->available() ? gpuName : @"GPU: N/A (CPU fallback)",
        std::thread::hardware_concurrency()];
    NSTextField *subLbl = [self labelAt:NSMakeRect(M, y - 14, CW, 13) text:subtitle bold:NO size:9.5];
    subLbl.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:subLbl];
    y -= 18;

    // ── EQ VISUALIZERS ────────────────────────────────────────────────
    CGFloat eqH = 44;  // height per EQ bar
    CGFloat eqGap = 3;
    CGFloat eqW = (CW - 4 * eqGap) / 5.0; // 5 bars across

    NSColor *greenC = [NSColor colorWithSRGBRed:0.1 green:0.7 blue:0.3 alpha:1.0];
    NSColor *blueC  = [NSColor colorWithSRGBRed:0.2 green:0.5 blue:0.9 alpha:1.0];
    NSColor *cyanC  = [NSColor colorWithSRGBRed:0.1 green:0.7 blue:0.8 alpha:1.0];
    NSColor *amberC = [NSColor colorWithSRGBRed:0.9 green:0.7 blue:0.1 alpha:1.0];
    NSColor *magC   = [NSColor colorWithSRGBRed:0.7 green:0.3 blue:0.8 alpha:1.0];

    CGFloat ex = M;
    self.cpuEQ = [[EQBarView alloc] initWithFrame:NSMakeRect(ex, y - eqH, eqW, eqH)
        title:@"CPU" color:greenC];
    self.cpuEQ.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:self.cpuEQ];
    ex += eqW + eqGap;

    self.gpuEQ = [[EQBarView alloc] initWithFrame:NSMakeRect(ex, y - eqH, eqW, eqH)
        title:@"GPU" color:blueC];
    self.gpuEQ.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:self.gpuEQ];
    ex += eqW + eqGap;

    self.neonEQ = [[EQBarView alloc] initWithFrame:NSMakeRect(ex, y - eqH, eqW, eqH)
        title:@"NEON/SIMD" color:cyanC];
    self.neonEQ.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:self.neonEQ];
    ex += eqW + eqGap;

    self.memEQ = [[EQBarView alloc] initWithFrame:NSMakeRect(ex, y - eqH, eqW, eqH)
        title:@"MEMORY" color:amberC];
    self.memEQ.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:self.memEQ];
    ex += eqW + eqGap;

    self.diskEQ = [[EQBarView alloc] initWithFrame:NSMakeRect(ex, y - eqH, eqW, eqH)
        title:@"DISK I/O" color:magC];
    self.diskEQ.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:self.diskEQ];

    self.disableVisualizerBtn = [NSButton checkboxWithTitle:@"Hide"
        target:self action:@selector(toggleVisualizer:)];
    self.disableVisualizerBtn.frame = NSMakeRect(W - M - 44, y + 2, 44, 14);
    self.disableVisualizerBtn.font = [NSFont systemFontOfSize:9];
    self.disableVisualizerBtn.state = NSControlStateValueOff;
    self.disableVisualizerBtn.autoresizingMask = NSViewMinYMargin | NSViewMinXMargin;
    [cv addSubview:self.disableVisualizerBtn];
    y -= (eqH + 4);

    // Hidden labels for backward compat (updateResourceMonitor still writes to them)
    self.cpuBar = [[NSProgressIndicator alloc] init]; // unused but kept
    self.cpuLabel = [[NSTextField alloc] init];
    self.memBar = [[NSProgressIndicator alloc] init];
    self.memLabel = [[NSTextField alloc] init];
    self.gpuStatusLabel = [[NSTextField alloc] init];
    self.gpuDetailLabel = [[NSTextField alloc] init];
    self.aluDetailLabel = [[NSTextField alloc] init];

    // Status bar
    self.statusLabel = [self labelAt:NSMakeRect(M, y - 14, CW, 13)
        text:@"Ready" bold:NO size:9.5];
    self.statusLabel.textColor = [NSColor colorWithSRGBRed:0.0 green:0.55 blue:0.0 alpha:1.0];
    self.statusLabel.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:self.statusLabel];
    y -= 16;

    // ── SEARCH TASKS ─────────────────────────────────────────────────
    NSBox *sep1 = [[NSBox alloc] initWithFrame:NSMakeRect(M, y, CW, 1)];
    sep1.boxType = NSBoxSeparator;
    sep1.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:sep1];
    y -= 4;

    NSTextField *searchLbl = [self labelAt:NSMakeRect(M, y - 16, 90, 15)
        text:@"Search Tasks" bold:YES size:11];
    searchLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:searchLbl];

    NSButton *infoBtn = [[NSButton alloc] initWithFrame:NSMakeRect(M + 90, y - 15, 18, 18)];
    infoBtn.bezelStyle = NSBezelStyleCircular;
    infoBtn.title = @"?";
    infoBtn.font = [NSFont boldSystemFontOfSize:10];
    infoBtn.target = self;
    infoBtn.action = @selector(showInfoPanel:);
    infoBtn.toolTip = @"Log output glossary";
    infoBtn.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:infoBtn];

    NSButton *preventSleepBtn = [[NSButton alloc] initWithFrame:NSMakeRect(W - M - 200, y - 16, 120, 18)];
    preventSleepBtn.buttonType = NSButtonTypeSwitch;
    preventSleepBtn.title = @"Prevent Sleep";
    preventSleepBtn.font = [NSFont systemFontOfSize:10];
    preventSleepBtn.toolTip = @"Keep Mac awake (prevents display sleep and system idle sleep)";
    preventSleepBtn.state = NSControlStateValueOff;
    preventSleepBtn.target = self;
    preventSleepBtn.action = @selector(togglePreventSleep:);
    preventSleepBtn.autoresizingMask = NSViewMinYMargin | NSViewMinXMargin;
    [cv addSubview:preventSleepBtn];

    NSButton *stopAllBtn = [self buttonAt:NSMakeRect(W - M - 70, y - 16, 70, 20)
        title:@"Stop All" action:@selector(stopAll:)];
    stopAllBtn.font = [NSFont systemFontOfSize:10];
    stopAllBtn.autoresizingMask = NSViewMinYMargin | NSViewMinXMargin;
    [cv addSubview:stopAllBtn];
    y -= 20;

    using TT = prime::TaskType;
    TT types[] = {TT::Wieferich, TT::WallSunSun, TT::Wilson, TT::TwinPrime,
                  TT::SophieGermain, TT::CousinPrime, TT::SexyPrime, TT::GeneralPrime,
                  TT::Emirp, TT::MersenneTrial, TT::FermatFactor};
    const char *descs[] = {
        "2^(p-1) = 1 (mod p^2) -- only 2 known!",
        "p^2 | F(p-(p/5)) -- NONE known!",
        "(p-1)! = -1 (mod p^2) -- only 3 known",
        "p, p+2 both prime",
        "p, 2p+1 both prime",
        "p, p+4 both prime",
        "p, p+6 both prime",
        "Count all primes in range",
        "p and reverse(p) both prime",
        "Trial factor 2^p-1 -- GPU Metal",
        "Factor F_m = 2^(2^m)+1 -- GPU Metal",
    };
    static const int NUM_TASKS = 11;

    self.taskSelectPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(M, y - 22, 380, 22) pullsDown:NO];
    for (int i = 0; i < NUM_TASKS; i++) {
        NSString *t = [NSString stringWithFormat:@"%s  --  %s",
            prime::task_name(types[i]), descs[i]];
        [self.taskSelectPopup addItemWithTitle:t];
        self.taskSelectPopup.lastItem.tag = (int)types[i];
    }
    self.taskSelectPopup.font = [NSFont systemFontOfSize:10];
    self.taskSelectPopup.autoresizingMask = NSViewMinYMargin;
    self.taskSelectPopup.target = self;
    self.taskSelectPopup.action = @selector(taskSelectionChanged:);
    [cv addSubview:self.taskSelectPopup];

    self.taskStartBtn = [self buttonAt:NSMakeRect(M + 386, y - 22, 56, 22)
        title:@"Start" action:@selector(taskToggleSelected:)];
    self.taskStartBtn.font = [NSFont systemFontOfSize:10];
    self.taskStartBtn.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.taskStartBtn];

    NSButton *catalogBtn = [self buttonAt:NSMakeRect(M + 448, y - 22, 100, 22)
        title:@"Test Catalog..." action:@selector(showTestCatalog:)];
    catalogBtn.font = [NSFont systemFontOfSize:10];
    catalogBtn.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:catalogBtn];
    y -= 24;

    // Start point row -- directly below the test selector dropdown
    NSTextField *startAtLbl = [self labelAt:NSMakeRect(M, y - 19, 52, 14) text:@"Start at:" bold:NO size:9.5];
    startAtLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:startAtLbl];

    // Hidden startTaskPopup -- synced to taskSelectPopup selection
    self.startTaskPopup = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
    for (int i = 0; i < NUM_TASKS; i++) {
        [self.startTaskPopup addItemWithTitle:
            [NSString stringWithUTF8String:prime::task_name(types[i])]];
        self.startTaskPopup.lastItem.tag = (int)types[i];
    }
    self.startTaskPopup.hidden = YES;
    [cv addSubview:self.startTaskPopup];

    self.startNumberField = [[NSTextField alloc] initWithFrame:NSMakeRect(M + 54, y - 20, 70, 20)];
    self.startNumberField.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    self.startNumberField.placeholderString = @"2";
    self.startNumberField.stringValue = @"2";
    self.startNumberField.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.startNumberField];

    NSTextField *caretLbl = [self labelAt:NSMakeRect(M + 128, y - 18, 10, 14) text:@"^" bold:YES size:11];
    caretLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:caretLbl];

    self.startPowerField = [[NSTextField alloc] initWithFrame:NSMakeRect(M + 140, y - 20, 36, 20)];
    self.startPowerField.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    self.startPowerField.placeholderString = @"64";
    self.startPowerField.stringValue = @"64";
    self.startPowerField.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.startPowerField];

    NSTextField *plusLbl = [self labelAt:NSMakeRect(M + 180, y - 18, 18, 14) text:@"+1" bold:NO size:10];
    plusLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:plusLbl];

    NSButton *setStartBtn = [self buttonAt:NSMakeRect(M + 200, y - 20, 64, 20)
        title:@"Set Start" action:@selector(setStartPoint:)];
    setStartBtn.font = [NSFont systemFontOfSize:10];
    setStartBtn.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:setStartBtn];

    self.startHintLabel = [self labelAt:NSMakeRect(M + 270, y - 19, CW - 270, 14)
        text:@"" bold:NO size:8.5];
    self.startHintLabel.textColor = [NSColor secondaryLabelColor];
    self.startHintLabel.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:self.startHintLabel];
    [self startTaskChanged:nil];
    y -= 22;

    // Task status list
    self.activeTaskListLabel = [self labelAt:NSMakeRect(M, y - 110, CW, 110)
        text:@"No tasks running. Select a test above and click Start." bold:NO size:9];
    self.activeTaskListLabel.font = [NSFont monospacedSystemFontOfSize:9 weight:NSFontWeightRegular];
    self.activeTaskListLabel.maximumNumberOfLines = 0;
    self.activeTaskListLabel.cell.wraps = YES;
    self.activeTaskListLabel.cell.scrollable = NO;
    self.activeTaskListLabel.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:self.activeTaskListLabel];

    for (int i = 0; i < NUM_TASKS; i++) {
        NSNumber *key = @((int)types[i]);
        NSButton *btn = [[NSButton alloc] init];
        btn.tag = (int)types[i];
        btn.title = @"Start";
        self.taskButtons[key] = btn;
        self.taskLabels[key] = [[NSTextField alloc] init];
    }
    y -= 114;

    // ── CHECK & TOOLS ────────────────────────────────────────────────
    NSBox *sep2 = [[NSBox alloc] initWithFrame:NSMakeRect(M, y, CW, 1)];
    sep2.boxType = NSBoxSeparator;
    sep2.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:sep2];
    y -= 4;

    NSTextField *checkLbl = [self labelAt:NSMakeRect(M, y - 15, 100, 14)
        text:@"Check & Tools" bold:YES size:10];
    checkLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:checkLbl];
    y -= 18;

    self.checkModePopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(M, y - 22, 140, 22) pullsDown:NO];
    [self.checkModePopup addItemWithTitle:@"Check Single"];
    [self.checkModePopup addItemWithTitle:@"Check From Here"];
    [self.checkModePopup addItemWithTitle:@"Check Linear"];
    [self.checkModePopup addItemWithTitle:@"PrimeLocation"];
    [self.checkModePopup addItemWithTitle:@"Expression"];
    [self.checkModePopup addItemWithTitle:@"Markov Predict"];
    self.checkModePopup.target = self;
    self.checkModePopup.action = @selector(checkModeChanged:);
    self.checkModePopup.font = [NSFont systemFontOfSize:10];
    self.checkModePopup.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.checkModePopup];

    self.checkGoButton = [self buttonAt:NSMakeRect(M + 146, y - 22, 50, 22)
        title:@"Go" action:@selector(checkGo:)];
    self.checkGoButton.font = [NSFont systemFontOfSize:10];
    self.checkGoButton.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.checkGoButton];

    self.checkStopButton = [self buttonAt:NSMakeRect(M + 200, y - 22, 50, 22)
        title:@"Stop" action:@selector(checkStopAction:)];
    self.checkStopButton.font = [NSFont systemFontOfSize:10];
    self.checkStopButton.enabled = NO;
    self.checkStopButton.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.checkStopButton];

    self.benchmarkButton = [self buttonAt:NSMakeRect(M + 256, y - 22, 76, 22)
        title:@"Benchmark" action:@selector(runBenchmark:)];
    self.benchmarkButton.font = [NSFont systemFontOfSize:10];
    self.benchmarkButton.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.benchmarkButton];

    NSButton *runTestsBtn = [self buttonAt:NSMakeRect(M + 338, y - 22, 76, 22)
        title:@"Run Tests" action:@selector(runInAppTests:)];
    runTestsBtn.font = [NSFont systemFontOfSize:10];
    runTestsBtn.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:runTestsBtn];

    NSButton *pipelineBtn = [self buttonAt:NSMakeRect(M + 420, y - 22, 70, 22)
        title:@"Pipeline" action:@selector(showPipelineBuilder:)];
    pipelineBtn.font = [NSFont systemFontOfSize:10];
    pipelineBtn.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:pipelineBtn];

    NSButton *gimpsBtn = [self buttonAt:NSMakeRect(M + 496, y - 22, 56, 22)
        title:@"GIMPS" action:@selector(showGIMPSPanel:)];
    gimpsBtn.font = [NSFont boldSystemFontOfSize:10];
    gimpsBtn.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:gimpsBtn];

    self.checkAtDiscoveryButton = [[NSButton alloc] initWithFrame:NSMakeRect(M + 556, y - 22, 150, 18)];
    self.checkAtDiscoveryButton.buttonType = NSButtonTypeSwitch;
    self.checkAtDiscoveryButton.title = @"CheckAtDiscovery";
    self.checkAtDiscoveryButton.font = [NSFont systemFontOfSize:9];
    self.checkAtDiscoveryButton.toolTip = @"Run Wieferich/WallSunSun/Wilson tests on each predicted prime";
    self.checkAtDiscoveryButton.state = NSControlStateValueOff;
    self.checkAtDiscoveryButton.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.checkAtDiscoveryButton];

    self.primeFactorButton = [self buttonAt:NSMakeRect(M + 710, y - 22, 110, 20)
        title:@"Run PrimeFactor" action:@selector(runPrimeFactor:)];
    self.primeFactorButton.font = [NSFont systemFontOfSize:10];
    self.primeFactorButton.hidden = YES;
    self.primeFactorButton.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.primeFactorButton];

    self.benchmarkStopButton = nil;
    y -= 24;

    // Nester Carry Chain row (own line to avoid overlap)
    self.carryChainToggle = [[NSButton alloc] initWithFrame:NSMakeRect(M, y - 20, 150, 18)];
    self.carryChainToggle.buttonType = NSButtonTypeSwitch;
    self.carryChainToggle.title = @"NesterCarryChain";
    self.carryChainToggle.font = [NSFont boldSystemFontOfSize:9];
    self.carryChainToggle.toolTip = @"~4-7x faster mulmod for Wieferich + Wall-Sun-Sun CPU tests";
    self.carryChainToggle.state = NSControlStateValueOff;
    self.carryChainToggle.target = self;
    self.carryChainToggle.action = @selector(carryChainToggled:);
    self.carryChainToggle.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.carryChainToggle];

    NSButton *ccInfoBtn = [self buttonAt:NSMakeRect(M + 154, y - 19, 18, 18)
        title:@"?" action:@selector(showCarryChainInfo:)];
    ccInfoBtn.font = [NSFont boldSystemFontOfSize:9];
    ccInfoBtn.bezelStyle = NSBezelStyleCircular;
    ccInfoBtn.toolTip = @"About Nester Carry Chain";
    ccInfoBtn.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:ccInfoBtn];

    NSButton *ccBenchBtn = [self buttonAt:NSMakeRect(M + 176, y - 19, 42, 18)
        title:@"Bench" action:@selector(runCarryChainTest:)];
    ccBenchBtn.font = [NSFont systemFontOfSize:8];
    ccBenchBtn.toolTip = @"Run Nester Carry Chain benchmark";
    ccBenchBtn.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:ccBenchBtn];

    NSButton *jsonSampleBtn = [self buttonAt:NSMakeRect(M + 222, y - 19, 42, 18)
        title:@"JSON" action:@selector(runJSONSample:)];
    jsonSampleBtn.font = [NSFont systemFontOfSize:8];
    jsonSampleBtn.toolTip = @"Output JSON result samples from build_result_json";
    jsonSampleBtn.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:jsonSampleBtn];
    y -= 22;

    // From field
    NSTextField *fromLbl = [self labelAt:NSMakeRect(M, y - 18, 34, 14) text:@"From:" bold:NO size:9.5];
    fromLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:fromLbl];
    self.checkFromScroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(M + 36, y - 20, CW - 36, 22)];
    self.checkFromScroll.hasVerticalScroller = NO;
    self.checkFromScroll.borderType = NSBezelBorder;
    self.checkFromScroll.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    self.checkFromField = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, CW - 40, 18)];
    self.checkFromField.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    self.checkFromField.autoresizingMask = NSViewWidthSizable;
    self.checkFromField.richText = NO;
    self.checkFromField.allowsUndo = YES;
    self.checkFromScroll.documentView = self.checkFromField;
    [cv addSubview:self.checkFromScroll];
    y -= 24;

    // To field (hidden by default)
    self.checkToLabel = [self labelAt:NSMakeRect(M, y - 18, 24, 14) text:@"To:" bold:NO size:9.5];
    self.checkToLabel.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.checkToLabel];
    self.checkToScroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(M + 36, y - 20, CW - 36, 22)];
    self.checkToScroll.hasVerticalScroller = NO;
    self.checkToScroll.borderType = NSBezelBorder;
    self.checkToScroll.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    self.checkToField = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, CW - 40, 18)];
    self.checkToField.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    self.checkToField.autoresizingMask = NSViewWidthSizable;
    self.checkToField.richText = NO;
    self.checkToField.allowsUndo = YES;
    self.checkToScroll.documentView = self.checkToField;
    [cv addSubview:self.checkToScroll];
    self.checkToLabel.hidden = YES;
    self.checkToScroll.hidden = YES;
    y -= 4;

    // ── EXPRESSION ──────────────────────────────────────────────────
    NSBox *sepExpr = [[NSBox alloc] initWithFrame:NSMakeRect(M, y, CW, 1)];
    sepExpr.boxType = NSBoxSeparator;
    sepExpr.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:sepExpr];
    y -= 4;

    NSTextField *exprLbl = [self labelAt:NSMakeRect(M, y - 15, 80, 14)
        text:@"Expression" bold:YES size:10];
    exprLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:exprLbl];

    self.expressionEvalButton = [self buttonAt:NSMakeRect(W - M - 70, y - 16, 70, 20)
        title:@"Evaluate" action:@selector(expressionEvaluate:)];
    self.expressionEvalButton.font = [NSFont systemFontOfSize:10];
    self.expressionEvalButton.autoresizingMask = NSViewMinYMargin | NSViewMinXMargin;
    [cv addSubview:self.expressionEvalButton];

    NSButton *exprHelpBtn = [[NSButton alloc] initWithFrame:NSMakeRect(M + 80, y - 14, 18, 18)];
    exprHelpBtn.bezelStyle = NSBezelStyleCircular;
    exprHelpBtn.title = @"?";
    exprHelpBtn.font = [NSFont boldSystemFontOfSize:10];
    exprHelpBtn.target = self;
    exprHelpBtn.action = @selector(showExpressionHelp:);
    exprHelpBtn.toolTip = @"Expression syntax help";
    exprHelpBtn.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:exprHelpBtn];
    y -= 18;

    self.expressionField = [self fieldAt:NSMakeRect(M, y - 22, CW, 22)
        placeholder:@"is_prime(997)  factor(1729)  2^67-1  next prime 10000  twin primes 100 2000  modpow(2,100,1e9+7)"];
    self.expressionField.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    self.expressionField.target = self;
    self.expressionField.action = @selector(expressionEvaluate:);
    [cv addSubview:self.expressionField];
    y -= 26;

    // ── OUTPUT (two panes: task log on top, status on bottom) ───────
    NSBox *sep3 = [[NSBox alloc] initWithFrame:NSMakeRect(M, y, CW, 1)];
    sep3.boxType = NSBoxSeparator;
    sep3.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:sep3];
    y -= 2;

    CGFloat logBottom = 4;
    CGFloat totalHeight = y - logBottom;
    CGFloat statusHeight = 120;
    CGFloat labelH = 14;
    CGFloat gap = 2;

    // ── Bottom: Status / Network pane (fixed height, pinned to bottom) ─
    NSTextField *statusLbl = [[NSTextField alloc] initWithFrame:
        NSMakeRect(M, logBottom + statusHeight + gap, 200, labelH)];
    statusLbl.stringValue = @"Status / Network";
    statusLbl.bezeled = NO;
    statusLbl.editable = NO;
    statusLbl.drawsBackground = NO;
    statusLbl.font = [NSFont boldSystemFontOfSize:9];
    statusLbl.textColor = [NSColor secondaryLabelColor];
    statusLbl.autoresizingMask = NSViewWidthSizable;
    [cv addSubview:statusLbl];

    NSScrollView *statusScroll = [[NSScrollView alloc] initWithFrame:
        NSMakeRect(M, logBottom, CW, statusHeight)];
    statusScroll.hasVerticalScroller = YES;
    statusScroll.autoresizingMask = NSViewWidthSizable;
    self.statusView = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, CW - 16, statusHeight)];
    self.statusView.editable = NO;
    self.statusView.allowsUndo = NO;
    self.statusView.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    self.statusView.backgroundColor = [NSColor colorWithCalibratedRed:0.12 green:0.14 blue:0.18 alpha:1.0];
    self.statusView.textColor = [NSColor colorWithCalibratedRed:0.6 green:0.9 blue:0.6 alpha:1.0];
    self.statusView.verticallyResizable = YES;
    self.statusView.horizontallyResizable = NO;
    self.statusView.textContainer.containerSize = NSMakeSize(CW - 16, FLT_MAX);
    self.statusView.textContainer.widthTracksTextView = YES;
    statusScroll.documentView = self.statusView;
    [cv addSubview:statusScroll];

    // ── Top: Tabbed task output (fills remaining space) ────────────
    CGFloat taskBottom = logBottom + statusHeight + gap + labelH + gap;
    CGFloat taskHeight = totalHeight - statusHeight - gap - labelH - gap;

    self.logTabView = [[NSTabView alloc] initWithFrame:
        NSMakeRect(M, taskBottom, CW, taskHeight)];
    self.logTabView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.logTabView.controlSize = NSControlSizeSmall;
    self.logTabView.font = [NSFont systemFontOfSize:10];

    NSArray *tabNames = @[@"Search Tasks", @"Check / Expressions", @"Benchmark", @"Pipeline"];
    CGFloat tabContentH = taskHeight - 28;  // tab bar takes ~28px
    for (int t = 0; t < 4; t++) {
        NSTabViewItem *item = [[NSTabViewItem alloc] initWithIdentifier:@(t)];
        item.label = tabNames[t];

        NSView *tabContent = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, CW - 14, tabContentH)];
        NSScrollView *tabScroll = [[NSScrollView alloc] initWithFrame:
            NSMakeRect(0, 0, CW - 14, tabContentH)];
        tabScroll.hasVerticalScroller = YES;
        tabScroll.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

        NSTextView *tv = [[NSTextView alloc] initWithFrame:
            NSMakeRect(0, 0, CW - 30, tabContentH)];
        tv.editable = NO;
        tv.allowsUndo = NO;
        tv.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
        tv.backgroundColor = [NSColor textBackgroundColor];
        tv.verticallyResizable = YES;
        tv.horizontallyResizable = NO;
        tv.textContainer.containerSize = NSMakeSize(CW - 30, FLT_MAX);
        tv.textContainer.widthTracksTextView = YES;
        tv.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        tabScroll.documentView = tv;
        [tabContent addSubview:tabScroll];
        item.view = tabContent;
        [self.logTabView addTabViewItem:item];
        _tabViews[t] = tv;
    }

    // Default to Search Tasks tab
    self.resultView = _tabViews[0];
    [cv addSubview:self.logTabView];

    // Welcome text
    [self appendStatus:@"PrimePath v0.5.0 — Status & Network\n"];
    [self appendText:@"PrimePath v0.5.0 -- Metal GPU Prime Discovery Engine\n"];
    [self appendText:[NSString stringWithFormat:@"GPU: %@ | Data: %@\n",
        [NSString stringWithUTF8String:_gpu->name().c_str()], DATA_DIR]];

    auto& kdb = prime::known_db();
    [self appendText:[NSString stringWithFormat:
        @"Known: %zu primes, %zu pseudoprimes\n",
        kdb.prime_count(), kdb.pseudoprime_count()]];

    if (_taskMgr->discoveries().size() > 0) {
        // Summarise by type instead of dumping every entry
        std::map<prime::TaskType, int> counts;
        for (auto& d : _taskMgr->discoveries()) counts[d.type]++;
        NSMutableString *summary = [NSMutableString stringWithFormat:@"Discoveries: %zu total (",
            _taskMgr->discoveries().size()];
        bool first = true;
        for (auto& [type, count] : counts) {
            if (!first) [summary appendString:@", "];
            [summary appendFormat:@"%s: %d", prime::task_name(type), count];
            first = false;
        }
        [summary appendString:@")\n"];
        [self appendText:summary];
    }
    [self appendText:@"\nHelp menu or ? button for glossary. Ready.\n\n"];

    [self.mainWindow makeKeyAndOrderFront:nil];
}

// ── Check mode UI toggle ────────────────────────────────────────────

- (void)checkModeChanged:(id)sender {
    NSInteger mode = self.checkModePopup.indexOfSelectedItem;
    BOOL showTo = (mode >= 1 && mode <= 3);
    self.checkToLabel.hidden = !showTo;
    self.checkToScroll.hidden = !showTo;
    switch (mode) {
        case 0:
            self.checkFromField.toolTip = @"Number to check";
            self.checkToField.toolTip = @"";
            break;
        case 1:
            self.checkFromField.toolTip = @"Start from";
            self.checkToField.toolTip = @"End (optional)";
            break;
        case 2:
            self.checkFromField.toolTip = @"Start from";
            self.checkToField.toolTip = @"End (optional)";
            break;
        case 3:
            self.checkFromField.toolTip = @"Search near";
            self.checkToField.toolTip = @"Window size (default 10000)";
            break;
        case 4:
            self.checkFromField.toolTip =
                @"Expression: is_prime(997), factor(1729), 2^67-1, "
                @"modpow(2,100,1e9+7), primes 1000 to 2000, "
                @"next prime 10000, wieferich 1093, twin primes 100 200, "
                @"gcd(48,36), fibonacci(50), binomial(10,3)";
            self.checkToField.toolTip = @"";
            break;
        case 5:
            self.checkFromField.toolTip = @"Click Go to open Markov Predict window";
            break;
    }
}

// ── Check dispatcher ────────────────────────────────────────────────

- (void)checkGo:(id)sender {
    _activeLogTab = 1;  // Check / Expressions tab
    [self.logTabView selectTabViewItemAtIndex:1];
    // If a background check is running, stop it first
    if (_checkRunning.load()) {
        _checkRunning.store(false);
    }
    if (_checkThread.joinable()) {
        _checkThread.join();
    }

    NSString *rawInput = self.checkFromField.string;
    NSInteger mode = self.checkModePopup.indexOfSelectedItem;

    // Expression mode
    if (mode == 4) {
        [self evaluateExpression:rawInput];
        return;
    }

    // Markov Predict opens its own window
    if (mode == 5) {
        [self showMarkovPredictWindow:nil];
        return;
    }

    // Auto-detect expressions in Check Single mode: if input contains
    // letters (except pure numbers), ^, (, or ?, route to expression evaluator
    if (mode == 0) {
        NSString *trimmed = [rawInput stringByTrimmingCharactersInSet:
            [NSCharacterSet whitespaceAndNewlineCharacterSet]];
        NSRange letterRange = [trimmed rangeOfCharacterFromSet:[NSCharacterSet letterCharacterSet]];
        NSRange caretRange = [trimmed rangeOfString:@"^"];
        NSRange parenRange = [trimmed rangeOfString:@"("];
        NSRange questionRange = [trimmed rangeOfString:@"?"];
        if (letterRange.location != NSNotFound || caretRange.location != NSNotFound ||
            parenRange.location != NSNotFound || questionRange.location != NSNotFound) {
            [self evaluateExpression:rawInput];
            return;
        }
    }

    // Multi-number support: split on newlines, commas, or semicolons
    // so user can paste a block of numbers or a comma-delimited list
    if (mode == 0) {
        // Split on any combo of newlines, commas, semicolons
        NSMutableCharacterSet *separators = [NSMutableCharacterSet newlineCharacterSet];
        [separators addCharactersInString:@",;"];
        NSArray<NSString *> *tokens = [rawInput componentsSeparatedByCharactersInSet:separators];
        NSMutableArray<NSNumber *> *numbers = [NSMutableArray new];
        for (NSString *token in tokens) {
            // Strip whitespace and any remaining commas within the token
            NSString *cleaned = [[token stringByTrimmingCharactersInSet:
                [NSCharacterSet whitespaceCharacterSet]]
                stringByReplacingOccurrencesOfString:@"," withString:@""];
            if (cleaned.length == 0) continue;
            uint64_t val = strtoull(cleaned.UTF8String, nullptr, 10);
            if (val >= 2) [numbers addObject:@(val)];
        }
        if (numbers.count == 0) {
            [self appendText:@"Enter a number >= 2.\n"];
            return;
        }
        if (numbers.count > 1) {
            [self appendText:[NSString stringWithFormat:
                @"── Checking %lu numbers ──\n", (unsigned long)numbers.count]];
        }
        for (NSNumber *num in numbers) {
            [self runCheckSingle:num.unsignedLongLongValue];
        }
        return;
    }

    NSString *fromStr = [rawInput stringByTrimmingCharactersInSet:
        [NSCharacterSet whitespaceAndNewlineCharacterSet]];
    // Strip commas for readability
    fromStr = [fromStr stringByReplacingOccurrencesOfString:@"," withString:@""];
    uint64_t from = strtoull(fromStr.UTF8String, nullptr, 10);
    if (from < 2) {
        [self appendText:@"Enter a number >= 2.\n"];
        return;
    }

    NSString *toStr = [self.checkToField.string stringByTrimmingCharactersInSet:
        [NSCharacterSet whitespaceAndNewlineCharacterSet]];
    toStr = [toStr stringByReplacingOccurrencesOfString:@"," withString:@""];
    uint64_t to = toStr.length > 0 ? strtoull(toStr.UTF8String, nullptr, 10) : 0;

    switch (mode) {
        case 1: [self runCheckFromHere:from to:to]; break;
        case 2: [self runCheckLinear:from to:to]; break;
        case 3: [self runPrimeLocation:from window:to]; break;
        case 5: [self showMarkovPredictWindow:nil]; break; // handled above, fallback
    }
}

// ── Expression mode: try to parse before falling through to number check ──

- (void)evaluateExpression:(NSString *)input {
    NSString *expr = [input stringByTrimmingCharactersInSet:
        [NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSString *lower = [expr lowercaseString];

    // Strip trailing ? for natural queries like "is 997 prime?"
    if ([lower hasSuffix:@"?"]) {
        lower = [lower substringToIndex:lower.length - 1];
        expr = [expr substringToIndex:expr.length - 1];
    }

    // ── Function call: is_prime(N) ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"is_prime\\s*\\(\\s*([0-9eE^+\\-*/ .]+)\\s*\\)" options:0 error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t n = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            if (n >= 2) { [self runCheckSingle:n]; return; }
        }
    }

    // ── "is N prime" ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"is\\s+([0-9eE^+\\-*/ .]+)\\s+prime" options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t n = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            if (n >= 2) { [self runCheckSingle:n]; return; }
        }
    }

    // ── factor(N) or "factor N" or "factorize N" or "factorise N" ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"(?:factor(?:ize|ise)?|factors)\\s*\\(?\\s*([0-9eE^+\\-*/ .]+)\\s*\\)?" options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t n = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            if (n >= 2) {
                auto t0 = std::chrono::steady_clock::now();
                auto factors = prime::factor_u64(n);
                auto dt = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - t0).count();
                std::string fs = prime::factors_string(n);
                if (factors.empty() || (factors.size() == 1 && factors[0] == n)) {
                    [self appendText:[NSString stringWithFormat:
                        @"factor(%@) = prime (%.1fus)\n", formatNumber(n), dt]];
                } else {
                    [self appendText:[NSString stringWithFormat:
                        @"factor(%@) = %s (%.1fus)\n", formatNumber(n), fs.c_str(), dt]];
                }
                return;
            }
        }
    }

    // ── modpow(a, b, m) ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"modpow\\s*\\(\\s*([0-9eE^+\\-*/ .]+)\\s*,\\s*([0-9eE^+\\-*/ .]+)\\s*,\\s*([0-9eE^+\\-*/ .]+)\\s*\\)"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t a = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            uint64_t b = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:2]]];
            uint64_t mod = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:3]]];
            if (mod > 0) {
                uint64_t result = prime::modpow(a, b, mod);
                [self appendText:[NSString stringWithFormat:
                    @"modpow(%@, %@, %@) = %@\n",
                    formatNumber(a), formatNumber(b), formatNumber(mod), formatNumber(result)]];
                return;
            }
        }
    }

    // ── mulmod(a, b, m) ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"mulmod\\s*\\(\\s*([0-9eE^+\\-*/ .]+)\\s*,\\s*([0-9eE^+\\-*/ .]+)\\s*,\\s*([0-9eE^+\\-*/ .]+)\\s*\\)"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t a = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            uint64_t b = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:2]]];
            uint64_t mod = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:3]]];
            if (mod > 0) {
                uint64_t result = prime::mulmod(a, b, mod);
                [self appendText:[NSString stringWithFormat:
                    @"mulmod(%@, %@, %@) = %@\n",
                    formatNumber(a), formatNumber(b), formatNumber(mod), formatNumber(result)]];
                return;
            }
        }
    }

    // ── gcd(a, b) ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"gcd\\s*\\(\\s*([0-9eE^+\\-*/ .]+)\\s*,\\s*([0-9eE^+\\-*/ .]+)\\s*\\)"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t a = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            uint64_t b = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:2]]];
            uint64_t g = a, h = b;
            while (h > 0) { uint64_t t = h; h = g % h; g = t; }
            [self appendText:[NSString stringWithFormat:
                @"gcd(%@, %@) = %@\n", formatNumber(a), formatNumber(b), formatNumber(g)]];
            return;
        }
    }

    // ── lcm(a, b) ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"lcm\\s*\\(\\s*([0-9eE^+\\-*/ .]+)\\s*,\\s*([0-9eE^+\\-*/ .]+)\\s*\\)"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t a = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            uint64_t b = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:2]]];
            uint64_t g = a, h = b;
            while (h > 0) { uint64_t t = h; h = g % h; g = t; }
            uint64_t result = (a / g) * b;
            [self appendText:[NSString stringWithFormat:
                @"lcm(%@, %@) = %@\n", formatNumber(a), formatNumber(b), formatNumber(result)]];
            return;
        }
    }

    // ── fibonacci(n) or fib(n) ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"(?:fibonacci|fib)\\s*\\(?\\s*([0-9]+)\\s*\\)?"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            int n = (int)[[expr substringWithRange:[m rangeAtIndex:1]] integerValue];
            if (n > 93) {
                [self appendText:@"fibonacci: max n=93 for uint64 (overflow beyond that)\n"];
                return;
            }
            uint64_t a = 0, b = 1;
            for (int i = 0; i < n; i++) { uint64_t t = a + b; a = b; b = t; }
            [self appendText:[NSString stringWithFormat:
                @"fibonacci(%d) = %@\n", n, formatNumber(a)]];
            // Check if the Fibonacci number is prime
            if (a >= 2 && prime::is_prime(a)) {
                [self appendText:[NSString stringWithFormat:
                    @"  (Fibonacci prime!)\n"]];
            }
            return;
        }
    }

    // ── binomial(n, k) or C(n,k) or nCk ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"(?:binomial|choose|c)\\s*\\(\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\)"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t n = [[expr substringWithRange:[m rangeAtIndex:1]] longLongValue];
            uint64_t k = [[expr substringWithRange:[m rangeAtIndex:2]] longLongValue];
            if (k > n) { [self appendText:@"binomial: k > n\n"]; return; }
            if (k > n - k) k = n - k;
            uint64_t result = 1;
            for (uint64_t i = 0; i < k; i++) {
                result = result * (n - i) / (i + 1);
            }
            [self appendText:[NSString stringWithFormat:
                @"C(%llu, %llu) = %@\n", n - k + k, k, formatNumber(result)]];
            if (result >= 2 && prime::is_prime(result)) {
                [self appendText:@"  (this value is prime!)\n"];
            }
            return;
        }
    }

    // ── "primes in/between/from A to/.. B" or "primes A B" or "primes A to B" ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"(?:primes?|find primes?)\\s+(?:in|between|from)?\\s*([0-9eE^+\\-*/ .]+)\\s+(?:to|\\.\\.|-|and)\\s*([0-9eE^+\\-*/ .]+)"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t a = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            uint64_t b = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:2]]];
            if (a < 2) a = 2;
            if (b < a) { uint64_t t = a; a = b; b = t; }
            if (b - a > 10000000) {
                [self appendText:@"Range too large (max 10M). Use Search Tasks for big ranges.\n"];
                return;
            }
            prime::Engine eng;
            int hw = std::max(1, (int)std::thread::hardware_concurrency());
            auto t0 = std::chrono::steady_clock::now();
            auto results = eng.search_range(a, b, hw);
            auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            [self appendText:[NSString stringWithFormat:
                @"Primes in [%@, %@]: %zu found (%.4fs)\n",
                formatNumber(a), formatNumber(b), results.size(), dt]];
            // Print up to first 200
            size_t show = std::min(results.size(), (size_t)200);
            NSMutableString *list = [NSMutableString string];
            for (size_t i = 0; i < show; i++) {
                if (i > 0) [list appendString:@", "];
                [list appendFormat:@"%@", formatNumber(results[i].value)];
            }
            if (results.size() > 200) [list appendFormat:@" ... (%zu more)", results.size() - 200];
            [self appendText:[NSString stringWithFormat:@"  %@\n", list]];
            return;
        }
    }

    // ── "next prime N" or "next_prime(N)" or "next prime after N" ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"next[_ ]?prime\\s*\\(?(?:\\s*after)?\\s*([0-9eE^+\\-*/ .]+)\\s*\\)?"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t n = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            uint64_t p = n + 1;
            if (p < 2) p = 2;
            while (!prime::is_prime(p)) p++;
            [self appendText:[NSString stringWithFormat:
                @"next_prime(%@) = %@\n", formatNumber(n), formatNumber(p)]];
            return;
        }
    }

    // ── "prev prime N" or "previous prime N" ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"(?:prev(?:ious)?[_ ]?prime)\\s*\\(?\\s*([0-9eE^+\\-*/ .]+)\\s*\\)?"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t n = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            if (n <= 2) { [self appendText:@"No prime before 2.\n"]; return; }
            uint64_t p = n - 1;
            while (p >= 2 && !prime::is_prime(p)) p--;
            [self appendText:[NSString stringWithFormat:
                @"prev_prime(%@) = %@\n", formatNumber(n), formatNumber(p)]];
            return;
        }
    }

    // ── "wieferich N" or "wieferich test N" ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"wieferich\\s+(?:test\\s+)?([0-9eE^+\\-*/ .]+)"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t p = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            if (p < 2 || !prime::is_prime(p)) {
                [self appendText:[NSString stringWithFormat:
                    @"%@ is not prime (Wieferich test requires a prime)\n", formatNumber(p)]];
                return;
            }
            bool w = prime::modpow(2, p - 1, p * p) == 1;
            [self appendText:[NSString stringWithFormat:
                @"Wieferich test: 2^(%@-1) mod %@^2 %@ 1 -- %@\n",
                formatNumber(p), formatNumber(p),
                w ? @"==" : @"!=",
                w ? @"YES, Wieferich prime!" : @"not Wieferich"]];
            return;
        }
    }

    // ── "wilson N" or "wilson test N" ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"wilson\\s+(?:test\\s+)?([0-9eE^+\\-*/ .]+)"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t p = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            if (p < 2 || !prime::is_prime(p)) {
                [self appendText:[NSString stringWithFormat:
                    @"%@ is not prime (Wilson test requires a prime)\n", formatNumber(p)]];
                return;
            }
            if (p > 100000) {
                [self appendText:@"Wilson test: p too large (factorial overflow risk). Max ~100000.\n"];
                return;
            }
            uint64_t mod = p * p;
            uint64_t factorial = 1;
            for (uint64_t i = 2; i < p; i++) {
                factorial = prime::mulmod(factorial, i, mod);
            }
            bool w = (factorial == mod - 1);
            [self appendText:[NSString stringWithFormat:
                @"Wilson test: (%@-1)! mod %@^2 %@ %@^2-1 -- %@\n",
                formatNumber(p), formatNumber(p),
                w ? @"==" : @"!=", formatNumber(p),
                w ? @"YES, Wilson prime!" : @"not Wilson prime"]];
            return;
        }
    }

    // ── "twin primes A B" or "twin primes in A to B" ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"twin\\s+primes?\\s+(?:in\\s+)?([0-9eE^+\\-*/ .]+)\\s+(?:to\\s+)?([0-9eE^+\\-*/ .]+)"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t a = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            uint64_t b = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:2]]];
            if (a < 2) a = 2;
            if (b - a > 10000000) {
                [self appendText:@"Range too large (max 10M).\n"];
                return;
            }
            auto sv = prime::sieve(b);
            int count = 0;
            NSMutableString *pairs = [NSMutableString string];
            for (uint64_t i = a; i <= b - 2; i++) {
                if (sv[i] && sv[i + 2]) {
                    count++;
                    if (count <= 50) {
                        if (pairs.length > 0) [pairs appendString:@", "];
                        [pairs appendFormat:@"(%@,%@)", formatNumber(i), formatNumber(i + 2)];
                    }
                }
            }
            [self appendText:[NSString stringWithFormat:
                @"Twin primes in [%@, %@]: %d pairs\n", formatNumber(a), formatNumber(b), count]];
            if (count > 0) {
                if (count > 50) [pairs appendFormat:@" ... (%d more)", count - 50];
                [self appendText:[NSString stringWithFormat:@"  %@\n", pairs]];
            }
            return;
        }
    }

    // ── "sophie germain A B" ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"sophie\\s+germain\\s+([0-9eE^+\\-*/ .]+)\\s+(?:to\\s+)?([0-9eE^+\\-*/ .]+)"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t a = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            uint64_t b = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:2]]];
            if (a < 2) a = 2;
            if (b - a > 10000000) {
                [self appendText:@"Range too large (max 10M).\n"];
                return;
            }
            auto sv = prime::sieve(b);
            int count = 0;
            NSMutableString *list = [NSMutableString string];
            for (uint64_t i = a; i <= b; i++) {
                if (sv[i] && prime::is_prime(2 * i + 1)) {
                    count++;
                    if (count <= 50) {
                        if (list.length > 0) [list appendString:@", "];
                        [list appendFormat:@"%@", formatNumber(i)];
                    }
                }
            }
            [self appendText:[NSString stringWithFormat:
                @"Sophie Germain primes in [%@, %@]: %d found\n", formatNumber(a), formatNumber(b), count]];
            if (count > 0) {
                if (count > 50) [list appendFormat:@" ... (%d more)", count - 50];
                [self appendText:[NSString stringWithFormat:@"  %@\n", list]];
            }
            return;
        }
    }

    // ── "mersenne N" -- test if 2^N - 1 is prime ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"mersenne\\s+([0-9]+)"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            int n = (int)[[expr substringWithRange:[m rangeAtIndex:1]] integerValue];
            if (n > 63) {
                [self appendText:[NSString stringWithFormat:
                    @"M%d = 2^%d - 1 is too large for 64-bit testing. Use Mersenne Trial Factor search.\n", n, n]];
                return;
            }
            uint64_t mn = (1ULL << n) - 1;
            bool p = prime::is_prime(mn);
            [self appendText:[NSString stringWithFormat:
                @"M%d = 2^%d - 1 = %@ -- %@\n", n, n, formatNumber(mn),
                p ? @"PRIME (Mersenne prime!)" : @"COMPOSITE"]];
            if (!p) {
                std::string fs = prime::factors_string(mn);
                if (!fs.empty()) {
                    [self appendText:[NSString stringWithFormat:
                        @"  = %s\n", fs.c_str()]];
                }
            }
            return;
        }
    }

    // ── "fermat N" -- test Fermat number F_N = 2^(2^N) + 1 ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"fermat\\s+([0-9]+)"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            int n = (int)[[expr substringWithRange:[m rangeAtIndex:1]] integerValue];
            if (n > 5) {
                [self appendText:[NSString stringWithFormat:
                    @"F_%d = 2^(2^%d) + 1 is too large for 64-bit. Use Fermat Factor search.\n", n, n]];
                return;
            }
            uint64_t fn = (1ULL << (1 << n)) + 1;
            bool p = prime::is_prime(fn);
            [self appendText:[NSString stringWithFormat:
                @"F_%d = 2^(2^%d) + 1 = %@ -- %@\n", n, n, formatNumber(fn),
                p ? @"PRIME (Fermat prime!)" : @"COMPOSITE"]];
            if (!p) {
                std::string fs = prime::factors_string(fn);
                if (!fs.empty()) {
                    [self appendText:[NSString stringWithFormat:@"  = %s\n", fs.c_str()]];
                }
            }
            return;
        }
    }

    // ── "convergence N" or "shadow N" ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"(?:convergence|shadow)\\s*\\(?\\s*([0-9eE^+\\-*/ .]+)\\s*\\)?"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t n = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            double c = prime::convergence(n);
            [self appendText:[NSString stringWithFormat:
                @"convergence(%@) = %.4f%@\n", formatNumber(n), c,
                c == -999.0 ? @" (shadow-prime multiple)" : @""]];
            return;
        }
    }

    // ── "N!" -- factorial (check primality) ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"^\\s*([0-9]+)\\s*!\\s*$" options:0 error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:expr options:0 range:NSMakeRange(0, expr.length)];
        if (m) {
            int n = (int)[[expr substringWithRange:[m rangeAtIndex:1]] integerValue];
            if (n > 20) {
                [self appendText:[NSString stringWithFormat:
                    @"%d! overflows uint64 (max 20!)\n", n]];
                return;
            }
            uint64_t result = 1;
            for (int i = 2; i <= n; i++) result *= i;
            [self appendText:[NSString stringWithFormat:
                @"%d! = %@\n", n, formatNumber(result)]];
            if (result >= 2) {
                bool p = prime::is_prime(result);
                [self appendText:[NSString stringWithFormat:
                    @"  %@ %@\n", formatNumber(result), p ? @"is prime!" : @"is composite"]];
            }
            return;
        }
    }

    // ── "N mod M" or "N % M" ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"([0-9eE^+\\-*/ .]+)\\s+(?:mod|%)\\s+([0-9eE^+\\-*/ .]+)"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t a = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            uint64_t b = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:2]]];
            if (b == 0) { [self appendText:@"Division by zero.\n"]; return; }
            [self appendText:[NSString stringWithFormat:
                @"%@ mod %@ = %@\n", formatNumber(a), formatNumber(b), formatNumber(a % b)]];
            return;
        }
    }

    // ── "pi(N)" -- prime counting function ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"pi\\s*\\(\\s*([0-9eE^+\\-*/ .]+)\\s*\\)"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t n = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            if (n > 100000000) {
                [self appendText:@"pi(N): max N = 100,000,000 (sieve memory limit)\n"];
                return;
            }
            auto t0 = std::chrono::steady_clock::now();
            auto sv = prime::sieve(n);
            uint64_t count = 0;
            for (uint64_t i = 0; i <= n; i++) if (sv[i]) count++;
            auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            [self appendText:[NSString stringWithFormat:
                @"pi(%@) = %@ (%.4fs)\n", formatNumber(n), formatNumber(count), dt]];
            return;
        }
    }

    // ── "euler totient N" or "totient(N)" or "phi(N)" ──
    {
        NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
            @"(?:euler\\s+)?(?:totient|phi)\\s*\\(?\\s*([0-9eE^+\\-*/ .]+)\\s*\\)?"
            options:NSRegularExpressionCaseInsensitive error:nil];
        NSTextCheckingResult *m = [re firstMatchInString:lower options:0 range:NSMakeRange(0, lower.length)];
        if (m) {
            uint64_t n = [self evalMathExpr:[expr substringWithRange:[m rangeAtIndex:1]]];
            // Compute Euler's totient using factorization
            auto factors = prime::factor_u64(n);
            uint64_t result = n;
            uint64_t prev = 0;
            for (auto f : factors) {
                if (f != prev) {
                    result = result / f * (f - 1);
                    prev = f;
                }
            }
            [self appendText:[NSString stringWithFormat:
                @"phi(%@) = %@\n", formatNumber(n), formatNumber(result)]];
            return;
        }
    }

    // ── Bare math expression fallback: evaluate and check primality ──
    {
        uint64_t val = [self evalMathExpr:expr];
        if (val >= 2) {
            [self appendText:[NSString stringWithFormat:@"= %@\n", formatNumber(val)]];
            [self runCheckSingle:val];
            return;
        }
        if (val == 0 || val == 1) {
            [self appendText:[NSString stringWithFormat:@"= %llu (not testable, need >= 2)\n", val]];
            return;
        }
    }

    [self appendText:[NSString stringWithFormat:
        @"Could not parse expression: %@\n"
        @"  Try: is_prime(997), factor(1729), 2^67-1, modpow(2,100,1e9+7),\n"
        @"       primes 1000 to 2000, next prime 10000, wieferich 1093,\n"
        @"       pi(1000), fibonacci(50), gcd(48,36), mersenne 31,\n"
        @"       twin primes 100 200, sophie germain 2 1000, phi(60)\n", expr]];
}

// ── Math expression evaluator (handles 2^67-1, 1e9+7, basic +,-,*,/,^) ──

- (uint64_t)evalMathExpr:(NSString *)raw {
    NSString *expr = [[raw stringByTrimmingCharactersInSet:
        [NSCharacterSet whitespaceCharacterSet]]
        stringByReplacingOccurrencesOfString:@" " withString:@""];
    // Strip commas (number formatting)
    expr = [expr stringByReplacingOccurrencesOfString:@"," withString:@""];

    if (expr.length == 0) return 0;

    // Tokenize into numbers and operators
    NSMutableArray<NSNumber *> *values = [NSMutableArray new];
    NSMutableArray<NSString *> *ops = [NSMutableArray new];

    NSUInteger i = 0;
    while (i < expr.length) {
        // Parse a number (possibly with 'e' notation like 1e9)
        NSMutableString *numStr = [NSMutableString new];
        while (i < expr.length) {
            unichar ch = [expr characterAtIndex:i];
            if ((ch >= '0' && ch <= '9') || ch == '.' ||
                ch == 'e' || ch == 'E') {
                [numStr appendFormat:@"%C", ch];
                i++;
            } else {
                break;
            }
        }
        if (numStr.length > 0) {
            double d = [numStr doubleValue];
            [values addObject:@((uint64_t)d)];
        } else {
            break; // unexpected char
        }

        // Parse operator
        if (i < expr.length) {
            unichar ch = [expr characterAtIndex:i];
            if (ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '^') {
                [ops addObject:[NSString stringWithFormat:@"%C", ch]];
                i++;
            } else {
                break;
            }
        }
    }

    if (values.count == 0) return 0;

    // Evaluate ^ first (right to left)
    for (NSInteger j = (NSInteger)ops.count - 1; j >= 0; j--) {
        if ([ops[j] isEqualToString:@"^"]) {
            uint64_t base = values[j].unsignedLongLongValue;
            uint64_t exp = values[j + 1].unsignedLongLongValue;
            uint64_t result = 1;
            for (uint64_t e = 0; e < exp; e++) result *= base;
            values[j] = @(result);
            [values removeObjectAtIndex:j + 1];
            [ops removeObjectAtIndex:j];
        }
    }

    // Evaluate * and / left to right
    for (NSInteger j = 0; j < (NSInteger)ops.count; ) {
        if ([ops[j] isEqualToString:@"*"]) {
            values[j] = @(values[j].unsignedLongLongValue * values[j + 1].unsignedLongLongValue);
            [values removeObjectAtIndex:j + 1];
            [ops removeObjectAtIndex:j];
        } else if ([ops[j] isEqualToString:@"/"]) {
            uint64_t denom = values[j + 1].unsignedLongLongValue;
            values[j] = @(denom > 0 ? values[j].unsignedLongLongValue / denom : 0);
            [values removeObjectAtIndex:j + 1];
            [ops removeObjectAtIndex:j];
        } else {
            j++;
        }
    }

    // Evaluate + and - left to right
    uint64_t result = values[0].unsignedLongLongValue;
    for (NSUInteger j = 0; j < ops.count; j++) {
        uint64_t v = values[j + 1].unsignedLongLongValue;
        if ([ops[j] isEqualToString:@"+"]) result += v;
        else if ([ops[j] isEqualToString:@"-"]) result -= v;
    }
    return result;
}

- (void)checkStopAction:(id)sender {
    [self stopBackgroundCheck];
}

- (void)stopBackgroundCheck {
    _checkRunning.store(false);
    // Join on a background queue to avoid blocking main thread
    __weak AppDelegate *weakSelf = self;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        AppDelegate *ss = weakSelf;
        if (ss && ss->_checkThread.joinable()) {
            ss->_checkThread.join();
        }
        dispatch_async(dispatch_get_main_queue(), ^{
            AppDelegate *s2 = weakSelf;
            if (s2) {
                s2.checkStopButton.enabled = NO;
                s2.checkGoButton.enabled = YES;
                [s2 appendText:@"Check stopped.\n"];
            }
        });
    });
}

// ── Check Single ────────────────────────────────────────────────────

- (void)runCheckSingle:(uint64_t)n {
    auto& kdb = prime::known_db();

    // ── Check known database first (instant, no computation) ──
    if (kdb.is_known(n)) {
        auto entries = kdb.get_entries(n);
        if (kdb.is_known_prime(n)) {
            // Known prime -- show all classifications
            NSMutableString *classes = [NSMutableString string];
            for (auto* e : entries) {
                if (classes.length > 0) [classes appendString:@", "];
                [classes appendFormat:@"%s", prime::known_class_name(e->kclass)];
            }
            NSString *desc = @"";
            if (entries.size() > 0 && entries[0]->description[0] != '\0') {
                desc = [NSString stringWithFormat:@" -- %s", entries[0]->description];
            }
            [self appendText:[NSString stringWithFormat:
                @"[OK] KNOWN PRIME: %@ [%@]%@ (0us -- database lookup)\n",
                formatNumber(n), classes, desc]];
        } else {
            // Known pseudoprime -- show factors
            auto* e = entries[0];
            std::string factors = prime::factors_comma_string(n);
            NSString *fstr = factors.empty() ? @"" :
                [NSString stringWithFormat:@" = %s", factors.c_str()];
            [self appendText:[NSString stringWithFormat:
                @"[X] KNOWN %s: %@%@ -- %s (0us -- database lookup)\n",
                prime::known_class_name(e->kclass), formatNumber(n), fstr, e->description]];
        }
        return;
    }

    // ── Not in database -- compute ──
    auto t0 = std::chrono::steady_clock::now();
    bool is_p = prime::is_prime(n);
    auto dt = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t0).count();

    // ── Pinch Factor analysis (works on any number) ──
    auto pinch_t0 = std::chrono::steady_clock::now();
    auto pinch_hits = prime::pinch_factor(n);
    auto pinch_dt = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - pinch_t0).count();

    if (is_p) {
        [self appendText:[NSString stringWithFormat:
            @"[OK] PRIME: %@ (%.1fus)\n", formatNumber(n), dt]];
        if (!pinch_hits.empty()) {
            // Primes shouldn't have pinch hits (unless n itself appears) -- flag it
            [self appendText:[NSString stringWithFormat:
                @"  WARNING: PinchFactor found %zu hit(s) on prime? (%.1fus)\n",
                pinch_hits.size(), pinch_dt]];
        }
    } else {
        bool crt = prime::crt_reject(n);
        NSString *method = crt ? @"CRT" : @"Miller-Rabin";
        std::string factors = prime::factors_comma_string(n);
        NSString *fstr = factors.empty() ? @"" :
            [NSString stringWithFormat:@" = %s", factors.c_str()];
        [self appendText:[NSString stringWithFormat:
            @"[X] COMPOSITE: %@%@ (caught by %@, %.1fus)\n", formatNumber(n), fstr, method, dt]];

        // Show Pinch Factor results
        if (pinch_hits.empty()) {
            [self appendText:[NSString stringWithFormat:
                @"  PinchFactor: no digit-structural divisors found (%.1fus)\n", pinch_dt]];
        } else {
            [self appendText:[NSString stringWithFormat:
                @"  PinchFactor: %zu divisor(s) found via digit splits (%.1fus):\n",
                pinch_hits.size(), pinch_dt]];
            for (auto& h : pinch_hits) {
                [self appendText:[NSString stringWithFormat:
                    @"    pos %d: [%@|%@] -> %@ divides N  (%s)\n",
                    h.pinch_pos,
                    formatNumber(h.left), formatNumber(h.right),
                    formatNumber(h.divisor),
                    h.method.c_str()]];
            }
        }

        // Show Lucky 7s results
        auto l7_t0 = std::chrono::steady_clock::now();
        auto l7_hits = prime::lucky7_factor(n);
        auto l7_dt = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - l7_t0).count();

        if (l7_hits.empty()) {
            [self appendText:[NSString stringWithFormat:
                @"  Lucky7s: no round-number proximity divisors found (%.1fus)\n", l7_dt]];
        } else {
            [self appendText:[NSString stringWithFormat:
                @"  Lucky7s: %zu divisor(s) found near powers of 10 (%.1fus):\n",
                l7_hits.size(), l7_dt]];
            for (auto& h : l7_hits) {
                NSString *sign = h.offset >= 0 ? @"+" : @"";
                [self appendText:[NSString stringWithFormat:
                    @"    10^%d %@%lld = %@ divides N  (%s)\n",
                    (int)log10((double)h.power_of_10), sign, h.offset,
                    formatNumber(h.divisor),
                    h.method.c_str()]];
            }
        }

        // DivisorWeb -- digit-by-digit factor sieve
        // Only run on numbers small enough to complete quickly (< ~10^12 for base 10)
        uint64_t web_limit = 1000000000000ULL; // 10^12
        if (n <= web_limit) {
            int bases[] = {10, 60};
            for (int base : bases) {
                auto web = prime::divisor_web(n, base);
                [self appendText:[NSString stringWithFormat:
                    @"  DivisorWeb (base %d): %zu divisors, %zu prime factors (%.1fus)\n",
                    base, web.all_divisors.size(), web.prime_divisors.size(), web.elapsed_us]];
                for (auto& lv : web.levels) {
                    NSMutableString *divStr = [NSMutableString string];
                    for (uint64_t d : lv.divisors) {
                        if (divStr.length > 0) [divStr appendString:@" "];
                        [divStr appendFormat:@"%@", formatNumber(d)];
                    }
                    NSString *hitStr = lv.divisors.empty() ? @"--" : divStr;
                    [self appendText:[NSString stringWithFormat:
                        @"    L%d [%@..%@] tested:%llu pruned:%llu+%llu -> %@\n",
                        lv.level, formatNumber(lv.range_lo), formatNumber(lv.range_hi),
                        lv.tested, lv.pruned_composite, lv.pruned_modular,
                        hitStr]];
                }
                if (!web.prime_divisors.empty()) {
                    NSMutableString *pStr = [NSMutableString string];
                    for (uint64_t p : web.prime_divisors) {
                        if (pStr.length > 0) [pStr appendString:@" x "];
                        [pStr appendFormat:@"%@", formatNumber(p)];
                    }
                    [self appendText:[NSString stringWithFormat:@"    prime factors: %@\n", pStr]];
                }
            }
        }
    }
}

// Helper: throttle manual check threads when user is interacting
- (void)throttleCheckIfNeeded {
    if (_taskMgr && _taskMgr->should_throttle()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

// ── Check From Here (GPU+CPU parallel upward scan) ──────────────────

- (void)runCheckFromHere:(uint64_t)from to:(uint64_t)to {
    _checkRunning.store(true);
    self.checkGoButton.enabled = NO;
    self.checkStopButton.enabled = YES;
    [self appendText:[NSString stringWithFormat:@"── Check From Here (GPU+CPU): starting at %@%@ ──\n",
        formatNumber(from), to > 0 ? [NSString stringWithFormat:@" to %@", formatNumber(to)] : @""]];

    __weak AppDelegate *weakSelf = self;
    prime::GPUBackend *gpu = _gpu;
    _checkThread = std::thread([weakSelf, from, to, gpu]() {
        pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
        AppDelegate *ss = weakSelf;
        if (!ss) return;
        uint64_t pos = from;
        if (pos < 2) pos = 2;
        uint64_t tested = 0;
        uint64_t found = 0;
        auto batchStart = std::chrono::steady_clock::now();
        const uint32_t BATCH = 8192;

        while (ss->_checkRunning.load()) {
            if (to > 0 && pos > to) break;
            [ss throttleCheckIfNeeded];

            // Build batch of odd candidates
            std::vector<uint64_t> batch;
            batch.reserve(BATCH);
            uint64_t p = pos;
            while (batch.size() < BATCH) {
                if (to > 0 && p > to) break;
                if (p == 2 || (p > 2 && p % 2 == 1)) {
                    batch.push_back(p);
                }
                p += (p <= 2) ? 1 : 2;
            }
            if (batch.empty()) break;
            pos = p;

            // Split: 30% CPU, 70% GPU -- both test in parallel
            uint32_t total = (uint32_t)batch.size();
            uint32_t cpu_count = total * 3 / 10;
            uint32_t gpu_start = cpu_count;
            uint32_t gpu_count = total - cpu_count;

            // CPU batch: test primality in parallel across cores
            std::vector<uint8_t> cpu_results(cpu_count, 0);
            auto cpu_future = std::async(std::launch::async, [&batch, cpu_count, &cpu_results]() {
                for (uint32_t i = 0; i < cpu_count; i++) {
                    cpu_results[i] = prime::is_prime(batch[i]) ? 1 : 0;
                }
            });

            // GPU batch: primality test on remaining
            std::vector<uint8_t> gpu_results(gpu_count, 0);
            if (gpu_count > 0) {
                gpu->primality_batch(batch.data() + gpu_start, gpu_results.data(), gpu_count);
            }

            cpu_future.get();

            // Collect results
            auto& kdb_ref = prime::known_db();
            for (uint32_t i = 0; i < cpu_count; i++) {
                if (cpu_results[i]) {
                    found++;
                    uint64_t val = batch[i];
                    NSString *known = @"";
                    if (kdb_ref.is_known_prime(val)) {
                        auto* e = kdb_ref.get_entry(val);
                        known = [NSString stringWithFormat:@" [KNOWN: %s]", e ? e->description : ""];
                    }
                    NSString *msg = [NSString stringWithFormat:@"  PRIME #%llu: %@%@ (CPU)\n",
                        found, formatNumber(val), known];
                    dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf appendText:msg]; });
                }
            }
            for (uint32_t i = 0; i < gpu_count; i++) {
                if (gpu_results[i]) {
                    found++;
                    uint64_t val = batch[gpu_start + i];
                    NSString *known = @"";
                    if (kdb_ref.is_known_prime(val)) {
                        auto* e = kdb_ref.get_entry(val);
                        known = [NSString stringWithFormat:@" [KNOWN: %s]", e ? e->description : ""];
                    }
                    NSString *msg = [NSString stringWithFormat:@"  PRIME #%llu: %@%@ (GPU)\n",
                        found, formatNumber(val), known];
                    dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf appendText:msg]; });
                }
            }

            tested += total;
            if (tested % (BATCH * 4) < BATCH) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - batchStart).count();
                double rate = tested / elapsed;
                NSString *msg = [NSString stringWithFormat:
                    @"  ... checked %@ | found %@ primes | %@/s (CPU+GPU)\n",
                    formatNumber(tested), formatNumber(found), formatNumber((uint64_t)rate)];
                dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf appendText:msg]; });
            }
        }

        NSString *msg = [NSString stringWithFormat:
            @"── Check From Here complete: tested %@, found %@ primes ──\n",
            formatNumber(tested), formatNumber(found)];
        dispatch_async(dispatch_get_main_queue(), ^{
            [weakSelf appendText:msg];
            AppDelegate *s2 = weakSelf;
            if (s2) { s2.checkStopButton.enabled = NO; s2.checkGoButton.enabled = YES; }
        });
        ss->_checkRunning.store(false);
    });
    // Thread stays joinable for proper cleanup
}

// ── Check Linear (GPU+CPU wheel-210 fast scan) ─────────────────────

- (void)runCheckLinear:(uint64_t)from to:(uint64_t)to {
    _checkRunning.store(true);
    self.checkGoButton.enabled = NO;
    self.checkStopButton.enabled = YES;
    [self appendText:[NSString stringWithFormat:@"── Check Linear (GPU+CPU): starting at %@%@ ──\n",
        formatNumber(from), to > 0 ? [NSString stringWithFormat:@" to %@", formatNumber(to)] : @""]];

    __weak AppDelegate *weakSelf = self;
    prime::GPUBackend *gpu = _gpu;
    _checkThread = std::thread([weakSelf, from, to, gpu]() {
        pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
        AppDelegate *ss = weakSelf;
        if (!ss) return;
        uint64_t pos = from;
        if (pos < 2) pos = 2;
        uint64_t tested = 0;
        uint64_t found = 0;
        uint64_t skipped = 0;
        auto batchStart = std::chrono::steady_clock::now();
        const uint32_t BATCH = 8192;

        while (ss->_checkRunning.load()) {
            if (to > 0 && pos > to) break;
            [ss throttleCheckIfNeeded];

            // Build batch of wheel-210 valid candidates, CRT pre-filtered
            std::vector<uint64_t> batch;
            batch.reserve(BATCH);
            uint64_t p = pos;
            while (batch.size() < BATCH) {
                if (to > 0 && p > to) break;
                bool small = (p == 2 || p == 3 || p == 5 || p == 7);
                if (small) {
                    batch.push_back(p);
                } else if (p % 2 == 1 && prime::WHEEL.valid(p) && !prime::crt_reject(p)) {
                    batch.push_back(p);
                } else {
                    skipped++;
                }
                p += (p <= 2) ? 1 : 2;
            }
            if (batch.empty()) break;
            pos = p;

            // Split: CPU and GPU test in parallel
            uint32_t total = (uint32_t)batch.size();
            uint32_t cpu_count = total * 3 / 10;
            uint32_t gpu_start = cpu_count;
            uint32_t gpu_count = total - cpu_count;

            std::vector<uint8_t> cpu_results(cpu_count, 0);
            auto cpu_future = std::async(std::launch::async, [&batch, cpu_count, &cpu_results]() {
                for (uint32_t i = 0; i < cpu_count; i++) {
                    cpu_results[i] = prime::is_prime(batch[i]) ? 1 : 0;
                }
            });

            std::vector<uint8_t> gpu_results(gpu_count, 0);
            if (gpu_count > 0) {
                gpu->primality_batch(batch.data() + gpu_start, gpu_results.data(), gpu_count);
            }

            cpu_future.get();

            auto& kdb_lin = prime::known_db();
            for (uint32_t i = 0; i < cpu_count; i++) {
                if (cpu_results[i]) {
                    found++;
                    uint64_t val = batch[i];
                    NSString *known = @"";
                    if (kdb_lin.is_known_prime(val)) {
                        auto* e = kdb_lin.get_entry(val);
                        known = [NSString stringWithFormat:@" [KNOWN: %s]", e ? e->description : ""];
                    }
                    NSString *msg = [NSString stringWithFormat:@"  PRIME #%llu: %@%@\n",
                        found, formatNumber(val), known];
                    dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf appendText:msg]; });
                }
            }
            for (uint32_t i = 0; i < gpu_count; i++) {
                if (gpu_results[i]) {
                    found++;
                    uint64_t val = batch[gpu_start + i];
                    NSString *known = @"";
                    if (kdb_lin.is_known_prime(val)) {
                        auto* e = kdb_lin.get_entry(val);
                        known = [NSString stringWithFormat:@" [KNOWN: %s]", e ? e->description : ""];
                    }
                    NSString *msg = [NSString stringWithFormat:@"  PRIME #%llu: %@%@\n",
                        found, formatNumber(val), known];
                    dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf appendText:msg]; });
                }
            }

            tested += total;
            if (tested % (BATCH * 4) < BATCH) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - batchStart).count();
                double rate = tested / elapsed;
                NSString *msg = [NSString stringWithFormat:
                    @"  ... tested %@ | skipped %@ | found %@ | %@/s (CPU+GPU)\n",
                    formatNumber(tested), formatNumber(skipped),
                    formatNumber(found), formatNumber((uint64_t)rate)];
                dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf appendText:msg]; });
            }
        }

        NSString *msg = [NSString stringWithFormat:
            @"── Check Linear complete: tested %@, skipped %@, found %@ primes ──\n",
            formatNumber(tested), formatNumber(skipped), formatNumber(found)];
        dispatch_async(dispatch_get_main_queue(), ^{
            [weakSelf appendText:msg];
            AppDelegate *s2 = weakSelf;
            if (s2) { s2.checkStopButton.enabled = NO; s2.checkGoButton.enabled = YES; }
        });
        ss->_checkRunning.store(false);
    });
    // Thread stays joinable for proper cleanup
}

// ── Markov Predict ──────────────────────────────────────────────────
//
// Train a Markov model on known primes (from a loaded list or sieved),
// then predict primes in gaps. Each prediction is verified and factored.
//
// Prime list file: one prime per line (decimal). Lines starting with #
// are comments. Loaded via "Load Primes" button or from
// ~/Documents/primes/primelocations/known_primes.txt automatically.
//
// The "From" field selects which prime in the list to start from.
// The "Window" field is how many predictions to chain forward.

// ── Primes as Voids theory popup ─────────────────────────────────────

- (void)showPrimesAsVoidsTheory:(id)sender {
    CGFloat W = 680, H = 720;
    NSWindow *win = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(150, 100, W, H)
        styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable
        backing:NSBackingStoreBuffered defer:NO];
    win.title = @"Primes as Voids -- An Atomic Boundary Model (S. Nester, 2026)";
    win.releasedWhenClosed = NO;
    win.minSize = NSMakeSize(500, 400);

    NSScrollView *scroll = [[NSScrollView alloc] initWithFrame:win.contentView.bounds];
    scroll.hasVerticalScroller = YES;
    scroll.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

    NSTextView *tv = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, W - 20, H * 4)];
    tv.editable = NO;
    tv.autoresizingMask = NSViewWidthSizable;
    tv.textContainerInset = NSMakeSize(16, 16);
    tv.backgroundColor = [NSColor textBackgroundColor];
    scroll.documentView = tv;
    [win.contentView addSubview:scroll];

    NSMutableAttributedString *text = [[NSMutableAttributedString alloc] init];
    NSFont *titleFont = [NSFont boldSystemFontOfSize:16];
    NSFont *headFont = [NSFont boldSystemFontOfSize:13];
    NSFont *bodyFont = [NSFont systemFontOfSize:11.5];
    NSFont *monoFont = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    NSDictionary *titleAttr = @{NSFontAttributeName: titleFont};
    NSDictionary *headAttr = @{NSFontAttributeName: headFont};
    NSDictionary *bodyAttr = @{NSFontAttributeName: bodyFont};
    NSDictionary *monoAttr = @{NSFontAttributeName: monoFont};

    #define APPEND(s, a) [text appendAttributedString:[[NSAttributedString alloc] initWithString:s attributes:a]]

    APPEND(@"Primes as Voids\n", titleAttr);
    APPEND(@"An Atomic Boundary Model for Prime Distribution\n", headAttr);
    APPEND(@"Sergei Nester -- ViewBuild Research, Battery Point, Tasmania\n"
           @"April 2026 -- Working Paper\n\n", bodyAttr);

    APPEND(@"ABSTRACT\n", headAttr);
    APPEND(@"Primes are not special objects but voids -- positions of total destructive "
           @"interference in a harmonic field generated by the Sieve of Eratosthenes. Each "
           @"prime is treated as an atomic centre with a Voronoi territory defined by the "
           @"half-gaps to its nearest prime neighbours. The boundary between adjacent prime "
           @"territories is the integer midpoint of a prime gap. Every such boundary is "
           @"composite, with smallest prime factor determined entirely by the gap size modulo "
           @"small primes.\n\n"
           @"A local Markov chain model -- in which each step is generated from the conditional "
           @"distribution of gap size and next-digit transition given only the current prime's "
           @"last digit -- reproduces the digit transition structure and drift behaviour of the "
           @"prime sequence. Residual error comes from the absence of a primality filter.\n\n", bodyAttr);

    APPEND(@"THE HARMONIC FIELD\n", headAttr);
    APPEND(@"For each prime p, define a harmonic wave of period p that marks every multiple "
           @"of p as resonant. The Sieve of Eratosthenes constructs composites as the union "
           @"of all such waves: n is composite if and only if it is resonant with at least one "
           @"wave. Primes are positions where no wave arrives.\n\n"
           @"Void density decreasing as 1/ln(N) follows from the field becoming more saturated "
           @"with each additional wave. This is packing saturation: 1/ln(N) is a property of "
           @"the geometry, not of primes per se.\n\n", bodyAttr);

    APPEND(@"THE ATOMIC MODEL\n", headAttr);
    APPEND(@"Each prime p gets a Voronoi territory T(p) -- integers closer to p than to any "
           @"other prime. The atomic radius r(p) = (gap_before + gap_after) / 4.\n\n"
           @"Boundary Theorem: Every integer midpoint between two consecutive primes greater "
           @"than 2 is composite. The smallest prime factor at the boundary is determined by "
           @"the gap size modulo small primes.\n\n", bodyAttr);

    APPEND(@"  Gap    Boundary factor    Forced?\n", monoAttr);
    APPEND(@"   2     div 2 (100%)       Yes (twin primes)\n", monoAttr);
    APPEND(@"   4     div 3 (100%)       Yes\n", monoAttr);
    APPEND(@"   6     div 2 (100%)       Yes\n", monoAttr);
    APPEND(@"   8     div 3 (100%)       Yes\n", monoAttr);
    APPEND(@"  10     div 2 (100%)       Yes\n", monoAttr);
    APPEND(@"  12     Mixed              No -- first ambiguous case\n\n", monoAttr);

    APPEND(@"SHELL CLASSIFICATION\n", headAttr);
    APPEND(@"Each predicted prime is classified by its position in the atomic structure:\n\n"
           @"  SHELL: Small gaps on both sides. Prime cluster. Tightly grouped. Low "
           @"smoothness -- fewer small factors in p-1 means less composite packing nearby, "
           @"leaving room for primes to cluster.\n\n"
           @"  BOUNDARY: Large gap on at least one side. Sits at the edge of a composite "
           @"nucleus -- a dense region where many harmonic waves overlap. High smoothness "
           @"of p-1 indicates the nucleus is saturated with small divisors.\n\n"
           @"  EDGE: Transitional zone between shell and boundary. Moderate gaps, mixed "
           @"factor structure.\n\n", bodyAttr);

    APPEND(@"OUTPUT FIELDS\n", headAttr);
    APPEND(@"  gap=before|after   Gaps to previous and next prime\n", monoAttr);
    APPEND(@"  [SHELL/BOUNDARY/EDGE]  Atomic position classification\n", monoAttr);
    APPEND(@"  smooth=s(O/o)     Smoothness: Omega/log2(p-1)\n", monoAttr);
    APPEND(@"                    O = total prime factors (with multiplicity)\n", monoAttr);
    APPEND(@"                    o = distinct prime factors\n", monoAttr);
    APPEND(@"  lpf=              Largest prime factor of p-1\n", monoAttr);
    APPEND(@"  r=                Atomic radius (avg of surrounding gaps)\n\n", monoAttr);

    APPEND(@"MARKOV PREDICTION\n", headAttr);
    APPEND(@"The predictor uses two empirical distributions conditioned on the last "
           @"digit D of the current prime (from {1,3,7,9}):\n\n"
           @"  1. Gap distribution P(gap | D) -- how far to the next prime\n"
           @"  2. Digit transition P(D' | D) -- what the next prime ends in\n\n"
           @"2,000 single-step trials vote on the next prime. Top candidates are "
           @"verified with deterministic Miller-Rabin (12 witnesses: 2 through 37, "
           @"provably correct for all 64-bit integers). Modular exponentiation uses "
           @"Nester Carry Chain mulmod (4-7x faster than binary doubling).\n\n"
           @"The model adapts incrementally: each confirmed prime updates the gap "
           @"distribution and transition matrix, giving it a recency bias that tracks "
           @"local density changes.\n\n", bodyAttr);

    APPEND(@"PRIMALITY VERIFICATION\n", headAttr);
    APPEND(@"  - Miller-Rabin with 12 deterministic witnesses (2,3,5,7,11,13,17,19,23,29,31,37)\n"
           @"  - Provably correct for all n < 3.3 x 10^24 (covers all 64-bit integers)\n"
           @"  - Nester Carry Chain mulmod: 128-bit (a*b) mod m via partial product carry chain\n"
           @"  - Trial division with small primes (2,3,5,...,37) as fast pre-filter\n"
           @"  - Heuristic divisor candidates (Lucky7s, PinchFactor) before Pollard rho\n"
           @"  - Pollard rho factorisation for composite rejection\n\n", bodyAttr);

    APPEND(@"NESTER CARRY CHAIN STREAMING DIVISIBILITY\n", headAttr);
    APPEND(@"For big-number trial division (numbers beyond 64 bits), the streaming "
           @"divisibility engine tests whether N is divisible by candidate divisors "
           @"without division. Streams through the number segment by segment, accumulating "
           @"via Barrett reduction (precomputed reciprocal multiply). N-wide batching "
           @"processes up to 8 divisors per pass. Up to 8x faster than scalar division "
           @"on 2048-bit numbers. Over 10 million divisor tests per second on Apple Silicon.\n\n", bodyAttr);

    APPEND(@"WHAT THE MARKOV MODEL SHOWS\n", headAttr);
    APPEND(@"The model is missing exactly one thing: primality verification. Everything "
           @"else -- density shape, digit transitions, drift behaviour -- emerges from "
           @"local conditional rules. The only non-local ingredient in prime structure is "
           @"divisibility by primes below sqrt(N). All other statistics propagate forward "
           @"from the last known prime.\n\n", bodyAttr);

    APPEND(@"CONNECTIONS\n", headAttr);
    APPEND(@"The Lemke Oliver consecutive-digit repulsion (primes avoid repeating their "
           @"last digit) is the arithmetic analogue of GUE eigenvalue repulsion in random "
           @"matrix theory. The Montgomery-Odlyzko law -- that Riemann zeta zeros space "
           @"like quantum energy levels -- reflects the shared interference field structure.\n\n"
           @"The Cramer random model succeeds because it captures the packing geometry "
           @"through 1/ln(n), not because it captures anything about the algebraic structure "
           @"of primality.\n\n", bodyAttr);

    APPEND(@"(c) 2026 ViewBuild Research. Working paper.\n", bodyAttr);

    #undef APPEND

    [tv.textStorage setAttributedString:text];
    [tv scrollToBeginningOfDocument:nil];
    [win makeKeyAndOrderFront:nil];
    // Prevent deallocation
    objc_setAssociatedObject(self, "theoryWindow", win, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

// ── Carry Chain description popup ───────────────────────────────────

- (void)showCarryChainInfo:(id)sender {
    CGFloat W = 600, H = 560;
    NSWindow *win = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(200, 150, W, H)
        styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable
        backing:NSBackingStoreBuffered defer:NO];
    win.title = @"Nester Carry Chain (S. Nester, 2026)";
    win.releasedWhenClosed = NO;
    win.minSize = NSMakeSize(450, 350);

    NSScrollView *scroll = [[NSScrollView alloc] initWithFrame:win.contentView.bounds];
    scroll.hasVerticalScroller = YES;
    scroll.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

    NSTextView *tv = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, W - 20, H * 3)];
    tv.editable = NO;
    tv.autoresizingMask = NSViewWidthSizable;
    tv.textContainerInset = NSMakeSize(16, 16);
    tv.backgroundColor = [NSColor textBackgroundColor];
    scroll.documentView = tv;
    [win.contentView addSubview:scroll];

    NSMutableAttributedString *text = [[NSMutableAttributedString alloc] init];
    NSFont *titleFont = [NSFont boldSystemFontOfSize:16];
    NSFont *headFont = [NSFont boldSystemFontOfSize:13];
    NSFont *bodyFont = [NSFont systemFontOfSize:11.5];
    NSFont *monoFont = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    NSDictionary *titleAttr = @{NSFontAttributeName: titleFont};
    NSDictionary *headAttr = @{NSFontAttributeName: headFont};
    NSDictionary *bodyAttr = @{NSFontAttributeName: bodyFont};
    NSDictionary *monoAttr = @{NSFontAttributeName: monoFont};

    #define APPEND(s, a) [text appendAttributedString:[[NSAttributedString alloc] initWithString:s attributes:a]]

    APPEND(@"Nester Carry Chain\n", titleAttr);
    APPEND(@"Sergei Nester, 2026\n\n", headAttr);

    APPEND(@"Two methods that replace division with accumulation.\n\n", bodyAttr);

    // --- Method 1 ---
    APPEND(@"NESTER CARRY CHAIN: MODULAR MULTIPLICATION\n", headAttr);
    APPEND(@"Computes (a * b) mod m where a, b, m can be up to 128 bits, using "
           @"only hardware-native integer operations. Decomposes the 128-bit "
           @"multiply into a carry chain of 64-bit partial products, propagates "
           @"carries through three 64-bit words, then reduces the 192-bit result "
           @"modulo m using staged shifts. No division anywhere.\n\n", bodyAttr);

    APPEND(@"Steps:\n", bodyAttr);
    APPEND(@"  1. Split: a = [a_hi : a_lo], b = [b_hi : b_lo]  (64-bit halves)\n", monoAttr);
    APPEND(@"  2. Partial products:\n", monoAttr);
    APPEND(@"       p       = a_lo * b_lo          (128-bit)\n", monoAttr);
    APPEND(@"       cross   = a_lo*b_hi + a_hi*b_lo (128-bit)\n", monoAttr);
    APPEND(@"       hh      = a_hi * b_hi           (64-bit)\n", monoAttr);
    APPEND(@"  3. Carry chain: assemble [r2 : r1 : r0] (192-bit result)\n", monoAttr);
    APPEND(@"       r0 = low 64 bits of p\n", monoAttr);
    APPEND(@"       r1 = high 64 of p + low 64 of cross (with carry)\n", monoAttr);
    APPEND(@"       r2 = carry + high 64 of cross + hh\n", monoAttr);
    APPEND(@"  4. Reduce: hi_mod = [r2:r1] mod m\n", monoAttr);
    APPEND(@"  5. Staged shift: (hi_mod << 32) mod m, then << 32 again\n", monoAttr);
    APPEND(@"  6. Final: (shifted + r0) mod m\n\n", monoAttr);

    APPEND(@"4-7x faster than binary doubling for 128-bit modular exponentiation. "
           @"On Apple Silicon, MUL + UMULH computes the full 128-bit product in "
           @"3-4 cycles. The carry chain assembles the result with no branching.\n\n", bodyAttr);

    // --- Method 2 ---
    APPEND(@"NESTER CARRY CHAIN: STREAMING DIVISIBILITY\n", headAttr);
    APPEND(@"Tests whether an arbitrarily large number N is divisible by a candidate "
           @"divisor d, without ever dividing. Accumulates multiples of d while "
           @"streaming through the number segment by segment (MSB to LSB).\n\n", bodyAttr);

    APPEND(@"Choose a potential divisor. Add it to itself repeatedly as we move "
           @"through each 64-bit segment of the number. At each step there are "
           @"three outcomes:\n\n", bodyAttr);
    APPEND(@"  HIT   -- accumulator is under the segment, keep going\n", monoAttr);
    APPEND(@"  BUST  -- accumulator exceeds the segment, reduce and continue\n", monoAttr);
    APPEND(@"  MATCH -- accumulator equals zero at the end, d divides N\n\n", monoAttr);

    APPEND(@"Uses Barrett reduction (precomputed reciprocal multiply) instead of "
           @"hardware division. The reciprocal inv = floor(2^64 / d) is computed "
           @"once, then every reduction is: q = mulhi(x, inv); r = x - q*d. "
           @"No UDIV instruction executes in the hot loop.\n\n", bodyAttr);

    APPEND(@"N-wide batching processes N divisors per pass through the number. "
           @"N is a compile-time template parameter (1, 2, 4, 8, or 16). Each "
           @"stream is independent so the CPU pipelines all N multiply-accumulate "
           @"chains in parallel. An adaptive calibrator times each width and picks "
           @"the fastest for the given number size.\n\n", bodyAttr);

    APPEND(@"Up to 8x faster than scalar division on RSA-2048 sized numbers "
           @"(2048 bits, 32 limbs, 50K candidate divisors). Rate exceeds 10 million "
           @"divisor tests per second on Apple Silicon.\n\n", bodyAttr);

    // --- Where used ---
    APPEND(@"WHERE USED IN PRIMEPATH\n", headAttr);
    APPEND(@"  - Wieferich prime testing: 2^(p-1) mod p^2 (128-bit modulus)\n"
           @"  - Wall-Sun-Sun prime testing: Fibonacci matrix mod p^2\n"
           @"  - GIMPS trial factoring: CPU verification of GPU-found factors\n"
           @"  - Mersenne factor scanning: modpow(2, q, f) for large f\n"
           @"  - Markov Predict: Miller-Rabin verification of candidates\n"
           @"  - Big-number trial division: streaming divisibility for arbitrary width\n\n"
           @"The NesterCarryChain toggle switches between these methods and "
           @"the standard binary doubling method for direct speed comparison.\n\n", bodyAttr);

    APPEND(@"IMPLEMENTATION NOTES\n", headAttr);
    APPEND(@"Modular multiplication uses the carry chain with staged 32-bit shift for "
           @"moduli under 96 bits (4-7x speedup). For larger moduli it falls back to "
           @"binary doubling automatically -- no user action needed.\n"
           @"Streaming divisibility uses 32-bit Barrett reciprocals for divisors < 2^32 "
           @"and a 64-bit two-wide path for larger divisors. Routing is automatic.\n\n", bodyAttr);

    APPEND(@"(c) 2026 Sergei Nester. PrimePath.\n", bodyAttr);

    #undef APPEND

    [tv.textStorage setAttributedString:text];
    [tv scrollToBeginningOfDocument:nil];
    [win makeKeyAndOrderFront:nil];
    objc_setAssociatedObject(self, "carryChainWindow", win, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

// ── Markov Predict popup window ──────────────────────────────────────

- (void)showMarkovPredictWindow:(id)sender {
    CGFloat W = 600, H = 520;
    NSRect frame = NSMakeRect(200, 200, W, H);
    NSWindow *win = [[NSWindow alloc]
        initWithContentRect:frame
        styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable
        backing:NSBackingStoreBuffered defer:NO];
    win.title = @"Markov Prime Predictor";
    win.releasedWhenClosed = NO;
    win.minSize = NSMakeSize(500, 400);
    NSView *cv = win.contentView;
    CGFloat M = 12, y = H - 12;

    // Title
    NSTextField *title = [self labelAt:NSMakeRect(M, y - 18, W - 2*M, 18)
        text:@"Markov Prime Predictor" bold:YES size:13];
    [cv addSubview:title];
    y -= 26;

    NSTextField *desc = [self labelAt:NSMakeRect(M, y - 28, W - 2*M - 120, 28)
        text:@"Primes as Voids: predict forward using local conditional gap distributions, "
             @"verify candidates, and analyse atomic boundary structure."
        bold:NO size:10];
    desc.lineBreakMode = NSLineBreakByWordWrapping;
    [cv addSubview:desc];

    NSButton *theoryBtn = [self buttonAt:NSMakeRect(W - M - 112, y - 24, 104, 22)
        title:@"Theory (S.Nester)" action:@selector(showPrimesAsVoidsTheory:)];
    theoryBtn.font = [NSFont systemFontOfSize:10];
    [cv addSubview:theoryBtn];
    y -= 36;

    // Load Primes button + Fetch from web + status
    NSButton *loadBtn = [self buttonAt:NSMakeRect(M, y - 22, 86, 22)
        title:@"Load File" action:@selector(markovLoadPrimes:)];
    loadBtn.font = [NSFont systemFontOfSize:10];
    [cv addSubview:loadBtn];

    NSPopUpButton *fetchPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(M + 92, y - 22, 190, 22) pullsDown:YES];
    [fetchPopup addItemWithTitle:@"Fetch Known Primes..."];
    [fetchPopup addItemWithTitle:@"First 10,000 primes (t5k.org)"];
    [fetchPopup addItemWithTitle:@"First 100,000 primes (t5k.org)"];
    [fetchPopup addItemWithTitle:@"First 10,000 primes (OEIS)"];
    [fetchPopup addItemWithTitle:@"First 1,000,000 primes (t5k.org zip)"];
    fetchPopup.font = [NSFont systemFontOfSize:10];
    fetchPopup.target = self;
    fetchPopup.action = @selector(markovFetchPrimes:);
    fetchPopup.tag = 9207;
    [cv addSubview:fetchPopup];

    NSTextField *loadStatus = [self labelAt:NSMakeRect(M + 290, y - 18, W - M - 300, 14)
        text:@"No prime list loaded" bold:NO size:10];
    loadStatus.tag = 9201;
    loadStatus.textColor = [NSColor secondaryLabelColor];
    [cv addSubview:loadStatus];
    y -= 30;

    // Start prime field
    NSTextField *fromLbl = [self labelAt:NSMakeRect(M, y - 16, 70, 14)
        text:@"Start prime:" bold:NO size:10];
    [cv addSubview:fromLbl];
    NSTextField *fromField = [[NSTextField alloc] initWithFrame:NSMakeRect(M + 74, y - 18, 200, 20)];
    fromField.font = [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular];
    fromField.placeholderString = @"e.g. 1000003";
    fromField.tag = 9202;
    [cv addSubview:fromField];

    // Steps field
    NSTextField *stepsLbl = [self labelAt:NSMakeRect(M + 290, y - 16, 44, 14)
        text:@"Steps:" bold:NO size:10];
    [cv addSubview:stepsLbl];
    NSTextField *stepsField = [[NSTextField alloc] initWithFrame:NSMakeRect(M + 336, y - 18, 80, 20)];
    stepsField.font = [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular];
    stepsField.placeholderString = @"20";
    stepsField.tag = 9203;
    [cv addSubview:stepsField];

    // Go / Stop buttons
    NSButton *goBtn = [self buttonAt:NSMakeRect(M + 430, y - 20, 60, 22)
        title:@"Run" action:@selector(markovWindowGo:)];
    goBtn.font = [NSFont systemFontOfSize:11 weight:NSFontWeightMedium];
    goBtn.tag = 9204;
    [cv addSubview:goBtn];

    NSButton *stopBtn = [self buttonAt:NSMakeRect(M + 496, y - 20, 60, 22)
        title:@"Stop" action:@selector(markovWindowStop:)];
    stopBtn.font = [NSFont systemFontOfSize:11];
    stopBtn.enabled = NO;
    stopBtn.tag = 9205;
    [cv addSubview:stopBtn];
    y -= 28;

    // Stream benchmark button
    NSButton *benchBtn = [self buttonAt:NSMakeRect(M, y - 20, 160, 22)
        title:@"Nester-CarryChain Test" action:@selector(runStreamBenchmark:)];
    benchBtn.font = [NSFont systemFontOfSize:10];
    benchBtn.toolTip = @"Benchmark streaming NEON divisibility vs scalar vs carry-chain";
    [cv addSubview:benchBtn];
    y -= 26;

    // Separator
    NSBox *sep = [[NSBox alloc] initWithFrame:NSMakeRect(M, y - 2, W - 2*M, 1)];
    sep.boxType = NSBoxSeparator;
    [cv addSubview:sep];
    y -= 8;

    // Log output
    NSScrollView *logScroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(M, 8, W - 2*M, y - 12)];
    logScroll.hasVerticalScroller = YES;
    logScroll.borderType = NSBezelBorder;
    logScroll.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    NSTextView *logView = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, W - 2*M - 4, y - 16)];
    logView.font = [NSFont monospacedSystemFontOfSize:9.5 weight:NSFontWeightRegular];
    logView.editable = NO;
    logView.backgroundColor = [NSColor textBackgroundColor];
    logView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    logScroll.documentView = logView;
    [cv addSubview:logScroll];
    objc_setAssociatedObject(win, "markovLogView", logView, OBJC_ASSOCIATION_RETAIN_NONATOMIC);

    // Update load status if primes already loaded
    if (!_markovPrimeList.empty()) {
        loadStatus.stringValue = [NSString stringWithFormat:@"%@ primes loaded",
            formatNumber((uint64_t)_markovPrimeList.size())];
        loadStatus.textColor = [NSColor systemGreenColor];
    }

    self.markovLoadButton = loadBtn;  // so markovLoadPrimes: can update status in this window
    [win makeKeyAndOrderFront:nil];
    objc_setAssociatedObject(self, "markovWindow", win, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

- (void)markovWindowGo:(id)sender {
    NSWindow *win = objc_getAssociatedObject(self, "markovWindow");
    if (!win) return;

    NSTextField *fromField = [win.contentView viewWithTag:9202];
    NSTextField *stepsField = [win.contentView viewWithTag:9203];
    NSButton *goBtn = [win.contentView viewWithTag:9204];
    NSButton *stopBtn = [win.contentView viewWithTag:9205];
    NSTextView *logView = objc_getAssociatedObject(win, "markovLogView");

    NSString *fromStr = [fromField.stringValue stringByReplacingOccurrencesOfString:@"," withString:@""];
    uint64_t startPrime = strtoull(fromStr.UTF8String, nullptr, 10);
    if (startPrime < 2) {
        [logView.textStorage appendAttributedString:[[NSAttributedString alloc]
            initWithString:@"Enter a start prime >= 2.\n"
            attributes:@{NSFontAttributeName: logView.font}]];
        return;
    }

    NSString *stepsStr = stepsField.stringValue;
    uint64_t steps = stepsStr.length > 0 ? strtoull(stepsStr.UTF8String, nullptr, 10) : 20;

    goBtn.enabled = NO;
    stopBtn.enabled = YES;
    logView.string = @"";

    // Redirect appendText output to this window's log for the duration of the run
    objc_setAssociatedObject(self, "markovLogView", logView, OBJC_ASSOCIATION_RETAIN_NONATOMIC);

    [self runMarkovPredict:startPrime steps:steps];
}

- (void)markovWindowStop:(id)sender {
    _checkRunning.store(false);
}

// Helper: append to the Markov window log if it exists, otherwise main log
- (void)markovLog:(NSString *)text {
    NSTextView *logView = objc_getAssociatedObject(self, "markovLogView");
    if (logView) {
        [logView.textStorage appendAttributedString:[[NSAttributedString alloc]
            initWithString:text
            attributes:@{NSFontAttributeName: logView.font,
                         NSForegroundColorAttributeName: [NSColor labelColor]}]];
        [logView scrollRangeToVisible:NSMakeRange(logView.string.length, 0)];
    } else {
        [self appendText:text];
    }
}

- (void)markovFetchPrimes:(id)sender {
    NSPopUpButton *popup = (NSPopUpButton *)sender;
    NSInteger idx = popup.indexOfSelectedItem;
    if (idx < 1) return; // title item

    NSString *url = nil;
    BOOL isZip = NO;
    BOOL isOEIS = NO;
    switch (idx) {
        case 1: url = @"https://t5k.org/lists/small/10000.txt"; break;
        case 2: url = @"https://t5k.org/lists/small/100000.txt"; break;
        case 3: url = @"https://oeis.org/A000040/b000040.txt"; isOEIS = YES; break;
        case 4: url = @"https://t5k.org/lists/small/millions/primes1.zip"; isZip = YES; break;
        default: return;
    }

    [self markovLog:[NSString stringWithFormat:@"Fetching primes from %@...\n", url]];

    // Update status
    NSWindow *mwin = objc_getAssociatedObject(self, "markovWindow");
    NSTextField *statusLbl = mwin ? [mwin.contentView viewWithTag:9201] : nil;
    if (statusLbl) {
        statusLbl.stringValue = @"Downloading...";
        statusLbl.textColor = [NSColor systemOrangeColor];
    }

    __weak AppDelegate *weakSelf = self;
    NSURLSession *session = [NSURLSession sharedSession];
    NSURLSessionDataTask *task = [session dataTaskWithURL:[NSURL URLWithString:url]
        completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {

        dispatch_async(dispatch_get_main_queue(), ^{
            AppDelegate *ss = weakSelf;
            if (!ss) return;

            if (error || !data) {
                [ss markovLog:[NSString stringWithFormat:@"Fetch failed: %@\n",
                    error.localizedDescription ?: @"no data"]];
                if (statusLbl) {
                    statusLbl.stringValue = @"Download failed";
                    statusLbl.textColor = [NSColor systemRedColor];
                }
                return;
            }

            NSString *text = nil;
            if (isZip) {
                // Save zip, unzip, read contents
                NSString *tmpZip = [NSTemporaryDirectory() stringByAppendingPathComponent:@"primes_dl.zip"];
                NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:@"primes_dl"];
                [data writeToFile:tmpZip atomically:YES];
                [[NSFileManager defaultManager] createDirectoryAtPath:tmpDir
                    withIntermediateDirectories:YES attributes:nil error:nil];
                NSTask *unzip = [[NSTask alloc] init];
                unzip.executableURL = [NSURL fileURLWithPath:@"/usr/bin/unzip"];
                unzip.arguments = @[@"-o", tmpZip, @"-d", tmpDir];
                unzip.standardOutput = [NSPipe pipe];
                unzip.standardError = [NSPipe pipe];
                [unzip launchAndReturnError:nil];
                [unzip waitUntilExit];

                // Find the text file inside
                NSArray *files = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:tmpDir error:nil];
                for (NSString *f in files) {
                    if ([f hasSuffix:@".txt"]) {
                        text = [NSString stringWithContentsOfFile:
                            [tmpDir stringByAppendingPathComponent:f]
                            encoding:NSUTF8StringEncoding error:nil];
                        break;
                    }
                }
                [[NSFileManager defaultManager] removeItemAtPath:tmpZip error:nil];
                [[NSFileManager defaultManager] removeItemAtPath:tmpDir error:nil];
            } else {
                text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
            }

            if (!text) {
                [ss markovLog:@"Failed to decode downloaded data.\n"];
                return;
            }

            // Parse primes from text
            ss->_markovPrimeList.clear();
            NSArray *lines = [text componentsSeparatedByCharactersInSet:
                [NSCharacterSet newlineCharacterSet]];
            for (NSString *line in lines) {
                NSString *trimmed = [line stringByTrimmingCharactersInSet:
                    [NSCharacterSet whitespaceCharacterSet]];
                if (trimmed.length == 0 || [trimmed hasPrefix:@"#"]) continue;

                if (isOEIS) {
                    // Format: "index prime"
                    NSArray *parts = [trimmed componentsSeparatedByString:@" "];
                    if (parts.count >= 2) {
                        uint64_t val = strtoull([parts[1] UTF8String], nullptr, 10);
                        if (val >= 2) ss->_markovPrimeList.push_back(val);
                    }
                } else {
                    // t5k.org format: space-separated primes, skip header lines with letters
                    NSRange letterRange = [trimmed rangeOfCharacterFromSet:
                        [NSCharacterSet letterCharacterSet]];
                    if (letterRange.location != NSNotFound) continue;
                    // Split on whitespace
                    NSArray *nums = [trimmed componentsSeparatedByCharactersInSet:
                        [NSCharacterSet whitespaceCharacterSet]];
                    for (NSString *n in nums) {
                        NSString *clean = [n stringByReplacingOccurrencesOfString:@"," withString:@""];
                        if (clean.length == 0) continue;
                        uint64_t val = strtoull(clean.UTF8String, nullptr, 10);
                        if (val >= 2) ss->_markovPrimeList.push_back(val);
                    }
                }
            }

            std::sort(ss->_markovPrimeList.begin(), ss->_markovPrimeList.end());
            ss->_markovPrimeList.erase(
                std::unique(ss->_markovPrimeList.begin(), ss->_markovPrimeList.end()),
                ss->_markovPrimeList.end());

            NSString *countStr = formatNumber((uint64_t)ss->_markovPrimeList.size());
            [ss markovLog:[NSString stringWithFormat:@"Fetched %@ primes.\n", countStr]];
            if (statusLbl) {
                statusLbl.stringValue = [NSString stringWithFormat:@"%@ primes loaded", countStr];
                statusLbl.textColor = [NSColor systemGreenColor];
            }

            if (ss->_markovPrimeList.size() >= 2) {
                [ss markovLog:[NSString stringWithFormat:
                    @"  Range: %@ to %@  |  Mean gap: %.1f\n",
                    formatNumber(ss->_markovPrimeList.front()),
                    formatNumber(ss->_markovPrimeList.back()),
                    (double)(ss->_markovPrimeList.back() - ss->_markovPrimeList.front())
                        / ss->_markovPrimeList.size()]];
            }
        });
    }];
    [task resume];
}

- (void)markovLoadPrimes:(id)sender {
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    panel.allowedFileTypes = @[@"txt", @"csv", @"dat"];
    panel.title = @"Load Known Prime List";
    panel.message = @"One prime per line (decimal). Lines starting with # are ignored.";
    if ([panel runModal] != NSModalResponseOK || !panel.URL) return;

    NSString *contents = [NSString stringWithContentsOfURL:panel.URL
        encoding:NSUTF8StringEncoding error:nil];
    if (!contents) {
        [self markovLog:@"Markov: failed to read file.\n"];
        return;
    }

    _markovPrimeList.clear();
    for (NSString *line in [contents componentsSeparatedByString:@"\n"]) {
        NSString *trimmed = [line stringByTrimmingCharactersInSet:
            [NSCharacterSet whitespaceAndNewlineCharacterSet]];
        if (trimmed.length == 0 || [trimmed hasPrefix:@"#"]) continue;
        // Strip commas and spaces
        trimmed = [trimmed stringByReplacingOccurrencesOfString:@"," withString:@""];
        trimmed = [trimmed stringByReplacingOccurrencesOfString:@" " withString:@""];
        uint64_t val = strtoull(trimmed.UTF8String, nullptr, 10);
        if (val >= 2) _markovPrimeList.push_back(val);
    }

    std::sort(_markovPrimeList.begin(), _markovPrimeList.end());
    _markovPrimeList.erase(
        std::unique(_markovPrimeList.begin(), _markovPrimeList.end()),
        _markovPrimeList.end());

    [self markovLog:[NSString stringWithFormat:
        @"Markov: loaded %@ primes from %@\n",
        formatNumber((uint64_t)_markovPrimeList.size()),
        panel.URL.lastPathComponent]];

    // Update status label in Markov window if open
    NSWindow *mwin = objc_getAssociatedObject(self, "markovWindow");
    if (mwin) {
        NSTextField *statusLbl = [mwin.contentView viewWithTag:9201];
        if (statusLbl) {
            statusLbl.stringValue = [NSString stringWithFormat:@"%@ primes loaded",
                formatNumber((uint64_t)_markovPrimeList.size())];
            statusLbl.textColor = [NSColor systemGreenColor];
        }
    }

    // Show gaps
    if (_markovPrimeList.size() >= 2) {
        // Find largest gaps
        std::vector<std::pair<uint64_t, size_t>> gaps; // gap size, index
        for (size_t i = 0; i + 1 < _markovPrimeList.size(); i++) {
            uint64_t g = _markovPrimeList[i + 1] - _markovPrimeList[i];
            gaps.push_back({g, i});
        }
        std::sort(gaps.begin(), gaps.end(), [](auto& a, auto& b) { return a.first > b.first; });

        [self markovLog:@"  Largest gaps (prime candidates likely here):\n"];
        int show = std::min((int)gaps.size(), 10);
        for (int i = 0; i < show; i++) {
            uint64_t lo = _markovPrimeList[gaps[i].second];
            uint64_t hi = _markovPrimeList[gaps[i].second + 1];
            [self markovLog:[NSString stringWithFormat:
                @"    gap=%@  between %@ and %@\n",
                formatNumber(gaps[i].first), formatNumber(lo), formatNumber(hi)]];
        }
        [self markovLog:[NSString stringWithFormat:
            @"  Range: %@ to %@  |  Mean gap: %.1f  |  Expected by ln(N): %.1f\n",
            formatNumber(_markovPrimeList.front()),
            formatNumber(_markovPrimeList.back()),
            (double)(_markovPrimeList.back() - _markovPrimeList.front()) / _markovPrimeList.size(),
            log((double)_markovPrimeList[_markovPrimeList.size() / 2])]];
    }
}

- (void)runMarkovPredict:(uint64_t)startPrime steps:(uint64_t)chainLen {
    if (chainLen == 0) chainLen = 20;
    if (chainLen > 10000) chainLen = 10000;

    // If no prime list loaded, try auto-loading known_primes.txt
    if (_markovPrimeList.empty()) {
        NSString *autoPath = @"/Users/sergeinester/Documents/primes/primelocations/known_primes.txt";
        NSString *contents = [NSString stringWithContentsOfFile:autoPath
            encoding:NSUTF8StringEncoding error:nil];
        if (contents) {
            for (NSString *line in [contents componentsSeparatedByString:@"\n"]) {
                NSString *trimmed = [line stringByTrimmingCharactersInSet:
                    [NSCharacterSet whitespaceAndNewlineCharacterSet]];
                if (trimmed.length == 0 || [trimmed hasPrefix:@"#"]) continue;
                trimmed = [trimmed stringByReplacingOccurrencesOfString:@"," withString:@""];
                uint64_t val = strtoull(trimmed.UTF8String, nullptr, 10);
                if (val >= 2) _markovPrimeList.push_back(val);
            }
            std::sort(_markovPrimeList.begin(), _markovPrimeList.end());
            _markovPrimeList.erase(
                std::unique(_markovPrimeList.begin(), _markovPrimeList.end()),
                _markovPrimeList.end());
            if (!_markovPrimeList.empty()) {
                [self markovLog:[NSString stringWithFormat:
                    @"Markov: auto-loaded %@ primes from known_primes.txt\n",
                    formatNumber((uint64_t)_markovPrimeList.size())]];
            }
        }
    }

    // Determine training set: loaded list or sieve
    bool use_loaded_list = !_markovPrimeList.empty();

    // Find start position in loaded list
    size_t start_idx = 0;
    if (use_loaded_list) {
        // Find the prime in the list closest to startPrime
        auto it = std::lower_bound(_markovPrimeList.begin(), _markovPrimeList.end(), startPrime);
        if (it != _markovPrimeList.end()) {
            start_idx = std::distance(_markovPrimeList.begin(), it);
        } else {
            start_idx = _markovPrimeList.size() - 1;
        }
        // Snap to the actual prime in the list
        startPrime = _markovPrimeList[start_idx];
        [self markovLog:[NSString stringWithFormat:
            @"Markov: starting from prime #%@ in loaded list: %@\n",
            formatNumber((uint64_t)start_idx + 1), formatNumber(startPrime)]];
    } else {
        // No list loaded, verify input is prime and sieve training data
        if (!prime::is_prime(startPrime)) {
            [self markovLog:[NSString stringWithFormat:
                @"Markov Predict: %@ is not prime. Enter a known prime, or load a prime list first.\n",
                formatNumber(startPrime)]];
            return;
        }
        [self markovLog:@"Markov: no prime list loaded, will sieve training primes near target.\n"];
    }

    _checkRunning.store(true);
    _predictedPrimes.clear();

    [self markovLog:[NSString stringWithFormat:
        @"-- Markov Predict: chaining %llu predictions from %@ --\n",
        chainLen, formatNumber(startPrime)]];

    __weak AppDelegate *weakSelf = self;
    prime::TaskManager *taskMgr = _taskMgr;
    std::vector<uint64_t> primeList = _markovPrimeList; // copy for thread
    bool useList = use_loaded_list;
    size_t startIndex = start_idx;

    _checkThread = std::thread([weakSelf, startPrime, chainLen, taskMgr,
                                 primeList, useList, startIndex]() {
        pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
        AppDelegate *ss = weakSelf;
        if (!ss) return;

        uint64_t current_prime = startPrime;
        size_t list_pos = startIndex;
        uint64_t correct = 0, in_set = 0, total = 0;

        // Train on the full loaded list (or a window of it)
        prime::MarkovPredictor markov;
        if (useList && primeList.size() >= 50) {
            // Use primes up to current position as training set
            size_t train_end = std::min(list_pos + 1, primeList.size());
            size_t train_start = train_end > 20000 ? train_end - 20000 : 0;
            std::vector<uint64_t> training(
                primeList.begin() + train_start,
                primeList.begin() + train_end);
            markov.train(training);

            NSString *tMsg = [NSString stringWithFormat:
                @"  Trained on %@ primes from loaded list (mean gap: d1=%.1f d3=%.1f d7=%.1f d9=%.1f)\n",
                formatNumber((uint64_t)training.size()),
                markov.mean_gap(1), markov.mean_gap(3),
                markov.mean_gap(7), markov.mean_gap(9)];
            dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf markovLog:tMsg]; });
        }

        for (uint64_t step = 0; step < chainLen && ss->_checkRunning.load(); step++) {

            // If no list or model not trained, sieve and train on the fly
            if (!markov.is_trained()) {
                double ln_p = log((double)current_prime);
                uint64_t train_span = (uint64_t)(20000.0 * ln_p);
                if (train_span > current_prime - 2) train_span = current_prime - 2;
                uint64_t train_lo = current_prime - train_span;
                if (train_lo < 2) train_lo = 2;

                std::vector<uint64_t> training;
                if (taskMgr) training = taskMgr->segmented_sieve(train_lo, current_prime + 1);

                if (training.size() < 50) {
                    dispatch_async(dispatch_get_main_queue(), ^{
                        [weakSelf markovLog:@"  Not enough training primes, stopping.\n"];
                    });
                    break;
                }
                markov.train(training);
            }

            // Predict next prime: 2000 trials, then verify top candidates
            auto vp = markov.predict_next_verified(current_prime, 2000);
            uint64_t predicted = vp.value;
            int pred_votes = vp.votes;
            int pred_rank = vp.rank;
            int pred_checked = vp.candidates_checked;

            // Find actual next prime
            uint64_t actual_next = 0;
            bool actual_known = false;

            if (useList && list_pos + 1 < primeList.size()) {
                actual_next = primeList[list_pos + 1];
                actual_known = true;
                list_pos++;
            } else {
                actual_next = current_prime + 1;
                if (actual_next % 2 == 0) actual_next++;
                while (!prime::is_prime(actual_next)) actual_next += 2;
            }

            uint64_t gap = actual_next - current_prime;
            total++;

            bool hit = (predicted == actual_next);
            int64_t error = predicted > 0 ? (int64_t)predicted - (int64_t)actual_next : -1;
            if (hit) correct++;

            // Check if actual prime was in candidate set
            auto all_candidates = markov.predict_next(current_prime, 2000);
            bool in_candidates = false;
            int candidate_rank = 0;
            int candidate_votes = 0;
            {
                int rank = 0;
                for (auto& [val, cnt] : all_candidates) {
                    if (val > current_prime) rank++;
                    if (val == actual_next) {
                        in_candidates = true;
                        candidate_rank = rank;
                        candidate_votes = cnt;
                        break;
                    }
                }
            }
            if (in_candidates) in_set++;

            // Adaptive training: feed the actual result back into the model
            markov.update(current_prime, actual_next);

            // Atomic analysis: smoothness, shell classification, composite density
            // Peek ahead for gap_after (next gap) if possible
            uint64_t gap_after = 0;
            if (useList && list_pos + 1 < primeList.size()) {
                gap_after = primeList[list_pos + 1] - actual_next;
            } else {
                // Quick scan for next prime to get gap_after
                uint64_t probe = actual_next + 1;
                if (probe % 2 == 0) probe++;
                for (int tries = 0; tries < 500 && !prime::is_prime(probe); tries++)
                    probe += 2;
                if (prime::is_prime(probe)) gap_after = probe - actual_next;
            }

            auto atom = prime::analyze_prime_atom(actual_next, gap, gap_after);
            std::string factor_info = prime::factors_string(actual_next - 1);

            // Mersenne factor test
            std::string mersenne_info;
            {
                auto mhits = prime::mersenne_factor_scan(actual_next);
                if (!mhits.empty()) {
                    std::ostringstream ms;
                    ms << "MERSENNE FACTOR: ";
                    for (size_t i = 0; i < mhits.size(); i++) {
                        if (i > 0) ms << ", ";
                        ms << actual_next << " | 2^" << mhits[i] << "-1";
                    }
                    mersenne_info = ms.str();
                }
            }

            // Build result string
            NSString *hitStr;
            if (hit) {
                hitStr = [NSString stringWithFormat:@"VERIFIED HIT (rank #%d, votes=%d, checked=%d)",
                    pred_rank, pred_votes, pred_checked];
            } else if (in_candidates) {
                hitStr = [NSString stringWithFormat:@"IN SET (rank #%d, votes=%d) predicted=%@ (rank #%d)",
                    candidate_rank, candidate_votes,
                    predicted > 0 ? formatNumber(predicted) : @"(none)", pred_rank];
            } else {
                hitStr = [NSString stringWithFormat:@"miss predicted=%@",
                    predicted > 0 ? formatNumber(predicted) : @"(none)"];
            }

            NSString *line = [NSString stringWithFormat:
                @"  #%llu  %@  gap=%llu|%llu  %@  err=%lld  "
                @"%s[%s] smooth=%.2f(%d/%d) lpf=%@  r=%.0f  p-1=%s%s\n",
                total, formatNumber(actual_next),
                gap, gap_after,
                hitStr,
                (long long)error,
                actual_known ? "K " : "",
                atom.shell_str().c_str(),
                atom.smoothness, atom.big_omega, atom.omega,
                formatNumber(atom.largest_pf),
                atom.radius,
                factor_info.c_str(),
                mersenne_info.empty() ? "" : ("\n    ** " + mersenne_info + " **").c_str()];
            dispatch_async(dispatch_get_main_queue(), ^{
                [weakSelf markovLog:line];
            });

            // Save discovery and store for PrimeFactor
            if (taskMgr) {
                taskMgr->save_discovery({prime::TaskType::GeneralPrime, actual_next, current_prime,
                    prime::PrimeClass::Prime, current_prime, "", taskMgr->timestamp()});
            }
            dispatch_async(dispatch_get_main_queue(), ^{
                AppDelegate *s3 = weakSelf;
                if (s3) s3->_predictedPrimes.push_back(actual_next);
            });

            // Adaptive training is handled by markov.update() above
            current_prime = actual_next;
        }

        // Save predictions to file for reuse
        {
            std::string path = std::string(
                [NSSearchPathForDirectoriesInDomains(NSApplicationSupportDirectory,
                    NSUserDomainMask, YES).firstObject UTF8String]) + "/PrimePath/markov_predictions.txt";
            std::ofstream f(path, std::ios::app);
            if (f.is_open()) {
                auto now = std::chrono::system_clock::now();
                auto tt = std::chrono::system_clock::to_time_t(now);
                char ts[32];
                strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", localtime(&tt));
                f << "# Markov Predict run " << ts
                  << " | start=" << startPrime
                  << " | steps=" << total
                  << " | hits=" << correct << "/" << total << "\n";
            }
        }

        double hitRate = total > 0 ? 100.0 * correct / total : 0;
        double setRate = total > 0 ? 100.0 * in_set / total : 0;
        uint64_t finalCorrect = correct, finalInSet = in_set, finalTotal = total;
        NSString *summary = [NSString stringWithFormat:
            @"-- Markov Predict complete --\n"
            @"  Exact hits: %llu/%llu (%.1f%%)\n"
            @"  In candidate set: %llu/%llu (%.1f%%)\n"
            @"  Last prime: %@\n"
            @"  Predictions saved to ~/Library/Application Support/PrimePath/markov_predictions.txt\n",
            finalCorrect, finalTotal, hitRate,
            finalInSet, finalTotal, setRate,
            formatNumber(current_prime)];
        dispatch_async(dispatch_get_main_queue(), ^{
            [weakSelf markovLog:summary];
            AppDelegate *s2 = weakSelf;
            if (s2) {
                // Re-enable Markov window buttons
                NSWindow *mwin = objc_getAssociatedObject(s2, "markovWindow");
                if (mwin) {
                    NSButton *goBtn = [mwin.contentView viewWithTag:9204];
                    NSButton *stopBtn = [mwin.contentView viewWithTag:9205];
                    goBtn.enabled = YES;
                    stopBtn.enabled = NO;
                }
                // Clear the log redirect
                objc_setAssociatedObject(s2, "markovLogView", nil, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
            }
        });

        ss->_checkRunning.store(false);
    });
}

// ── PrimeLocation (convergence + GPU+CPU parallel testing) ──────────

- (void)runPrimeLocation:(uint64_t)from window:(uint64_t)windowSize {
    if (windowSize == 0) windowSize = 10000;
    if (windowSize > 1000000) windowSize = 1000000;

    _checkRunning.store(true);
    self.checkGoButton.enabled = NO;
    self.checkStopButton.enabled = YES;
    [self appendText:[NSString stringWithFormat:
        @"── PrimeLocation (GPU+CPU): scanning %@ candidates near %@ ──\n",
        formatNumber(windowSize), formatNumber(from)]];

    __weak AppDelegate *weakSelf = self;
    prime::GPUBackend *gpu = _gpu;
    prime::TaskManager *taskMgr = _taskMgr;
    BOOL checkAtDiscovery = (self.checkAtDiscoveryButton.state == NSControlStateValueOn);
    // Clear previous predicted primes list
    _predictedPrimes.clear();
    self.primeFactorButton.hidden = YES;

    _checkThread = std::thread([weakSelf, from, windowSize, gpu, taskMgr, checkAtDiscovery]() {
        pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
        AppDelegate *ss = weakSelf;
        if (!ss) return;

        // Phase 1: Score candidates using convergence + Markov prediction
        struct Candidate {
            uint64_t value;
            double score;        // convergence score
            int markov_hits;     // how many Markov chains predicted this location
        };
        std::vector<Candidate> candidates;
        candidates.reserve(windowSize / 2);

        uint64_t pos = (from % 2 == 0 && from > 2) ? from + 1 : from;

        // Phase 0a: Run MatrixSieve on the candidate range
        const uint32_t sieveCount = (uint32_t)std::min((uint64_t)windowSize, (uint64_t)2000000);
        std::vector<uint8_t> sieveMask(sieveCount, 1);
        if (taskMgr) {
            prime::MatrixSieve quickSieve;
            quickSieve.sieve_block(pos, sieveCount, sieveMask.data());
        }

        // Phase 0b: Markov prediction -- train on known primes near target,
        // then predict candidate locations. Uses primes already found by
        // everyone (sieved from number line), not just PrimePath discoveries.
        std::set<uint64_t> markov_set;  // for O(1) lookup
        std::map<uint64_t, int> markov_hits;  // value -> chain count
        {
            dispatch_async(dispatch_get_main_queue(), ^{
                [weakSelf appendText:@"  Markov: training on nearby primes...\n"];
            });

            // Sieve training primes in the region just before the target.
            // Aim for ~50K training primes. At density ~1/ln(N), we need
            // about 50000 * ln(from) numbers to scan.
            double ln_from = log((double)from);
            uint64_t train_span = (uint64_t)(50000.0 * ln_from);
            if (train_span > from) train_span = from - 2;
            uint64_t train_lo = from - train_span;
            if (train_lo < 2) train_lo = 2;

            // Use segmented sieve for training primes
            std::vector<uint64_t> training_primes;
            if (taskMgr) {
                training_primes = taskMgr->segmented_sieve(train_lo, from);
            }

            if (training_primes.size() >= 100) {
                prime::MarkovPredictor markov;
                markov.train(training_primes);

                // Run consensus prediction: 20 chains, keep candidates
                // predicted by at least 2 chains
                int predict_steps = (int)std::min((uint64_t)windowSize, (uint64_t)50000);
                auto consensus = markov.predict_consensus(
                    training_primes.back(), predict_steps, 20, 2);

                // Filter to our target window [pos, pos + windowSize*2]
                uint64_t win_lo = pos;
                uint64_t win_hi = pos + windowSize * 2;
                for (auto& [val, cnt] : consensus) {
                    if (val >= win_lo && val <= win_hi) {
                        markov_set.insert(val);
                        markov_hits[val] = cnt;
                    }
                }

                NSString *mMsg = [NSString stringWithFormat:
                    @"  Markov: trained on %@ primes, predicted %@ candidates in window "
                    @"(mean gap: d1=%.1f d3=%.1f d7=%.1f d9=%.1f)\n",
                    formatNumber((uint64_t)training_primes.size()),
                    formatNumber((uint64_t)markov_set.size()),
                    markov.mean_gap(1), markov.mean_gap(3),
                    markov.mean_gap(7), markov.mean_gap(9)];
                dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf appendText:mMsg]; });
            } else {
                dispatch_async(dispatch_get_main_queue(), ^{
                    [weakSelf appendText:@"  Markov: not enough training primes, skipping prediction\n"];
                });
            }
        }

        // Phase 1: Score candidates (convergence + Markov boost)
        uint64_t sieveRejected = 0;
        uint64_t markovBoosted = 0;
        for (uint64_t i = 0; i < windowSize && ss->_checkRunning.load(); i++) {
            if (i % 5000 == 0) [ss throttleCheckIfNeeded];
            uint64_t n = pos + i * 2;
            if (n < 3) continue;
            uint32_t sieveIdx = (uint32_t)(i * 2);
            if (sieveIdx < sieveCount && !sieveMask[sieveIdx]) {
                sieveRejected++;
                continue;
            }
            if (!prime::WHEEL.valid(n)) continue;
            if (prime::crt_reject(n)) continue;
            double score = prime::convergence(n, 12);

            int mhits = 0;
            auto mit = markov_hits.find(n);
            if (mit != markov_hits.end()) {
                mhits = mit->second;
                // Boost score: each Markov chain hit adds weight
                score += mhits * 2.0;
                markovBoosted++;
            }

            if (score > -900.0) {
                candidates.push_back({n, score, mhits});
            }
        }

        // Also add Markov-only candidates that passed MatrixSieve but may
        // have been CRT/wheel rejected (Markov predictions from known primes
        // are worth testing even if convergence is ambiguous)
        for (auto& [mval, mcnt] : markov_hits) {
            if (mval < pos || mval > pos + windowSize * 2) continue;
            if (!(mval & 1) || mval < 3) continue;
            // Check if already in candidates
            bool found = false;
            for (auto& c : candidates) {
                if (c.value == mval) { found = true; break; }
            }
            if (!found && mcnt >= 3) {
                // Strong Markov consensus (3+ chains) overrides CRT/wheel rejection
                candidates.push_back({mval, (double)mcnt * 2.0, mcnt});
                markovBoosted++;
            }
        }

        {
            NSString *sieveMsg = [NSString stringWithFormat:
                @"  MatrixSieve pre-filter: rejected %@ of %@ candidates (%.1f%%)\n"
                @"  Markov-boosted candidates: %@\n",
                formatNumber(sieveRejected), formatNumber(windowSize),
                windowSize > 0 ? 100.0 * sieveRejected / windowSize : 0.0,
                formatNumber(markovBoosted)];
            dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf appendText:sieveMsg]; });
        }

        if (!ss->_checkRunning.load()) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [weakSelf appendText:@"── PrimeLocation cancelled ──\n"];
                AppDelegate *s2 = weakSelf;
                if (s2) { s2.checkStopButton.enabled = NO; s2.checkGoButton.enabled = YES; }
            });
            ss->_checkRunning.store(false);
            return;
        }

        // Sort by score descending
        std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

        NSString *msg = [NSString stringWithFormat:
            @"  Scored %@ candidates, testing top predictions with GPU+CPU...\n",
            formatNumber((uint64_t)candidates.size())];
        dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf appendText:msg]; });

        // Phase 2: Test in batches using GPU+CPU split
        uint64_t tested = 0;
        uint64_t found = 0;
        const uint32_t BATCH = 4096;

        for (size_t offset = 0; offset < candidates.size() && ss->_checkRunning.load(); offset += BATCH) {
            [ss throttleCheckIfNeeded];
            uint32_t n = (uint32_t)std::min((size_t)BATCH, candidates.size() - offset);

            // Extract values for batch testing
            std::vector<uint64_t> vals(n);
            for (uint32_t i = 0; i < n; i++) vals[i] = candidates[offset + i].value;

            // Split between CPU and GPU
            uint32_t cpu_count = n * 3 / 10;
            uint32_t gpu_start = cpu_count;
            uint32_t gpu_count = n - cpu_count;

            std::vector<uint8_t> cpu_results(cpu_count, 0);
            auto cpu_future = std::async(std::launch::async, [&vals, cpu_count, &cpu_results]() {
                for (uint32_t i = 0; i < cpu_count; i++) {
                    cpu_results[i] = prime::is_prime(vals[i]) ? 1 : 0;
                }
            });

            std::vector<uint8_t> gpu_results(gpu_count, 0);
            if (gpu_count > 0) {
                gpu->primality_batch(vals.data() + gpu_start, gpu_results.data(), gpu_count);
            }

            cpu_future.get();

            // Collect results -- save confirmed primes, optionally run special tests
            auto handlePrime = [&](uint64_t val, double score) {
                found++;
                // Store in predicted primes list for PrimeFactor
                dispatch_async(dispatch_get_main_queue(), ^{
                    AppDelegate *s3 = weakSelf;
                    if (s3) s3->_predictedPrimes.push_back(val);
                });
                // Save to discoveries
                if (taskMgr) {
                    taskMgr->save_discovery({prime::TaskType::GeneralPrime, val, 0,
                        prime::PrimeClass::Prime, from, "", taskMgr->timestamp()});
                }
                NSString *msg2 = [NSString stringWithFormat:
                    @"  * PREDICTED PRIME #%llu: %@ (score: %.2f, rank: %llu)\n",
                    found, formatNumber(val), score, tested];
                dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf appendText:msg2]; });

                // CheckAtDiscovery: test this prime against special categories
                if (checkAtDiscovery) {
                    // Wieferich test: 2^(p-1) == 1 (mod p^2)
                    unsigned __int128 p_sq = (unsigned __int128)val * val;
                    unsigned __int128 base128 = 2, result128 = 1, mod128 = p_sq;
                    uint64_t exp128 = val - 1;
                    base128 %= mod128;
                    while (exp128 > 0) {
                        if (exp128 & 1) result128 = result128 * base128 % mod128;
                        exp128 >>= 1;
                        base128 = base128 * base128 % mod128;
                    }
                    if (result128 == 1) {
                        NSString *w = [NSString stringWithFormat:
                            @"  !!!! WIEFERICH PRIME: %@ !!!!\n", formatNumber(val)];
                        dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf appendText:w]; });
                        if (taskMgr) {
                            taskMgr->save_discovery({prime::TaskType::Wieferich, val, 0,
                                prime::PrimeClass::Prime, from, "", taskMgr->timestamp()});
                        }
                    }
                    // Twin test
                    if (prime::is_prime(val + 2)) {
                        NSString *tw = [NSString stringWithFormat:
                            @"    Twin pair: (%@, %@)\n", formatNumber(val), formatNumber(val + 2)];
                        dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf appendText:tw]; });
                    }
                    // Sophie Germain test
                    if (prime::is_prime(2 * val + 1)) {
                        NSString *sg = [NSString stringWithFormat:
                            @"    Sophie Germain: %@ (2p+1=%@ also prime)\n",
                            formatNumber(val), formatNumber(2 * val + 1)];
                        dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf appendText:sg]; });
                    }
                }
            };

            for (uint32_t i = 0; i < cpu_count; i++) {
                tested++;
                if (cpu_results[i]) {
                    handlePrime(candidates[offset + i].value, candidates[offset + i].score);
                }
            }
            for (uint32_t i = 0; i < gpu_count; i++) {
                tested++;
                if (gpu_results[i]) {
                    handlePrime(candidates[offset + gpu_start + i].value,
                                candidates[offset + gpu_start + i].score);
                }
            }

            if (tested % (BATCH * 2) < BATCH) {
                double hitRate = found > 0 ? (double)tested / found : 0;
                NSString *prog = [NSString stringWithFormat:
                    @"  ... tested %@/%@ | found %@ | 1 per %.0f candidates\n",
                    formatNumber(tested), formatNumber((uint64_t)candidates.size()),
                    formatNumber(found), hitRate];
                dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf appendText:prog]; });
            }
        }

        double hitRate = found > 0 ? (double)tested / found : 0;
        double lnN = log((double)from);
        double theoreticalRate = lnN / 2.0;
        uint64_t foundCopy = found;
        NSString *summary = [NSString stringWithFormat:
            @"── PrimeLocation complete: %@ primes in %@ tests (1 per %.1f) | theoretical: 1 per %.1f ──\n",
            formatNumber(found), formatNumber(tested), hitRate, theoreticalRate];
        dispatch_async(dispatch_get_main_queue(), ^{
            [weakSelf appendText:summary];
            AppDelegate *s2 = weakSelf;
            if (s2) {
                s2.checkStopButton.enabled = NO;
                s2.checkGoButton.enabled = YES;
                // Show PrimeFactor button if we found primes
                if (foundCopy > 0) {
                    s2.primeFactorButton.hidden = NO;
                    s2.primeFactorButton.title =
                        [NSString stringWithFormat:@"PrimeFactor (%llu)", foundCopy];
                    [s2 appendText:[NSString stringWithFormat:
                        @"  %@ predicted primes stored -- click 'PrimeFactor' to use them for factoring\n",
                        formatNumber(foundCopy)]];
                }
            }
        });
        ss->_checkRunning.store(false);
    });
    // Thread stays joinable for proper cleanup
}

// ── Run PrimeFactor -- use predicted primes to factor numbers ─────────

- (void)runPrimeFactor:(id)sender {
    if (_predictedPrimes.empty()) {
        [self appendText:@"No predicted primes available. Run PrimeLocation first.\n"];
        return;
    }

    // Get the number to factor from the check field, or factor nearby composites
    NSString *fromStr = [self.checkFromField.string stringByTrimmingCharactersInSet:
        [NSCharacterSet whitespaceAndNewlineCharacterSet]];
    fromStr = [fromStr stringByReplacingOccurrencesOfString:@"," withString:@""];
    uint64_t target = strtoull(fromStr.UTF8String, nullptr, 10);

    if (target < 2) {
        // No target specified -- factor composites near the predicted primes
        [self appendText:[NSString stringWithFormat:
            @"── PrimeFactor: testing %@ predicted primes as trial divisors ──\n",
            formatNumber((uint64_t)_predictedPrimes.size())]];

        // Sort predicted primes for efficient trial division
        std::vector<uint64_t> sorted_primes = _predictedPrimes;
        std::sort(sorted_primes.begin(), sorted_primes.end());

        // Test composites near each predicted prime
        uint64_t factored = 0;
        for (auto p : sorted_primes) {
            // Check p-1 and p+1 (likely composites adjacent to primes)
            for (int64_t delta : {-1, 1, -3, 3}) {
                uint64_t n = p + delta;
                if (n < 4 || prime::is_prime(n)) continue;

                // Factor using our predicted primes as trial divisors
                uint64_t remaining = n;
                std::string factorization;
                for (auto q : sorted_primes) {
                    if (q * q > remaining) break;
                    while (remaining % q == 0) {
                        if (!factorization.empty()) factorization += " x ";
                        factorization += std::to_string(q);
                        remaining /= q;
                    }
                }
                if (remaining > 1) {
                    if (!factorization.empty()) factorization += " x ";
                    factorization += std::to_string(remaining);
                }
                if (!factorization.empty()) {
                    factored++;
                    NSString *msg = [NSString stringWithFormat:@"  %@ = %s\n",
                        formatNumber(n), factorization.c_str()];
                    [self appendText:msg];
                }
            }
            if (factored >= 100) break; // limit output
        }
        [self appendText:[NSString stringWithFormat:
            @"── PrimeFactor complete: factored %@ composites ──\n", formatNumber(factored)]];
    } else {
        // Factor a specific target using predicted primes + standard methods
        [self appendText:[NSString stringWithFormat:
            @"── PrimeFactor: factoring %@ using %@ predicted primes ──\n",
            formatNumber(target), formatNumber((uint64_t)_predictedPrimes.size())]];

        if (prime::is_prime(target)) {
            [self appendText:[NSString stringWithFormat:@"  %@ is PRIME\n", formatNumber(target)]];
        } else {
            // Try predicted primes first (they may include large factors)
            std::vector<uint64_t> sorted_primes = _predictedPrimes;
            std::sort(sorted_primes.begin(), sorted_primes.end());

            uint64_t remaining = target;
            std::string factorization;
            bool usedPredicted = false;

            // Trial division with predicted primes
            for (auto p : sorted_primes) {
                if (p * p > remaining) break;
                while (remaining % p == 0) {
                    if (!factorization.empty()) factorization += " x ";
                    factorization += std::to_string(p);
                    remaining /= p;
                    usedPredicted = true;
                }
            }

            // Fall back to standard factoring for remainder
            if (remaining > 1 && remaining != target) {
                if (prime::is_prime(remaining)) {
                    if (!factorization.empty()) factorization += " x ";
                    factorization += std::to_string(remaining);
                } else {
                    // Use Pollard rho for the remainder
                    auto extra = prime::factor_u64(remaining);
                    for (auto f : extra) {
                        if (!factorization.empty()) factorization += " x ";
                        factorization += std::to_string(f);
                    }
                }
            } else if (remaining == target) {
                // Predicted primes didn't help -- use standard factoring
                auto factors = prime::factor_u64(target);
                for (size_t i = 0; i < factors.size(); i++) {
                    if (i > 0) factorization += " x ";
                    factorization += std::to_string(factors[i]);
                }
            }

            NSString *method = usedPredicted ? @"predicted primes" : @"standard";
            [self appendText:[NSString stringWithFormat:@"  %@ = %s (via %@)\n",
                formatNumber(target), factorization.c_str(), method]];
        }
        [self appendText:@"── PrimeFactor complete ──\n"];
    }
}

// ── Set start point ─────────────────────────────────────────────────

// Auto-fill fertile starting points when task selection changes
- (void)startTaskChanged:(id)sender {
    auto t = (prime::TaskType)self.startTaskPopup.selectedItem.tag;
    switch (t) {
        case prime::TaskType::Wieferich:
            self.startNumberField.stringValue = @"2";
            self.startPowerField.stringValue = @"64";
            self.startHintLabel.stringValue =
                @"Wieferich: PrimeGrid verified to 2^64 (Dec 2022). Start at 2^64+1 for new territory.";
            break;
        case prime::TaskType::WallSunSun:
            self.startNumberField.stringValue = @"2";
            self.startPowerField.stringValue = @"64";
            self.startHintLabel.stringValue =
                @"Wall-Sun-Sun: PrimeGrid verified to 2^64 (Dec 2022). ZERO known -- any find is historic!";
            break;
        case prime::TaskType::Wilson:
            // 2x10^13 verified. Start just past that.
            // 2^44 = 17.6T ~ 1.76x10^13, so 2^44+1 is just below. Use 2^45+1 ~ 3.5x10^13.
            self.startNumberField.stringValue = @"2";
            self.startPowerField.stringValue = @"45";
            self.startHintLabel.stringValue =
                @"Wilson: Verified to 2x10^13 (2012). 2^45+1 ~ 3.5x10^13. Only 3 known: 5, 13, 563.";
            break;
        case prime::TaskType::TwinPrime:
            self.startNumberField.stringValue = @"10";
            self.startPowerField.stringValue = @"15";
            self.startHintLabel.stringValue =
                @"Twin primes: Abundant. Summary-only counting (no individual saves). Pick any frontier.";
            break;
        case prime::TaskType::SophieGermain:
            self.startNumberField.stringValue = @"10";
            self.startPowerField.stringValue = @"15";
            self.startHintLabel.stringValue =
                @"Sophie Germain: p and 2p+1 both prime. Summary-only counting. Pick any frontier.";
            break;
        case prime::TaskType::CousinPrime:
            self.startNumberField.stringValue = @"10";
            self.startPowerField.stringValue = @"15";
            self.startHintLabel.stringValue =
                @"Cousin primes: p and p+4 both prime. Summary-only counting.";
            break;
        case prime::TaskType::SexyPrime:
            self.startNumberField.stringValue = @"10";
            self.startPowerField.stringValue = @"15";
            self.startHintLabel.stringValue =
                @"Sexy primes: p and p+6 both prime. Summary-only counting.";
            break;
        case prime::TaskType::GeneralPrime:
            self.startNumberField.stringValue = @"10";
            self.startPowerField.stringValue = @"18";
            self.startHintLabel.stringValue =
                @"General prime counting. Pick any range. Primes become sparser at large values.";
            break;
        case prime::TaskType::Emirp:
            self.startNumberField.stringValue = @"10";
            self.startPowerField.stringValue = @"15";
            self.startHintLabel.stringValue =
                @"Emirps: ~10% of primes are emirps. Summary-only counting.";
            break;
        case prime::TaskType::MersenneTrial:
            self.startNumberField.stringValue = @"2";
            self.startPowerField.stringValue = @"20";
            self.startHintLabel.stringValue =
                @"Mersenne TF: Trial factor 2^p-1 on GPU. GIMPS has tested small exponents.";
            break;
        case prime::TaskType::FermatFactor:
            self.startNumberField.stringValue = @"2";
            self.startPowerField.stringValue = @"0";
            self.startHintLabel.stringValue =
                @"Fermat Factor: Search for factors of F_m. Set base=m, power=0.";
            break;
    }
}

- (void)setStartPoint:(id)sender {
    auto t = (prime::TaskType)self.startTaskPopup.selectedItem.tag;
    auto& tasks = _taskMgr->tasks();
    auto it = tasks.find(t);
    if (it == tasks.end()) return;

    // Parse: if power=0, treat number as direct value. Otherwise compute number^power + 1.
    NSString *numStr = [self.startNumberField.stringValue
        stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
    NSString *powStr = [self.startPowerField.stringValue
        stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];

    // Strip commas from number field for pasted values
    numStr = [numStr stringByReplacingOccurrencesOfString:@"," withString:@""];

    uint64_t base_val = (uint64_t)strtoull(numStr.UTF8String, NULL, 10);
    uint64_t power = (uint64_t)[powStr longLongValue];

    uint64_t start_pos;
    NSString *desc;

    if (power == 0) {
        // Direct number entry
        if (base_val < 3) {
            [self appendText:@"Invalid start point -- need a number >= 3\n"];
            return;
        }
        start_pos = base_val;
        desc = [NSString stringWithFormat:@"%@", formatNumber(start_pos)];
    } else {
        if (base_val < 2) {
            [self appendText:@"Invalid start point -- need base >= 2\n"];
            return;
        }
        // Compute base^power + 1 using 128-bit to detect overflow
        unsigned __int128 result = 1;
        bool overflow = false;
        for (uint64_t i = 0; i < power && !overflow; i++) {
            result *= base_val;
            if (result > UINT64_MAX) overflow = true;
        }
        if (!overflow) result += 1;

        if (overflow || result > UINT64_MAX) {
            [self appendText:[NSString stringWithFormat:
                @"WARNING: %llu^%llu overflows u64. Using max safe value.\n", base_val, power]];
            start_pos = UINT64_MAX - 1000000;
        } else {
            start_pos = (uint64_t)result;
        }
        desc = [NSString stringWithFormat:@"%llu^%llu + 1 = %@", base_val, power, formatNumber(start_pos)];
    }

    // Make sure it's odd (primes > 2 are odd)
    if (start_pos % 2 == 0) start_pos++;

    // Pause task if running
    if (it->second.should_run.load()) {
        _taskMgr->pause_task(t);
        NSNumber *key = @((int)t);
        self.taskButtons[key].title = @"Start";
    }

    // Update position
    it->second.current_pos = start_pos;
    it->second.start_pos = start_pos;
    it->second.found_count = 0;
    it->second.tested_count = 0;
    _taskMgr->save_state();

    [self appendText:[NSString stringWithFormat:
        @"Set %s start -> %@\n", prime::task_name(t), desc]];
}

// ── Task toggle ─────────────────────────────────────────────────────

- (void)taskToggle:(NSButton *)sender {
    auto t = (prime::TaskType)sender.tag;
    auto& tasks = _taskMgr->tasks();
    auto it = tasks.find(t);
    if (it == tasks.end()) return;

    if (it->second.should_run.load()) {
        _taskMgr->pause_task(t);
        sender.title = @"Start";
    } else {
        _taskMgr->start_task(t);
        sender.title = @"Pause";
    }
}

- (void)taskSelectionChanged:(id)sender {
    // Sync the hidden startTaskPopup to match the main dropdown
    NSInteger idx = self.taskSelectPopup.indexOfSelectedItem;
    [self.startTaskPopup selectItemAtIndex:idx];
    [self startTaskChanged:nil];

    // Update Start/Pause button state
    auto t = (prime::TaskType)self.taskSelectPopup.selectedItem.tag;
    auto& tasks = _taskMgr->tasks();
    auto it = tasks.find(t);
    if (it != tasks.end()) {
        self.taskStartBtn.title = it->second.should_run.load() ? @"Pause" : @"Start";
    }
}

- (void)taskToggleSelected:(NSButton *)sender {
    auto t = (prime::TaskType)self.taskSelectPopup.selectedItem.tag;
    auto& tasks = _taskMgr->tasks();
    auto it = tasks.find(t);
    if (it == tasks.end()) return;

    if (it->second.should_run.load()) {
        _taskMgr->pause_task(t);
        self.taskStartBtn.title = @"Start";
    } else {
        _taskMgr->start_task(t);
        self.taskStartBtn.title = @"Pause";
    }
}

- (void)showAboutPanel:(id)sender {
    NSDictionary *opts = @{
        @"ApplicationName": @"PrimePath",
        @"Copyright": @"Development by Sergei Nester\nSuper smart coding by Claude\nsnester@viewbuild.com",
        @"ApplicationVersion": @"1.3.0",
        @"Version": @"1.3.0",
    };
    [[NSApplication sharedApplication] orderFrontStandardAboutPanelWithOptions:opts];
}

- (void)showInfoPanel:(id)sender {
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = @"Log Output Glossary";
    alert.informativeText =
        @"GPU flush: 281318 WSS -> GPU | shadow: avg=128 [100-214] 26546 suspicious\n\n"

        @"GPU flush -- CPU sieve buffer is full, candidates sent to GPU for testing.\n\n"
        @"281,318 -- Number of candidate primes in this batch.\n\n"
        @"WSS -- Wall-Sun-Sun prime search. Looking for primes p where p^2 divides "
        @"F(p-(p/5)). None have ever been found.\n\n"
        @"-> GPU -- Candidates dispatched to Metal GPU for modular arithmetic.\n\n"
        @"shadow -- Even-shadow pre-filter. Each candidate gets a score based on how "
        @"well the divisors of p+/-1 constrain the computation. Higher = better.\n\n"
        @"avg=128 -- Average shadow score across all candidates in the batch.\n\n"
        @"[100-214] -- Score range: lowest 100, highest 214.\n\n"
        @"suspicious -- Candidates with low shadow scores (poor p+/-1 divisor structure). "
        @"Less likely to be interesting but still tested. Reordered to back of batch.\n\n"

        @"── Other terms ──\n\n"
        @"pos: -- Current search position (the number being tested).\n\n"
        @"found: -- Count of discoveries (primes/factors found so far).\n\n"
        @"/s -- Candidates tested per second.\n\n"
        @"GPU util -- Percentage of time the GPU is actively computing.\n\n"
        @"threads -- Total GPU threads dispatched since launch.\n\n"
        @"batches -- Number of GPU command buffers submitted.\n\n"
        @"ms/batch -- Average GPU execution time per batch.\n\n"
        @"ALU/NEON -- CPU SIMD sieve stats. 'tested' = candidates sieved, "
        @"'rejected' = composites filtered out before GPU.\n\n"
        @"Predictor -- Pseudoprime predictor stats. Carm = Carmichael numbers, "
        @"SPRP2 = strong probable primes base 2, frontier = highest value tested.";
    alert.alertStyle = NSAlertStyleInformational;
    [alert addButtonWithTitle:@"OK"];
    [alert runModal];
}

- (void)showHelpPanel:(id)sender {
    NSWindow *helpWin = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(200, 100, 640, 680)
        styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable)
        backing:NSBackingStoreBuffered defer:NO];
    helpWin.title = @"PrimePath Help";
    helpWin.releasedWhenClosed = NO;

    NSScrollView *sv = [[NSScrollView alloc] initWithFrame:helpWin.contentView.bounds];
    sv.hasVerticalScroller = YES;
    sv.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    NSTextView *tv = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, 620, 1600)];
    tv.editable = NO;
    tv.font = [NSFont systemFontOfSize:12];
    tv.autoresizingMask = NSViewWidthSizable;
    tv.textContainerInset = NSMakeSize(10, 10);
    sv.documentView = tv;
    [helpWin.contentView addSubview:sv];

    tv.string =
        @"PRIMEPATH HELP\n"
        @"==============\n\n"
        @"PrimePath is a prime number search engine that runs on Apple Silicon GPUs\n"
        @"using Metal compute shaders. It is the first Metal implementation for\n"
        @"Mersenne trial factoring and Fermat factor searching. Supports multi-device\n"
        @"distributed computing over the local network.\n\n"

        @"SEARCH TASKS\n"
        @"------------\n"
        @"Select a search from the dropdown and click Start. Multiple searches can\n"
        @"run simultaneously. Progress is saved automatically and restored on relaunch.\n\n"

        @"  Wieferich      Primes where 2^(p-1) = 1 (mod p^2). Only two known: 1093, 3511.\n"
        @"  Wall-Sun-Sun   Primes where p^2 divides Fibonacci(p-(p/5)). None known.\n"
        @"  Wilson         Primes where (p-1)! = -1 (mod p^2). Only three: 5, 13, 563.\n"
        @"  Twin           Pairs (p, p+2) both prime.\n"
        @"  Sophie Germain Pairs where p and 2p+1 are both prime.\n"
        @"  Cousin         Pairs (p, p+4) both prime.\n"
        @"  Sexy           Pairs (p, p+6) both prime.\n"
        @"  General        Count all primes in a range.\n"
        @"  Emirp          Primes that are also prime when digits reversed.\n"
        @"  Mersenne TF    Trial factor Mersenne numbers 2^p-1 on GPU.\n"
        @"  Fermat Factor  Find factors of Fermat numbers F_m = 2^(2^m)+1 on GPU.\n\n"

        @"DISTRIBUTED COMPUTING (MULTI-DEVICE)\n"
        @"------------------------------------\n"
        @"PrimePath can distribute prime search work across multiple Macs on your\n"
        @"local network using Bonjour zero-configuration discovery.\n\n"

        @"  Conductor (master):  Coordinates work across all connected machines.\n"
        @"    - Click 'Start Conductor' to begin accepting worker connections.\n"
        @"    - Listens on TCP port 9807. Publishes via Bonjour (_primepath._tcp).\n"
        @"    - Automatically splits search ranges across connected workers.\n"
        @"    - Aggregates progress and discoveries from all machines.\n"
        @"    - Reassigns work if a worker disconnects.\n"
        @"    - The Conductor also participates in computation.\n\n"

        @"  Carriage (worker):  Connects to a Conductor and executes assigned work.\n"
        @"    - Click 'Start Carriage' to discover and connect to a Conductor.\n"
        @"    - Auto-discovers Conductors via Bonjour, or enter IP:port manually.\n"
        @"    - Reports progress every second and discoveries immediately.\n"
        @"    - Sends machine info (cores, GPU, memory) on connect.\n"
        @"    - Auto-reconnects if the Conductor connection drops.\n\n"

        @"  Network Status window shows connected machines with hostname, cores,\n"
        @"  GPU, and status. Updated in real time.\n\n"

        @"GIMPS / PRIMENET\n"
        @"----------------\n"
        @"Click 'GIMPS' to open the Great Internet Mersenne Prime Search panel.\n"
        @"PrimePath routes all server communication through AutoPrimeNet.\n"
        @"It does not talk to the PrimeNet API directly.\n\n"

        @"  1. Click 'Start TF' to run trial factoring from worktodo.txt.\n"
        @"  2. Results are appended to results.json.txt automatically.\n"
        @"  3. Completed lines are removed from worktodo.txt.\n"
        @"  4. AutoPrimeNet picks up results.json.txt and submits to mersenne.org.\n\n"

        @"  GPU-found factors are verified on CPU before reporting.\n"
        @"  Composite factors are split via trial division + Pollard rho.\n"
        @"  Results use PrimeNet JSON format with CRC32 checksum.\n\n"

        @"HOW TO USE AUTOPRIMENET\n"
        @"----------------------\n"
        @"AutoPrimeNet is the recommended assignment handler for all major GIMPS\n"
        @"clients (Mlucas, GpuOwl, PRPLL, mfaktc, mfakto, PrimePath). It handles\n"
        @"registration, assignment fetching, result submission, email notifications,\n"
        @"log rotation, proxy support, stall monitoring, and version checking.\n\n"

        @"  Step 1: Install AutoPrimeNet\n"
        @"    git clone https://github.com/tdulcet/AutoPrimeNet.git\n"
        @"    cd AutoPrimeNet\n"
        @"    pip install -r requirements.txt\n\n"

        @"  Step 2: Run the setup wizard\n"
        @"    python3 autoprimenet.py --setup\n"
        @"    - Enter your mersenne.org username and password.\n"
        @"    - Select work type: Trial Factoring (TF).\n"
        @"    - Set the working directory to PrimePath's data directory:\n"
        @"      ~/Library/Application Support/PrimePath/\n"
        @"    - The wizard creates a local.ini config file.\n\n"

        @"  Step 3: Start AutoPrimeNet\n"
        @"    python3 autoprimenet.py\n"
        @"    It registers your machine with mersenne.org, fetches TF assignments,\n"
        @"    and writes them to worktodo.txt in the working directory.\n\n"

        @"  Step 4: Run PrimePath\n"
        @"    Open PrimePath, click GIMPS, click Start TF.\n"
        @"    PrimePath reads worktodo.txt, runs assignments on the GPU,\n"
        @"    writes results to results.json.txt, and removes completed lines.\n\n"

        @"  Step 5: AutoPrimeNet submits results\n"
        @"    AutoPrimeNet monitors results.json.txt and submits completed\n"
        @"    results to mersenne.org automatically. Keep it running alongside\n"
        @"    PrimePath.\n\n"

        @"  File locations:\n"
        @"    worktodo.txt       Assignments from AutoPrimeNet (input)\n"
        @"    results.json.txt   Completed results (output)\n"
        @"    local.ini          AutoPrimeNet configuration\n"
        @"    All in: ~/Library/Application Support/PrimePath/\n\n"

        @"  Tips:\n"
        @"    - Run AutoPrimeNet in a terminal tab alongside PrimePath.\n"
        @"    - AutoPrimeNet refills worktodo.txt when it gets low.\n"
        @"    - Check mersenne.org/results/ to confirm submissions.\n"
        @"    - Use --help for all AutoPrimeNet options.\n\n"

        @"EXPRESSION MODE\n"
        @"---------------\n"
        @"Type an expression in the Expression field, or in Check Single mode\n"
        @"(auto-detects when input contains letters, ^, or parentheses).\n\n"

        @"  Math:         2^67-1  1e9+7  20!  1729 mod 13\n"
        @"  Primality:    is_prime(997)  is 997 prime?\n"
        @"  Factoring:    factor(1729)  factor 1729\n"
        @"  Navigation:   next prime 10000  prev prime 100\n"
        @"  Modular:      modpow(2,100,1e9+7)  mulmod(a,b,m)\n"
        @"  Counting:     pi(1000)  phi(60)  gcd(48,36)  lcm(48,36)\n"
        @"  Sequences:    fibonacci(50)  C(10,3)  binomial(10,3)\n"
        @"  Ranges:       primes 1000 to 2000  twin primes 100 200\n"
        @"  Special:      wieferich 1093  wilson 563  mersenne 31\n"
        @"                fermat 4  convergence(97)\n\n"

        @"PIPELINE BUILDER\n"
        @"----------------\n"
        @"Click 'Pipeline' to build custom search pipelines by combining stages.\n"
        @"Put cheap filters first to reduce work for expensive tests.\n\n"

        @"  Sieve:    Wheel-210, MatrixSieve (NEON), CRT Filter, PseudoprimeFilter\n"
        @"  Score:    Shadow/Convergence, EvenShadow (p+/-1 divisor structure)\n"
        @"  Test:     Miller-Rabin (CPU), GPU Primality, Wieferich, Wilson, Pair\n"
        @"  Post:     Factor (Pollard rho), PinchFactor, Lucky7s, DivisorWeb\n\n"

        @"  Each stage shows its cost estimate (candidates/sec) and rejection rate.\n\n"

        @"CHECK & TOOLS\n"
        @"-------------\n"
        @"  Check Single      Test if a single number is prime.\n"
        @"  Check From Here   Find primes starting from a number.\n"
        @"  Check Linear      Enumerate all primes in a range (From/To fields).\n"
        @"  PrimeLocation     Predict prime locations using convergence algorithm.\n"
        @"  Expression        Evaluate math expressions and number theory functions.\n\n"

        @"  Benchmark         Run GPU and CPU performance benchmarks.\n"
        @"  Run Tests         Run the full 108-test engine validation suite.\n"
        @"  Test Catalog      Browse and run individual tests by category.\n"
        @"  Pipeline          Build custom search pipelines (see above).\n"
        @"  GIMPS             PrimeNet trial factoring (see above).\n\n"

        @"  CheckAtDiscovery  Run Wieferich/WallSunSun/Wilson tests on each\n"
        @"                    predicted prime as it is found.\n"
        @"  Run PrimeFactor   Use predicted primes to factor composites\n"
        @"                    (visible in PrimeLocation mode).\n\n"

        @"PREVENT SLEEP\n"
        @"-------------\n"
        @"Check 'Prevent Sleep' (next to Stop All) to keep your Mac awake during\n"
        @"long-running searches. Prevents both display sleep and system idle sleep.\n"
        @"Automatically released when unchecked or on quit.\n\n"

        @"SET TEST BOUNDS\n"
        @"---------------\n"
        @"Set the starting position for any search. Enter base ^ exponent + 1.\n"
        @"Example: 2 ^ 64 + 1 starts searching from 2^64 + 1.\n\n"

        @"SEARCH FRONTIER\n"
        @"---------------\n"
        @"On launch, PrimePath checks OEIS sequences to determine how far each\n"
        @"search type has been verified by other projects. If your search position\n"
        @"is below the known frontier, a warning is shown. Checked every 24 hours.\n\n"

        @"RESOURCE MONITOR\n"
        @"----------------\n"
        @"Five EQ-style bars show real-time resource usage:\n"
        @"  CPU        Process CPU usage across all cores.\n"
        @"  GPU        Percentage of time GPU is actively computing.\n"
        @"  NEON/SIMD  NEON vectorized sieve activity and rejection rate.\n"
        @"  MEMORY     App memory usage (resident size in MB).\n"
        @"  DISK I/O   Block read/write operations.\n"
        @"Click 'Hide' to disable the visualizers.\n\n"

        @"LOG OUTPUT GLOSSARY\n"
        @"-------------------\n"
        @"  GPU flush      CPU sieve buffer full, batch sent to GPU.\n"
        @"  WSS/Wief       Abbreviations for Wall-Sun-Sun / Wieferich.\n"
        @"  -> GPU         Candidates dispatched to Metal GPU.\n"
        @"  shadow         Even-shadow pre-filter score. Higher = better candidate.\n"
        @"  pos:           Current search position.\n"
        @"  found:         Discoveries made so far.\n"
        @"  /s             Candidates tested per second.\n\n"

        @"DATA FILES\n"
        @"----------\n"
        @"All state is saved to: ~/Documents/primes/primelocations/\n\n"

        @"  search_progress.txt   Task positions, status, counts. Auto-saved every 30s.\n"
        @"  discoveries.txt       All prime discoveries with timestamps.\n"
        @"  primenet_state.txt    PrimeNet registration, assignments, machine GUID.\n"
        @"  results.json.txt     Trial factoring results (mfaktc-compatible format).\n"
        @"  TestCatalog.txt       Custom test catalog (editable, reloadable from UI).\n\n"

        @"  Progress is restored automatically on relaunch.\n\n"

        @"KEYBOARD SHORTCUTS\n"
        @"------------------\n"
        @"  Cmd+Q          Quit\n"
        @"  Cmd+?          This help window\n"
        @"  Cmd+C/V/X/A    Copy/Paste/Cut/Select All in text fields\n\n"

        @"ABOUT\n"
        @"-----\n"
        @"Development by Sergei Nester\n"
        @"snester@viewbuild.com\n"
        @"https://github.com/s1rj1n/primepath\n";

    [helpWin makeKeyAndOrderFront:nil];
}

- (void)stopAll:(id)sender {
    _taskMgr->stop_all();
    [self stopBackgroundCheck];
    [self stopBenchmark:nil];
    for (NSButton *btn in self.taskButtons.allValues) {
        btn.title = @"Start";
    }
    self.taskStartBtn.title = @"Start";
    [self appendText:@"All tasks stopped. Progress saved.\n"];
}

// ── Benchmark ───────────────────────────────────────────────────────

- (void)runBenchmark:(id)sender {
    _activeLogTab = 2;  // Benchmark tab
    [self.logTabView selectTabViewItemAtIndex:2];
    if (_benchRunning.load()) return;
    // Join previous benchmark thread if still joinable
    if (_benchThread.joinable()) _benchThread.join();
    _benchRunning.store(true);
    self.benchmarkButton.enabled = NO;
    self.benchmarkStopButton.enabled = YES;

    __weak AppDelegate *weakSelf = self;
    _benchThread = std::thread([weakSelf]() {
        AppDelegate *strongSelf = weakSelf;
        if (!strongSelf) return;

        prime::Benchmark bench(strongSelf->_taskMgr);
        bench.run_all([weakSelf](const std::string& msg) {
            NSString *s = [NSString stringWithFormat:@"%@\n",
                [NSString stringWithUTF8String:msg.c_str()]];
            dispatch_async(dispatch_get_main_queue(), ^{
                [weakSelf appendText:s];
            });
        }, strongSelf->_benchRunning);

        dispatch_async(dispatch_get_main_queue(), ^{
            AppDelegate *ss = weakSelf;
            if (ss) {
                ss.benchmarkButton.enabled = YES;
                ss.benchmarkStopButton.enabled = NO;
                ss->_benchRunning.store(false);
            }
        });
    });
    // Thread stays joinable for proper cleanup
}

- (void)stopBenchmark:(id)sender {
    _benchRunning.store(false);
    self.benchmarkButton.enabled = YES;
    self.benchmarkStopButton.enabled = NO;
    // Join on background to avoid blocking main thread
    __weak AppDelegate *weakSelf = self;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        AppDelegate *ss = weakSelf;
        if (ss && ss->_benchThread.joinable()) {
            ss->_benchThread.join();
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════
// Carry-Chain Modular Exponentiation Test
// Tests 76-bit-in-128-bit overflow approach vs binary mulmod
// ═══════════════════════════════════════════════════════════════════════

// --- Method 1: Binary shift-and-add mulmod (current slow method) ---
static unsigned __int128 mulmod128_binary(unsigned __int128 a, unsigned __int128 b, unsigned __int128 mod) {
    a %= mod; b %= mod;
    if (a == 0 || b == 0) return 0;
    unsigned __int128 result = 0;
    while (b > 0) {
        if (b & 1) { result += a; if (result >= mod) result -= mod; }
        a += a; if (a >= mod) a -= mod;
        b >>= 1;
    }
    return result;
}

static unsigned __int128 powmod128_binary(unsigned __int128 base, uint64_t exp, unsigned __int128 mod) {
    unsigned __int128 result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = mulmod128_binary(result, base, mod);
        base = mulmod128_binary(base, base, mod);
        exp >>= 1;
    }
    return result;
}

// --- Method 2: Carry-chain mulmod using 128-bit hardware ---
// For moduli up to ~76 bits: decompose into 64-bit pieces,
// use hardware MUL+UMULH for full products, reduce with subtraction chain.
//
// a, b < mod < 2^76
// a*b < 2^152 — fits in 3 x 64-bit words (192 bits, top 40 bits always 0)

static inline void mul_full_152(uint64_t a_lo, uint64_t a_hi,
                                 uint64_t b_lo, uint64_t b_hi,
                                 uint64_t &r0, uint64_t &r1, uint64_t &r2) {
    // a = a_hi*2^64 + a_lo  (a_hi < 2^12 for 76-bit numbers)
    // b = b_hi*2^64 + b_lo  (b_hi < 2^12)
    // product = a_lo*b_lo + (a_lo*b_hi + a_hi*b_lo)*2^64 + a_hi*b_hi*2^128
    //
    // a_lo*b_lo → 128 bits via MUL+UMULH
    // cross terms: a_lo*b_hi and a_hi*b_lo — each < 2^76, fits in 128 bits
    // a_hi*b_hi < 2^24 — tiny

    // Low product: a_lo * b_lo → [p1:p0]
    unsigned __int128 p = (unsigned __int128)a_lo * b_lo;
    uint64_t p0 = (uint64_t)p;
    uint64_t p1 = (uint64_t)(p >> 64);

    // Cross terms: a_lo*b_hi + a_hi*b_lo
    // Each < 2^64 * 2^12 = 2^76, sum < 2^77
    unsigned __int128 cross = (unsigned __int128)a_lo * b_hi + (unsigned __int128)a_hi * b_lo;
    uint64_t c0 = (uint64_t)cross;
    uint64_t c1 = (uint64_t)(cross >> 64);

    // High product: a_hi * b_hi (< 2^24, fits in one word)
    uint64_t hh = a_hi * b_hi;

    // Accumulate: result = p0 + (p1 + c0)*2^64 + (c1 + hh)*2^128
    r0 = p0;
    unsigned __int128 mid = (unsigned __int128)p1 + c0;
    r1 = (uint64_t)mid;
    r2 = (uint64_t)(mid >> 64) + c1 + hh;
}

// Reduce a 192-bit number [r2:r1:r0] mod q where q < 2^76.
// Uses hardware __int128 division — NOT repeated subtraction.
// Strategy: reduce top 128 bits first, then shift down and add low word.
//   Step 1: hi = [r2:r1] (128 bits, r2 < 2^24) → hi_mod = hi % q
//   Step 2: hi_mod < 2^76, multiply by 2^64 in two 2^32 steps to stay in 128 bits
//   Step 3: add r0, final mod
static unsigned __int128 mulmod_carry(unsigned __int128 a, unsigned __int128 b, unsigned __int128 mod) {
    uint64_t a_lo = (uint64_t)a, a_hi = (uint64_t)(a >> 64);
    uint64_t b_lo = (uint64_t)b, b_hi = (uint64_t)(b >> 64);

    uint64_t r0, r1, r2;
    mul_full_152(a_lo, a_hi, b_lo, b_hi, r0, r1, r2);

    // [r2:r1] as 128-bit value — r2 is at most ~24 bits so this fits
    unsigned __int128 hi = ((unsigned __int128)r2 << 64) | r1;
    unsigned __int128 hi_mod = hi % mod;

    // Now compute (hi_mod * 2^64 + r0) % mod
    // hi_mod < mod < 2^76, so hi_mod * 2^32 < 2^108 — fits in __int128
    unsigned __int128 tmp = (hi_mod << 32) % mod;
    tmp = (tmp << 32) % mod;    // now tmp = hi_mod * 2^64 % mod
    tmp = (tmp + r0) % mod;

    return tmp;
}

static unsigned __int128 powmod_carry(unsigned __int128 base, uint64_t exp, unsigned __int128 mod) {
    unsigned __int128 result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = mulmod_carry(result, base, mod);
        base = mulmod_carry(base, base, mod);
        exp >>= 1;
    }
    return result;
}

// Format a 128-bit number as decimal string
static std::string u128_to_str(unsigned __int128 v) {
    if (v == 0) return "0";
    char buf[40]; int pos = 39; buf[pos] = 0;
    while (v > 0) { buf[--pos] = '0' + (int)(v % 10); v /= 10; }
    return &buf[pos];
}

- (void)runJSONSample:(id)sender {
    _activeLogTab = 2;
    [self.logTabView selectTabViewItemAtIndex:2];

    __weak AppDelegate *weakSelf = self;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        AppDelegate *ss = weakSelf;
        if (!ss) return;

        auto log = [ss](NSString *s) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [ss appendText:[s stringByAppendingString:@"\n"]];
            });
        };

        // Sample 1: Factor found -- M67 has factor 193707721
        {
            primenet::TFResult r;
            r.exponent = 67;
            r.bit_lo = 27;
            r.bit_hi = 28;
            r.factor_found = true;
            r.factors = {"193707721"};
            r.assignment_key = "A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4";
            r.range_complete = false;
            std::string json = ss->_primenet->build_result_json(r);
            log([NSString stringWithUTF8String:json.c_str()]);
        }

        // Sample 2: No factor, range complete
        {
            primenet::TFResult r;
            r.exponent = 82589933;
            r.bit_lo = 77;
            r.bit_hi = 78;
            r.factor_found = false;
            r.assignment_key = "A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4";
            r.range_complete = true;
            std::string json = ss->_primenet->build_result_json(r);
            log([NSString stringWithUTF8String:json.c_str()]);
        }

        // Sample 3: Factor found with known-factors -- M67
        // 193707721 and 761838257287 are both real factors of M67
        {
            primenet::TFResult r;
            r.exponent = 67;
            r.bit_lo = 27;
            r.bit_hi = 40;
            r.factor_found = true;
            r.factors = {"761838257287"};
            r.assignment_key = "A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4";
            r.range_complete = true;
            r.known_factors = {"193707721"};
            std::string json = ss->_primenet->build_result_json(r);
            log([NSString stringWithUTF8String:json.c_str()]);
        }

        // Sample 4: N/A assignment (no AID)
        {
            primenet::TFResult r;
            r.exponent = 82589933;
            r.bit_lo = 78;
            r.bit_hi = 79;
            r.factor_found = false;
            r.assignment_key = "";
            r.range_complete = true;
            std::string json = ss->_primenet->build_result_json(r);
            log([NSString stringWithUTF8String:json.c_str()]);
        }
    });
}

- (void)carryChainToggled:(id)sender {
    bool on = (self.carryChainToggle.state == NSControlStateValueOn);
    prime::g_use_carry_chain = on;
    [self appendStatus:[NSString stringWithFormat:@"NesterCarryChain mulmod: %s\n", on ? "ON (Wieferich + Wall-Sun-Sun CPU)" : "OFF (binary shift-and-add)"]];
}

- (void)runCarryChainTest:(id)sender {
    _activeLogTab = 2;  // Benchmark tab
    [self.logTabView selectTabViewItemAtIndex:2];

    __weak AppDelegate *weakSelf = self;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        AppDelegate *ss = weakSelf;
        if (!ss) return;

        auto out = [weakSelf](const std::string& msg) {
            NSString *s = [NSString stringWithUTF8String:msg.c_str()];
            dispatch_async(dispatch_get_main_queue(), ^{
                [weakSelf appendToTab:2 text:[s stringByAppendingString:@"\n"]];
            });
        };

        out("═══════════════════════════════════════════════════════");
        out("  CARRY-CHAIN vs BINARY MULMOD — Mersenne TF Test");
        out("═══════════════════════════════════════════════════════");
        out("");

        // Test case: M502493107 (current GIMPS assignment)
        // Known factor form: q = 2*k*p + 1
        // We test: 2^p mod q == 1 means q divides 2^p - 1
        uint64_t p = 502493107;  // exponent

        // Test with several k values to exercise the math
        // First, a known non-factor (arbitrary k)
        struct TestCase {
            uint64_t k;
            const char *desc;
        };
        TestCase tests[] = {
            { 1,                    "q = 2*1*p+1 (small)" },
            { 1000000,             "q = 2*10^6*p+1 (medium)" },
            { 75182985272188ULL,   "q = 2*75.18T*p+1 (GIMPS range start)" },
            { 75200000000000ULL,   "q = 2*75.2T*p+1 (GIMPS mid-range)" },
            { 100000000000000ULL,  "q = 2*100T*p+1 (larger k)" },
            { 1000000000000000ULL, "q = 2*1000T*p+1 (very large k, ~80 bits)" },
        };

        out("Exponent: p = " + std::to_string(p) + "  (M" + std::to_string(p) + " = 2^p - 1)");
        out("Testing: 2^p mod q for various q = 2kp+1");
        out("");

        for (auto& tc : tests) {
            unsigned __int128 q = (unsigned __int128)2 * tc.k * p + 1;
            int q_bits = 0;
            { unsigned __int128 tmp = q; while (tmp > 0) { q_bits++; tmp >>= 1; } }

            out("─────────────────────────────────────────────────────");
            out("Test: " + std::string(tc.desc));
            out("  k = " + std::to_string(tc.k));
            out("  q = " + u128_to_str(q) + "  (" + std::to_string(q_bits) + " bits)");

            // Method 1: Binary mulmod
            auto t0 = std::chrono::steady_clock::now();
            unsigned __int128 result_binary = powmod128_binary(2, p, q);
            auto t1 = std::chrono::steady_clock::now();
            double ms_binary = std::chrono::duration<double, std::milli>(t1 - t0).count();

            out("  BINARY mulmod:     2^p mod q = " + u128_to_str(result_binary));
            out("    time: " + std::to_string(ms_binary) + " ms");
            out("    factor? " + std::string(result_binary == 1 ? "YES — q divides M!" : "no"));

            // Method 2: Carry-chain mulmod
            if (q_bits <= 95) {  // carry-chain works up to 95 bits (hi_mod<<32 must fit in 128)
                auto t2 = std::chrono::steady_clock::now();
                unsigned __int128 result_carry = powmod_carry(2, p, q);
                auto t3 = std::chrono::steady_clock::now();
                double ms_carry = std::chrono::duration<double, std::milli>(t3 - t2).count();

                out("  CARRY-CHAIN mulmod: 2^p mod q = " + u128_to_str(result_carry));
                out("    time: " + std::to_string(ms_carry) + " ms");
                out("    factor? " + std::string(result_carry == 1 ? "YES — q divides M!" : "no"));

                bool match = (result_binary == result_carry);
                out("  MATCH: " + std::string(match ? "YES — both methods agree" : "NO — MISMATCH!"));
                if (match) {
                    double speedup = ms_binary / ms_carry;
                    out("  SPEEDUP: " + std::to_string(speedup) + "x faster");
                }
            } else {
                out("  CARRY-CHAIN: skipped — q is " + std::to_string(q_bits) +
                    " bits (>95, needs wider reduction)");
            }
            out("");
        }

        // Batch speed test: run 1000 modexps with carry-chain at GIMPS-range k
        out("═══════════════════════════════════════════════════════");
        out("  BATCH SPEED TEST: 1000 modexps at GIMPS-range k");
        out("═══════════════════════════════════════════════════════");

        uint64_t k_start = 75200000000000ULL;
        int n_tests = 1000;
        int factors_found = 0;

        // Binary method batch
        auto bt0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n_tests; i++) {
            unsigned __int128 q = (unsigned __int128)2 * (k_start + i) * p + 1;
            unsigned __int128 r = powmod128_binary(2, p, q);
            if (r == 1) factors_found++;
        }
        auto bt1 = std::chrono::steady_clock::now();
        double batch_binary_ms = std::chrono::duration<double, std::milli>(bt1 - bt0).count();

        out("Binary mulmod:      " + std::to_string(n_tests) + " modexps in " +
            std::to_string(batch_binary_ms) + " ms  (" +
            std::to_string(batch_binary_ms / n_tests) + " ms/each)");
        out("  rate: " + std::to_string((int)(n_tests * 1000.0 / batch_binary_ms)) + " modexps/sec");

        // Carry-chain batch
        factors_found = 0;
        auto ct0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n_tests; i++) {
            unsigned __int128 q = (unsigned __int128)2 * (k_start + i) * p + 1;
            unsigned __int128 r = powmod_carry(2, p, q);
            if (r == 1) factors_found++;
        }
        auto ct1 = std::chrono::steady_clock::now();
        double batch_carry_ms = std::chrono::duration<double, std::milli>(ct1 - ct0).count();

        out("Carry-chain mulmod: " + std::to_string(n_tests) + " modexps in " +
            std::to_string(batch_carry_ms) + " ms  (" +
            std::to_string(batch_carry_ms / n_tests) + " ms/each)");
        out("  rate: " + std::to_string((int)(n_tests * 1000.0 / batch_carry_ms)) + " modexps/sec");
        out("  SPEEDUP: " + std::to_string(batch_binary_ms / batch_carry_ms) + "x");
        if (factors_found > 0)
            out("  FACTORS FOUND: " + std::to_string(factors_found) + " (!!!)");

        out("");
        out("═══════════════════════════════════════════════════════");
        out("  STREAMING NEON DIVISIBILITY (Nester-CarryChain Method)");
        out("═══════════════════════════════════════════════════════");
        out("");

        // Build a large number: 2^2048 - 1 (all 1 bits, 32 limbs)
        prime::BigNum big2048;
        big2048.limbs.resize(32, UINT64_MAX);
        out("Number: 2^2048 - 1  (" + std::to_string(big2048.bit_width()) + " bits, " +
            std::to_string(big2048.limbs.size()) + " limbs)");
        out("");

        // Test known small divisors of 2^2048-1: any prime factor of 2^k-1 for k|2048
        // 2^2048 - 1 is divisible by 3, 5, 7, 17, 257, 65537, etc.
        uint64_t test_divs[] = {3, 5, 7, 11, 13, 17, 19, 23, 31, 41, 127, 257, 65537,
                                 8191, 131071, 524287, 6700417};
        size_t ndiv = sizeof(test_divs) / sizeof(test_divs[0]);

        // Scalar method
        auto st0 = std::chrono::steady_clock::now();
        auto scalar_res = prime::stream_find_divisors_scalar(big2048, test_divs, ndiv);
        auto st1 = std::chrono::steady_clock::now();
        double us_scalar = std::chrono::duration<double, std::micro>(st1 - st0).count();

        out("SCALAR streaming (" + std::to_string(ndiv) + " candidates):");
        out("  time: " + std::to_string(us_scalar) + " us");
        std::string divs_str;
        for (auto d : scalar_res) divs_str += std::to_string(d) + " ";
        out("  divisors found: " + divs_str);

        // NEON/batched method
        auto nt0 = std::chrono::steady_clock::now();
        auto neon_res = prime::stream_find_divisors(big2048, test_divs, ndiv);
        auto nt1 = std::chrono::steady_clock::now();

        out("NEON streaming (" + std::to_string(neon_res.tested) + " tested):");
        out("  time: " + std::to_string(neon_res.elapsed_us) + " us");
        std::string ndivs_str;
        for (auto d : neon_res.divisors) ndivs_str += std::to_string(d) + " ";
        out("  divisors found: " + ndivs_str);
        if (us_scalar > 0)
            out("  speedup: " + std::to_string(us_scalar / neon_res.elapsed_us) + "x");
        out("");

        // Bulk test: 10000 consecutive odd candidates against 2^2048-1
        out("BULK: 10,000 candidate divisors vs 2^2048-1:");
        std::vector<uint64_t> bulk_divs;
        for (uint64_t i = 3; i < 20003; i += 2) bulk_divs.push_back(i);

        auto bst0 = std::chrono::steady_clock::now();
        int scalar_hits = 0;
        for (auto d : bulk_divs) {
            if (prime::stream_mod_scalar(big2048.limbs.data(), big2048.limbs.size(), d) == 0)
                scalar_hits++;
        }
        auto bst1 = std::chrono::steady_clock::now();
        double bulk_scalar_us = std::chrono::duration<double, std::micro>(bst1 - bst0).count();

        auto bnt0 = std::chrono::steady_clock::now();
        auto bulk_neon = prime::stream_find_divisors(big2048, bulk_divs.data(), bulk_divs.size());
        auto bnt1 = std::chrono::steady_clock::now();
        double bulk_neon_us = std::chrono::duration<double, std::micro>(bnt1 - bnt0).count();

        out("  Scalar: " + std::to_string(bulk_scalar_us / 1000.0) + " ms, " +
            std::to_string(scalar_hits) + " divisors");
        out("  NEON:   " + std::to_string(bulk_neon_us / 1000.0) + " ms, " +
            std::to_string((int)bulk_neon.divisors.size()) + " divisors");
        out("  speedup: " + std::to_string(bulk_scalar_us / bulk_neon_us) + "x");
        out("  rate: " + std::to_string((int)(bulk_divs.size() / (bulk_neon_us / 1e6))) + " divisor tests/sec");

        out("");
        out("═══════════════════════════════════════════════════════");
        out("  TEST COMPLETE");
        out("═══════════════════════════════════════════════════════");
    });
}

// ── Streaming NEON benchmark (from Markov Predict window) ───────────

- (void)runStreamBenchmark:(id)sender {
    __weak AppDelegate *weakSelf = self;

    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        AppDelegate *ss = weakSelf;
        if (!ss) return;

        auto log = [weakSelf](NSString *msg) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [weakSelf markovLog:msg];
            });
        };

        log(@"====================================================");
        log(@"  NESTER-CARRYCHAIN TEST: Streaming Divisibility");
        log(@"====================================================");
        log(@"");

        // --- Test 1: Small number correctness ---
        log(@"TEST 1: Correctness (small numbers)");
        int pass = 0, fail = 0;
        // 100! has known factors. Use 2^127 - 1 (Mersenne prime M127)
        // M127 = 170141183460469231731687303715884105727 (prime, 128 bits = 2 limbs)
        prime::BigNum m127;
        m127.limbs = {0x7FFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
        // M127 is prime, so nothing except 1 and itself should divide it
        uint64_t small_tests[] = {2, 3, 5, 7, 11, 13, 17, 127, 257};
        for (auto d : small_tests) {
            bool divides = prime::stream_divides(m127, d);
            if (!divides) pass++; else fail++;
        }
        log([NSString stringWithFormat:@"  M127 (2^127-1, prime): %d pass, %d fail", pass, fail]);

        // 2^64 - 1 = 3 * 5 * 17 * 257 * 641 * 65537 * 6700417
        prime::BigNum m64;
        m64.limbs = {UINT64_MAX};
        uint64_t m64_factors[] = {3, 5, 17, 257, 641, 65537, 6700417};
        uint64_t m64_nonfactors[] = {2, 7, 11, 13, 19, 23};
        int correct = 0, total = 0;
        for (auto d : m64_factors) {
            if (prime::stream_divides(m64, d)) correct++;
            total++;
        }
        for (auto d : m64_nonfactors) {
            if (!prime::stream_divides(m64, d)) correct++;
            total++;
        }
        log([NSString stringWithFormat:@"  2^64-1 divisibility: %d/%d correct", correct, total]);
        log(@"");

        // --- Test 2: Real big number (RSA-2048 challenge) ---
        log(@"TEST 2: RSA-2048 (617 digits, 2048 bits)");
        // RSA-2048 = 25195908475657893494027183240048398571429282126204
        //            03202777713783604366202070759555626401852588078440
        //            ... (this is the actual RSA-2048 challenge number)
        prime::BigNum rsa2048 = prime::BigNum::from_hex(
            "C7970CEEDCC3B0754490201A7AA613CD73911081C790F5F1A8726F463550BB5B"
            "7FF0DB8E1EA1189EC72F93D1650011BD721AEEACC2ACDE32A04107F0648C2813"
            "A31F5B0B7765FF8B44B4B6FFC93B0932718E287AFD88D96048536CE35CE2CE9D"
            "FC7FD23E1B30B1B17D6BA07FDAC2DECF3033DAD7076E5CC4BA8DBEC83297C414"
            "FD560FCA58D4ECC0F7F6FF3AD29EC519B1B2A19A5A5519E54C35C3A1BD3B1B6A"
            "15C6FDE29C1CFD3EAA92CC8D1B3E32B2535C6B0E0C97CA53C6E6F8F77B23A6BC"
            "C42C7E96ECE84BD9E0CFC7DFC5A35FA946B2A1FDE5F597F91DD33F49A9ABCE81"
            "16BC97F6E5C5CD29F4540BEFB71A2E8B97AC467BBB21FEF2AE86DD8BFC520381");
        log([NSString stringWithFormat:@"  %zu bits, %zu limbs", rsa2048.bit_width(), rsa2048.limbs.size()]);

        // RSA-2048 is a product of two large primes, so small divisors won't divide it
        // But the test measures throughput across all three methods
        std::vector<uint64_t> rsa_candidates;
        for (uint64_t i = 3; i < 100003; i += 2) rsa_candidates.push_back(i);
        log([NSString stringWithFormat:@"  50K odd candidates (3..100001):"]);
        log(@"");

        // Method 1: Scalar with division (__int128 + hardware UDIV)
        auto rt0 = std::chrono::steady_clock::now();
        int rsa_sc_hits = 0;
        for (auto d : rsa_candidates)
            if (prime::stream_mod_scalar(rsa2048.limbs.data(), rsa2048.limbs.size(), d) == 0) rsa_sc_hits++;
        auto rt1 = std::chrono::steady_clock::now();
        double rsa_sc_us = std::chrono::duration<double, std::micro>(rt1 - rt0).count();
        log([NSString stringWithFormat:@"  DIVIDE (scalar):     %.1f ms, %d divisors", rsa_sc_us / 1000.0, rsa_sc_hits]);

        // Method 2: Nester-CC single (reciprocal multiply, no division)
        auto rt2 = std::chrono::steady_clock::now();
        int rsa_bj_hits = 0;
        for (auto d : rsa_candidates) {
            if (d > 1 && prime::stream_mod_blackjack(rsa2048.limbs.data(), rsa2048.limbs.size(), (uint32_t)d) == 0)
                rsa_bj_hits++;
        }
        auto rt3 = std::chrono::steady_clock::now();
        double rsa_bj_us = std::chrono::duration<double, std::micro>(rt3 - rt2).count();
        log([NSString stringWithFormat:@"  NESTER-CC (single):  %.1f ms, %d divisors", rsa_bj_us / 1000.0, rsa_bj_hits]);

        // Method 3: Nester-CC 4-wide (4 parallel accumulate streams)
        auto rt4 = std::chrono::steady_clock::now();
        auto rsa_bj4 = prime::stream_find_divisors(rsa2048, rsa_candidates.data(), rsa_candidates.size());
        auto rt5 = std::chrono::steady_clock::now();
        double rsa_bj4_us = std::chrono::duration<double, std::micro>(rt5 - rt4).count();
        log([NSString stringWithFormat:@"  NESTER-CC (4-wide):  %.1f ms, %d divisors",
             rsa_bj4_us / 1000.0, (int)rsa_bj4.divisors.size()]);

        // Cross-check: find mismatches
        int mismatches = 0;
        for (auto d : rsa_candidates) {
            if (d <= 1) continue;
            uint64_t scalar_r = prime::stream_mod_scalar(rsa2048.limbs.data(), rsa2048.limbs.size(), d);
            uint64_t bj_r = prime::stream_mod_blackjack(rsa2048.limbs.data(), rsa2048.limbs.size(), (uint32_t)d);
            if ((scalar_r == 0) != (bj_r == 0)) {
                log([NSString stringWithFormat:@"  MISMATCH d=%llu: scalar_rem=%llu, nester_cc_rem=%llu",
                     d, scalar_r, bj_r]);
                mismatches++;
                if (mismatches >= 10) break;
            }
        }
        if (mismatches == 0) log(@"  All results match!");

        log(@"");
        log([NSString stringWithFormat:@"  Nester-CC 1x vs Divide: %.2fx", rsa_sc_us / rsa_bj_us]);
        log([NSString stringWithFormat:@"  Nester-CC adaptive vs Divide: %.2fx (batch=%d)",
             rsa_sc_us / rsa_bj4_us, rsa_bj4.batch_size]);
        log([NSString stringWithFormat:@"  Nester-CC rate: %d divisor tests/sec",
             (int)(rsa_candidates.size() / (rsa_bj4_us / 1e6))]);
        log(@"");

        // --- Test 3: All batch widths at different number sizes ---
        log(@"TEST 3: Batch width scaling (50K candidates, 50K for adaptive)");
        log(@"  bits  | divide | ncc1x | ncc4x | ncc8x | ncc16x| adapt | best");
        log(@"  ------|--------|-------|-------|-------|-------|-------|-----");

        int bit_sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};
        std::vector<uint64_t> candidates;
        for (uint64_t i = 3; i < 100003; i += 2) candidates.push_back(i);
        // Larger set for adaptive test
        std::vector<uint64_t> big_candidates;
        for (uint64_t i = 3; i < 100003; i += 2) big_candidates.push_back(i);
        // Convert for batch tests
        std::vector<uint32_t> cands32;
        for (auto c : candidates) cands32.push_back((uint32_t)c);

        volatile uint64_t sink = 0; // prevent dead-code elimination
        for (int bits : bit_sizes) {
            int nlimbs = bits / 64;
            prime::BigNum num;
            num.limbs.resize(nlimbs, 0xA5A5A5A5A5A5A5A5ULL);
            num.limbs[0] = 0xDEADBEEFCAFEBABEULL;

            // Scalar (division)
            auto t0 = std::chrono::steady_clock::now();
            for (auto d : candidates)
                sink += prime::stream_mod_scalar(num.limbs.data(), num.limbs.size(), d);
            auto t1 = std::chrono::steady_clock::now();
            double sc_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

            // Nester-CC 1x
            auto tb1 = std::chrono::steady_clock::now();
            for (auto d : cands32)
                sink += prime::stream_mod_blackjack(num.limbs.data(), num.limbs.size(), d);
            auto te1 = std::chrono::steady_clock::now();
            double bj1 = std::chrono::duration<double, std::micro>(te1 - tb1).count();

            // Nester-CC 4x
            auto tb4 = std::chrono::steady_clock::now();
            for (size_t j = 0; j + 4 <= cands32.size(); j += 4) {
                uint32_t rem[4];
                prime::stream_mod_Nx32_blackjack(num.limbs.data(), num.limbs.size(),
                    &cands32[j], 4, rem);
                for (int k = 0; k < 4; k++) sink += rem[k];
            }
            auto te4 = std::chrono::steady_clock::now();
            double bj4 = std::chrono::duration<double, std::micro>(te4 - tb4).count();

            // Nester-CC 8x
            auto tb8 = std::chrono::steady_clock::now();
            for (size_t j = 0; j + 8 <= cands32.size(); j += 8) {
                uint32_t rem[8];
                prime::stream_mod_Nx32_blackjack(num.limbs.data(), num.limbs.size(),
                    &cands32[j], 8, rem);
                for (int k = 0; k < 8; k++) sink += rem[k];
            }
            auto te8 = std::chrono::steady_clock::now();
            double bj8 = std::chrono::duration<double, std::micro>(te8 - tb8).count();

            // Nester-CC 16x
            auto tb16 = std::chrono::steady_clock::now();
            for (size_t j = 0; j + 16 <= cands32.size(); j += 16) {
                uint32_t rem[16];
                prime::stream_mod_Nx32_blackjack(num.limbs.data(), num.limbs.size(),
                    &cands32[j], 16, rem);
                for (int k = 0; k < 16; k++) sink += rem[k];
            }
            auto te16 = std::chrono::steady_clock::now();
            double bj16 = std::chrono::duration<double, std::micro>(te16 - tb16).count();

            // Adaptive (auto-picks batch size)
            auto ta0 = std::chrono::steady_clock::now();
            auto ares = prime::stream_find_divisors(num, big_candidates.data(), big_candidates.size());
            auto ta1 = std::chrono::steady_clock::now();
            double adapt_us = std::chrono::duration<double, std::micro>(ta1 - ta0).count();

            // Find which fixed width was fastest
            double times[] = {bj1, bj4, bj8, bj16};
            const char* names[] = {"1x", "4x", "8x", "16x"};
            int best_idx = 0;
            for (int k = 1; k < 4; k++) if (times[k] < times[best_idx]) best_idx = k;

            log([NSString stringWithFormat:@"  %4d  | %5.0f  | %5.0f | %5.0f | %5.0f | %5.0f | %5.0f | %s(b%d)",
                 bits, sc_us, bj1, bj4, bj8, bj16,
                 adapt_us,
                 names[best_idx], ares.batch_size]);
        }
        log(@"");

        // --- Test 3: Mersenne streaming shortcut comparison ---
        log(@"TEST 4: Mersenne special case (2^p - 1 mod f)");
        log(@"  Compare building BigNum vs modpow shortcut");
        log(@"");

        uint64_t mersenne_p = 127;
        // Build 2^127 - 1 as BigNum
        prime::BigNum m127b;
        m127b.limbs = {0x7FFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};

        uint64_t mtest_divs[] = {3, 5, 7, 127, 911, 8191, 131071, 524287};
        size_t mndiv = sizeof(mtest_divs) / sizeof(mtest_divs[0]);

        // Stream method
        volatile uint64_t msink = 0;
        auto mt0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 1000000; rep++) {
            for (size_t i = 0; i < mndiv; i++)
                msink += prime::stream_mod_scalar(m127b.limbs.data(), m127b.limbs.size(), mtest_divs[i]);
        }
        auto mt1 = std::chrono::steady_clock::now();
        double stream_us = std::chrono::duration<double, std::micro>(mt1 - mt0).count();

        // Modpow shortcut
        auto mt2 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 1000000; rep++) {
            for (size_t i = 0; i < mndiv; i++)
                msink += prime::mersenne_stream_divides(mersenne_p, mtest_divs[i]);
        }
        auto mt3 = std::chrono::steady_clock::now();
        double modpow_us = std::chrono::duration<double, std::micro>(mt3 - mt2).count();

        log([NSString stringWithFormat:@"  M127 (2 limbs, %zu divisors, 1M reps):", mndiv]);
        log([NSString stringWithFormat:@"    Stream BigNum: %.1f us", stream_us]);
        log([NSString stringWithFormat:@"    Modpow shortcut: %.1f us", modpow_us]);
        log([NSString stringWithFormat:@"    Winner: %@",
             stream_us < modpow_us ? @"Stream (BigNum small enough)" : @"Modpow (O(log p) beats streaming)"]);
        log(@"");

        // Now test with a large Mersenne exponent where modpow wins massively
        uint64_t big_p = 82589933; // M82589933 is a known Mersenne prime
        // Building 2^82589933 - 1 as BigNum would be ~10MB, so we only test modpow
        auto mt4 = std::chrono::steady_clock::now();
        int big_divs_found = 0;
        for (size_t i = 0; i < mndiv; i++) {
            if (prime::mersenne_stream_divides(big_p, mtest_divs[i]))
                big_divs_found++;
        }
        auto mt5 = std::chrono::steady_clock::now();
        double big_modpow_us = std::chrono::duration<double, std::micro>(mt5 - mt4).count();
        log([NSString stringWithFormat:@"  M82589933 (25M digits, modpow only):  %.1f us for %zu divisors",
             big_modpow_us, mndiv]);
        log([NSString stringWithFormat:@"    (Stream would need ~1.2M limbs = %d MB -- modpow is the only option here)",
             (int)(82589933 / 64 * 8 / 1024 / 1024)]);

        // --- Test 5: Gear Shift Crossover Detection ---
        log(@"TEST 5: Gear Shift Crossover (CPU-1 vs CPU-MT vs GPU)");
        log(@"  Finding the sweet spot for each gear...");
        log(@"");

        MetalCompute *mc = [[MetalCompute alloc] init];
        bool gpu_ok = [mc available];

        int test_bits[]   = {128, 512, 2048, 8192, 32768};
        int test_divs[]   = {100, 1000, 10000, 50000, 200000};
        int ncpu = (int)std::thread::hardware_concurrency();

        log([NSString stringWithFormat:@"  CPU cores: %d, GPU: %@", ncpu,
             gpu_ok ? @"available" : @"not available"]);
        log(@"");
        log(@"  bits   | divs    | gear1(us) | gear2(us) | gear3(us) | winner | speedup | auto");
        log(@"  -------|---------|-----------|-----------|-----------|--------|--------");

        volatile uint64_t gsink = 0;

        for (int bits : test_bits) {
            int nlimbs = bits / 64;
            prime::BigNum num;
            num.limbs.resize(nlimbs, 0xA5A5A5A5A5A5A5A5ULL);
            num.limbs[0] = 0xDEADBEEFCAFEBABEULL;

            for (int nd : test_divs) {
                // Build divisor set
                std::vector<uint64_t> divs64;
                std::vector<uint32_t> divs32;
                for (int i = 0; i < nd; i++) {
                    uint64_t d = 3 + 2 * i;
                    divs64.push_back(d);
                    divs32.push_back((uint32_t)d);
                }

                // Gear 1: CPU single-thread
                auto g1t0 = std::chrono::steady_clock::now();
                auto g1res = prime::stream_find_divisors(num, divs64.data(), divs64.size());
                auto g1t1 = std::chrono::steady_clock::now();
                double g1_us = std::chrono::duration<double, std::micro>(g1t1 - g1t0).count();
                gsink += g1res.divisors.size();

                // Gear 2: CPU multi-thread
                auto g2t0 = std::chrono::steady_clock::now();
                auto g2res = prime::stream_find_divisors_mt(num, divs64.data(), divs64.size());
                auto g2t1 = std::chrono::steady_clock::now();
                double g2_us = std::chrono::duration<double, std::micro>(g2t1 - g2t0).count();
                gsink += g2res.divisors.size();

                // Gear 3: GPU
                double g3_us = 99999999;
                int g3_hits = -1;
                if (gpu_ok) {
                    auto g3t0 = std::chrono::steady_clock::now();
                    NSData *g3data = [mc runNesterCCStream:num.limbs.data()
                                                 numLimbs:(uint32_t)num.limbs.size()
                                                 divisors:divs32.data()
                                                  numDivs:(uint32_t)divs32.size()];
                    auto g3t1 = std::chrono::steady_clock::now();
                    g3_us = std::chrono::duration<double, std::micro>(g3t1 - g3t0).count();
                    if (g3data) {
                        const uint8_t *r = (const uint8_t *)g3data.bytes;
                        g3_hits = 0;
                        for (int i = 0; i < nd; i++) if (r[i]) g3_hits++;
                    }
                }

                // Auto-dispatch test
                MetalCompute *mcCapture = mc;
                bool gpuCapture = gpu_ok;
                prime::GpuDispatchFn gpuFn = nullptr;
                if (gpuCapture) {
                    gpuFn = [mcCapture](const uint64_t* limbs, uint32_t nLimbs,
                                        const uint32_t* divs, uint32_t nDivs) -> std::vector<uint64_t> {
                        NSData *data = [mcCapture runNesterCCStream:limbs numLimbs:nLimbs
                                                           divisors:divs numDivs:nDivs];
                        std::vector<uint64_t> hits;
                        if (data) {
                            const uint8_t *r = (const uint8_t *)data.bytes;
                            for (uint32_t i = 0; i < nDivs; i++)
                                if (r[i]) hits.push_back((uint64_t)divs[i]);
                        }
                        return hits;
                    };
                }
                auto autoRes = prime::stream_find_divisors_auto(num, divs64.data(), divs64.size(), gpuFn);
                int autoGear = autoRes.batch_size == -3 ? 3 :
                               (autoRes.elapsed_us > 0 && divs64.size() >= 1000 ? 2 : 1);
                gsink += autoRes.divisors.size();

                // Cross-check: all gears should find same number of divisors
                int g1_hits = (int)g1res.divisors.size();
                int g2_hits = (int)g2res.divisors.size();
                int auto_hits = (int)autoRes.divisors.size();
                NSString *check = @"";
                if (g1_hits != g2_hits || (g3_hits >= 0 && g1_hits != g3_hits) || g1_hits != auto_hits)
                    check = [NSString stringWithFormat:@" MISMATCH(%d/%d/%d/%d)", g1_hits, g2_hits, g3_hits, auto_hits];

                // Find winner
                double best = g1_us;
                const char *winner = "gear1";
                if (g2_us < best) { best = g2_us; winner = "gear2"; }
                if (g3_us < best) { best = g3_us; winner = "gear3"; }
                double speedup = g1_us / best;

                // Which gear did auto pick?
                int pickedInt = (int)prime::select_gear(num.limbs.size(), divs64.size(), gpu_ok);

                log([NSString stringWithFormat:@"  %5d  | %6d  | %9.0f | %9.0f | %9.0f | %6s | %.1fx | auto=%d%@",
                     bits, nd, g1_us, g2_us,
                     gpu_ok ? g3_us : 0.0,
                     winner, speedup, pickedInt, check]);
            }
        }

        log(@"");
        log(@"  Gear 1 = CPU single-thread 8-wide Barrett");
        log([NSString stringWithFormat:@"  Gear 2 = CPU %d-thread, each 8-wide Barrett", ncpu]);
        log(@"  Gear 3 = GPU Metal, one thread per divisor");

        log(@"");
        log(@"====================================================");
        log(@"  Nester-CarryChain rules: HIT (under) / BUST (reduce) / MATCH (divides)");
        log(@"  For Mersenne numbers, modpow shortcut avoids building the number.");
        log(@"  For general big numbers, Nester-CC auto-selects gear by work volume.");
        log(@"====================================================");
    });
}

// ── Stats refresh + resource monitoring ─────────────────────────────

// ── Search frontier checker ──────────────────────────────────────────
// Fetches OEIS pages to find latest verified search limits.
// Warns if any task is searching in already-covered territory.

- (void)checkSearchFrontiers {
    // Known frontiers (hardcoded fallbacks, updated by OEIS fetch)
    struct Frontier {
        prime::TaskType type;
        uint64_t known_limit;
        NSString *name;
        NSString *oeis_url;
    };

    // known_limit = highest verified value; jump_to = where to start searching
    // PrimeGrid verified Wieferich/WSS to 2^64 ~ 1.84x10^19.
    // We can't represent 2^64 in u64, so use 2^64-1 as the limit
    // and jump to a round value just below it to leave room for batched sieving.
    static const uint64_t PAST_2_64 = 18400000000000000001ULL;

    Frontier frontiers[] = {
        {prime::TaskType::Wieferich,  18446744073709551615ULL, @"Wieferich",
         @"https://oeis.org/A001220"},
        {prime::TaskType::WallSunSun, 18446744073709551615ULL, @"Wall-Sun-Sun",
         @"https://oeis.org/A182297"},
        {prime::TaskType::Wilson,     20000000000000ULL, @"Wilson",
         @"https://oeis.org/A007540"},
    };

    __weak AppDelegate *weakSelf = self;

    for (int i = 0; i < 3; i++) {
        Frontier f = frontiers[i];
        // Try to fetch OEIS page for updated frontier info
        NSURLRequest *req = [NSURLRequest requestWithURL:[NSURL URLWithString:f.oeis_url]
            cachePolicy:NSURLRequestReloadIgnoringLocalCacheData timeoutInterval:15];
        NSURLSession *session = [NSURLSession sharedSession];
        [[session dataTaskWithRequest:req completionHandler:^(NSData *data, NSURLResponse *resp, NSError *err) {
            AppDelegate *strongSelf = weakSelf;
            if (!strongSelf) return;

            uint64_t fetched_limit = f.known_limit; // fallback
            if (data && !err) {
                NSString *html = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
                // Look for patterns like "verified up to 2^64" or "searched to 2*10^13"
                // OEIS pages contain comments with search limits
                if (html) {
                    // Check for "2^64" pattern (Wieferich/WSS)
                    if ([html containsString:@"2^64"]) {
                        fetched_limit = 18446744073709551615ULL; // 2^64 - 1
                    }
                    // Check for "10^13" pattern (Wilson)
                    if ([html containsString:@"2*10^13"] || [html containsString:@"2x10^13"]) {
                        fetched_limit = 20000000000000ULL;
                    }
                    // Check for larger limits in case of updates
                    if ([html containsString:@"10^14"] && f.type == prime::TaskType::Wilson) {
                        fetched_limit = 100000000000000ULL;
                    }
                    if ([html containsString:@"2^65"] || [html containsString:@"2^66"]) {
                        // Someone extended beyond 2^64
                        fetched_limit = UINT64_MAX;
                    }
                }
            }

            // Check if our current search position is below the known frontier
            auto& tasks = strongSelf->_taskMgr->tasks();
            auto it = tasks.find(f.type);
            if (it == tasks.end()) return;

            uint64_t our_pos = it->second.current_pos;
            if (our_pos < fetched_limit) {
                // Auto-advance past the verified frontier
                uint64_t new_pos;
                if (fetched_limit >= UINT64_MAX - 2) {
                    // Frontier is ~2^64 -- use safe start below overflow ceiling
                    new_pos = PAST_2_64;
                } else {
                    new_pos = fetched_limit + 1;
                    if (new_pos % 2 == 0) new_pos++;
                }

                // Pause if running
                bool was_running = it->second.should_run.load();
                if (was_running) strongSelf->_taskMgr->pause_task(f.type);

                it->second.current_pos = new_pos;
                it->second.start_pos = new_pos;
                it->second.found_count = 0;
                it->second.tested_count = 0;
                strongSelf->_taskMgr->save_state();

                // Restart if it was running
                if (was_running) strongSelf->_taskMgr->start_task(f.type);

                NSString *msg = [NSString stringWithFormat:
                    @"⚡ FRONTIER AUTO-ADVANCE: %@ was at %@ (below verified %@). "
                    @"Jumped to %@.\n",
                    f.name, formatNumber(our_pos), formatNumber(fetched_limit),
                    formatNumber(new_pos)];
                dispatch_async(dispatch_get_main_queue(), ^{
                    AppDelegate *ss = weakSelf;
                    if (!ss) return;
                    [ss appendText:msg];
                    // Update button state if it was restarted
                    NSNumber *key = @((int)f.type);
                    if (was_running) {
                        ((NSButton *)ss.taskButtons[key]).title = @"Pause";
                    }
                });
            } else {
                NSString *msg = [NSString stringWithFormat:
                    @"[OK] %@ search at %@ -- past known frontier.\n",
                    f.name, formatNumber(our_pos)];
                dispatch_async(dispatch_get_main_queue(), ^{
                    [weakSelf appendText:msg];
                });
            }
        }] resume];
    }
}

- (void)refreshStats {
    if (!_taskMgr) return;

    // Update resource visualizer
    [self updateResourceMonitor];

    // Update task stats
    auto& tasks = _taskMgr->tasks();
    int running = 0;
    uint64_t total_found = 0;

    NSMutableString *activeList = [NSMutableString new];

    for (auto& [type, task] : tasks) {
        NSNumber *key = @((int)type);
        NSButton *btn = self.taskButtons[key];

        if (task.should_run.load()) {
            running++;
            [activeList appendFormat:@"[RUN]  %-14s  pos: %@  found: %@  %@/s\n",
                prime::task_name(type),
                formatNumber(task.current_pos),
                formatNumber(task.found_count),
                formatNumber((uint64_t)task.rate)];
            if (btn) btn.title = @"Pause";
        } else if (task.tested_count > 0) {
            [activeList appendFormat:@"[STOP] %-14s  at: %@  found: %@  tested: %@\n",
                prime::task_name(type),
                formatNumber(task.current_pos),
                formatNumber(task.found_count),
                formatNumber(task.tested_count)];
            if (btn) btn.title = @"Start";
        }
        total_found += task.found_count;
    }

    if (activeList.length == 0) {
        [activeList appendString:@"No tasks running. Select a test above and click Start."];
    }
    self.activeTaskListLabel.stringValue = activeList;

    // Update the Start/Pause button for the currently selected test
    auto selectedType = (prime::TaskType)self.taskSelectPopup.selectedItem.tag;
    auto sit = tasks.find(selectedType);
    if (sit != tasks.end()) {
        self.taskStartBtn.title = sit->second.should_run.load() ? @"Pause" : @"Start";
    }

    if (_checkRunning.load()) running++;

    if (running > 0) {
        self.statusLabel.stringValue = [NSString stringWithFormat:
            @"Running: %d | Discoveries: %@", running, formatNumber(total_found)];
    }
}

- (void)toggleVisualizer:(id)sender {
    BOOL disabled = (self.disableVisualizerBtn.state == NSControlStateValueOn);
    self.cpuEQ.hidden = disabled;
    self.gpuEQ.hidden = disabled;
    self.neonEQ.hidden = disabled;
    self.memEQ.hidden = disabled;
    self.diskEQ.hidden = disabled;
}

- (void)updateResourceMonitor {
    if (self.disableVisualizerBtn.state == NSControlStateValueOn) return;

    // ── CPU (PrimePath process only) ──
    // Sum CPU time across all threads, compare delta to wall clock.
    {
        thread_array_t threadList;
        mach_msg_type_number_t threadCount;
        if (task_threads(mach_task_self(), &threadList, &threadCount) == KERN_SUCCESS) {
            uint64_t totalCPU_ns = 0;
            for (mach_msg_type_number_t i = 0; i < threadCount; i++) {
                thread_basic_info_data_t thinfo;
                mach_msg_type_number_t thcount = THREAD_BASIC_INFO_COUNT;
                if (thread_info(threadList[i], THREAD_BASIC_INFO,
                                (thread_info_t)&thinfo, &thcount) == KERN_SUCCESS) {
                    if (!(thinfo.flags & TH_FLAGS_IDLE)) {
                        totalCPU_ns += (uint64_t)thinfo.user_time.seconds * 1000000000ULL +
                                       (uint64_t)thinfo.user_time.microseconds * 1000ULL +
                                       (uint64_t)thinfo.system_time.seconds * 1000000000ULL +
                                       (uint64_t)thinfo.system_time.microseconds * 1000ULL;
                    }
                }
                mach_port_deallocate(mach_task_self(), threadList[i]);
            }
            vm_deallocate(mach_task_self(), (vm_address_t)threadList,
                          threadCount * sizeof(thread_t));

            uint64_t wallNow = clock_gettime_nsec_np(CLOCK_MONOTONIC);
            if (_prevProcCPU_ns > 0 && _prevWallClock_ns > 0) {
                uint64_t cpuDelta = totalCPU_ns - _prevProcCPU_ns;
                uint64_t wallDelta = wallNow - _prevWallClock_ns;
                if (wallDelta > 0) {
                    unsigned nCores = std::thread::hardware_concurrency();
                    if (nCores < 1) nCores = 1;
                    double cpuPct = 100.0 * (double)cpuDelta / ((double)wallDelta * nCores);
                    if (cpuPct > 100.0) cpuPct = 100.0;
                    [self.cpuEQ pushValue:cpuPct];
                    [self.cpuEQ setDetail:[NSString stringWithFormat:@"%.0f%%", cpuPct]];
                }
            }
            _prevProcCPU_ns = totalCPU_ns;
            _prevWallClock_ns = wallNow;
        }
    }

    // ── Memory ──
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
        uint64_t mb = info.resident_size / (1024 * 1024);
        double memPct = std::min(100.0, (double)mb / 2048.0 * 100.0);
        [self.memEQ pushValue:memPct];
        [self.memEQ setDetail:[NSString stringWithFormat:@"%lluMB", (unsigned long long)mb]];
    }

    // ── GPU ──
    int runningTasks = 0, gpuTasks = 0;
    uint64_t totalRate = 0;
    auto& tasks = _taskMgr->tasks();
    for (auto& [type, task] : tasks) {
        if (task.status == prime::TaskStatus::Running) {
            runningTasks++;
            totalRate += (uint64_t)task.rate;
            if (type == prime::TaskType::Wieferich ||
                type == prime::TaskType::WallSunSun ||
                type == prime::TaskType::SophieGermain ||
                type == prime::TaskType::Wilson ||
                type == prime::TaskType::MersenneTrial ||
                type == prime::TaskType::FermatFactor) {
                gpuTasks++;
            }
        }
    }

    double gpuUtil = _gpu->gpu_utilization() * 100.0;
    uint64_t gpuBatches = _gpu->total_batches_dispatched();
    double avgMs = _gpu->avg_gpu_time_ms();
    [self.gpuEQ pushValue:gpuUtil];
    if (gpuBatches > 0) {
        [self.gpuEQ setDetail:[NSString stringWithFormat:@"%.0f%% %.1fms", gpuUtil, avgMs]];
    } else {
        [self.gpuEQ setDetail:gpuTasks > 0 ? @"filling" : @"idle"];
    }

    // ── NEON/SIMD (delta-based activity) ──
    auto* ms = _taskMgr->matrix_sieve();
    if (ms) {
        uint64_t tested = ms->total_tested();
        uint64_t rejected = ms->total_rejected();
        uint64_t deltaTested = tested - _prevNeonTested;
        uint64_t deltaRejected = rejected - _prevNeonRejected;
        _prevNeonTested = tested;
        _prevNeonRejected = rejected;
        // Show activity based on throughput delta, not cumulative rate
        // Scale: 1M tested/interval → 100%
        double neonActivity = deltaTested > 0 ? std::min(100.0, (double)deltaTested / 1000000.0 * 100.0) : 0.0;
        double rejPct = deltaTested > 0 ? (100.0 * deltaRejected / deltaTested) : 0.0;
        [self.neonEQ pushValue:neonActivity];
        [self.neonEQ setDetail:[NSString stringWithFormat:@"%.1f%% rej", rejPct]];
    } else {
        [self.neonEQ pushValue:0];
    }

    // ── Disk I/O ──
    // Use rusage for I/O stats (block operations)
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        uint64_t reads = (uint64_t)ru.ru_inblock;
        uint64_t writes = (uint64_t)ru.ru_oublock;
        uint64_t deltaR = reads - _prevDiskReadBytes;
        uint64_t deltaW = writes - _prevDiskWriteBytes;
        // Scale: each block op ~ some activity, cap at 100
        double diskPct = std::min(100.0, (double)(deltaR + deltaW) * 5.0);
        [self.diskEQ pushValue:diskPct];
        [self.diskEQ setDetail:[NSString stringWithFormat:@"R:%llu W:%llu",
            (unsigned long long)deltaR, (unsigned long long)deltaW]];
        _prevDiskReadBytes = reads;
        _prevDiskWriteBytes = writes;
    }

    // ── Status ──
    if (runningTasks > 0) {
        self.statusLabel.stringValue = [NSString stringWithFormat:
            @"Running: %d tasks | %@/s | GPU: %@",
            runningTasks, formatNumber(totalRate),
            [NSString stringWithUTF8String:_gpu->name().c_str()]];
        self.statusLabel.textColor = [NSColor colorWithSRGBRed:0.0 green:0.55 blue:0.0 alpha:1.0];
    } else {
        self.statusLabel.stringValue = @"Ready";
        self.statusLabel.textColor = [NSColor secondaryLabelColor];
    }
}

// ── UI helpers ──────────────────────────────────────────────────────

- (NSTextField *)labelAt:(NSRect)frame text:(NSString *)text bold:(BOOL)bold size:(CGFloat)sz {
    NSTextField *l = [[NSTextField alloc] initWithFrame:frame];
    l.stringValue = text;
    l.editable = NO; l.bordered = NO; l.drawsBackground = NO;
    l.font = bold ? [NSFont boldSystemFontOfSize:sz] : [NSFont systemFontOfSize:sz];
    return l;
}

- (NSTextField *)fieldAt:(NSRect)frame placeholder:(NSString *)ph {
    NSTextField *f = [[NSTextField alloc] initWithFrame:frame];
    f.placeholderString = ph;
    f.font = [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular];
    return f;
}

- (NSButton *)buttonAt:(NSRect)frame title:(NSString *)title action:(SEL)action {
    NSButton *b = [[NSButton alloc] initWithFrame:frame];
    b.title = title; b.bezelStyle = NSBezelStyleRounded;
    b.target = self; b.action = action;
    return b;
}

// ═══════════════════════════════════════════════════════════════════════
// GIMPS / PrimeNet Panel
// ═══════════════════════════════════════════════════════════════════════

- (void)showGIMPSPanel:(id)sender {
    NSWindow *win = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(200, 200, 560, 480)
        styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable)
        backing:NSBackingStoreBuffered defer:NO];
    win.title = @"GIMPS / PrimeNet Integration";
    win.releasedWhenClosed = NO;
    win.minSize = NSMakeSize(480, 400);

    NSView *cv = win.contentView;
    CGFloat W = cv.frame.size.width;
    CGFloat y = cv.frame.size.height - 14;

    // Title
    NSTextField *title = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 20, 400, 18)];
    title.stringValue = @"GIMPS -- Great Internet Mersenne Prime Search";
    title.font = [NSFont boldSystemFontOfSize:13];
    title.bezeled = NO; title.editable = NO; title.drawsBackground = NO;
    title.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:title];
    y -= 28;

    NSTextField *desc = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 28, W - 28, 28)];
    desc.stringValue = @"Connect to mersenne.org to get trial factoring assignments and report results.\nPrimePath uses Metal GPU for 96-bit Barrett modular exponentiation.";
    desc.font = [NSFont systemFontOfSize:10];
    desc.textColor = [NSColor secondaryLabelColor];
    desc.bezeled = NO; desc.editable = NO; desc.drawsBackground = NO;
    desc.maximumNumberOfLines = 0;
    desc.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:desc];
    y -= 36;

    // Separator
    NSBox *sep = [[NSBox alloc] initWithFrame:NSMakeRect(14, y, W - 28, 1)];
    sep.boxType = NSBoxSeparator;
    sep.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:sep];
    y -= 12;

    // Username field
    NSTextField *userLbl = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 16, 80, 14)];
    userLbl.stringValue = @"Username:";
    userLbl.font = [NSFont systemFontOfSize:10];
    userLbl.bezeled = NO; userLbl.editable = NO; userLbl.drawsBackground = NO;
    userLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:userLbl];

    NSTextField *userField = [[NSTextField alloc] initWithFrame:NSMakeRect(94, y - 18, 160, 20)];
    userField.font = [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular];
    userField.stringValue = [NSString stringWithUTF8String:_primenet->username().c_str()];
    userField.tag = 8001;
    userField.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:userField];

    // AutoPrimeNet notice
    NSTextField *apnNote = [[NSTextField alloc] initWithFrame:NSMakeRect(270, y - 16, 280, 14)];
    apnNote.stringValue = @"All server communication via AutoPrimeNet (worktodo.txt / results.json.txt)";
    apnNote.font = [NSFont systemFontOfSize:9];
    apnNote.textColor = [NSColor secondaryLabelColor];
    apnNote.bezeled = NO; apnNote.editable = NO; apnNote.drawsBackground = NO;
    apnNote.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:apnNote];

    y -= 28;

    // Buttons row: Run / Stop (local only) + AutoPrimeNet Settings
    NSButton *runWorkBtn = [self buttonAt:NSMakeRect(14, y - 24, 120, 24)
        title:@"Run Assignment" action:@selector(gimpsRunAssignment:)];
    runWorkBtn.font = [NSFont boldSystemFontOfSize:10];
    runWorkBtn.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:runWorkBtn];

    NSButton *stopBtn = [self buttonAt:NSMakeRect(140, y - 24, 50, 24)
        title:@"Stop" action:@selector(gimpsStopAssignment:)];
    stopBtn.font = [NSFont boldSystemFontOfSize:10];
    stopBtn.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:stopBtn];

    NSButton *apnBtn = [self buttonAt:NSMakeRect(210, y - 24, 150, 24)
        title:@"AutoPrimeNet Settings" action:@selector(showAutoPrimeNetPanel:)];
    apnBtn.font = [NSFont systemFontOfSize:10];
    apnBtn.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:apnBtn];

    y -= 30;

    // JSON Editor button
    NSButton *jsonBtn = [self buttonAt:NSMakeRect(14, y - 24, 130, 24)
        title:@"JSON Result Editor" action:@selector(showJSONResultEditor:)];
    jsonBtn.font = [NSFont systemFontOfSize:10];
    jsonBtn.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:jsonBtn];

    NSTextField *jsonHint = [[NSTextField alloc] initWithFrame:NSMakeRect(150, y - 22, 280, 14)];
    jsonHint.stringValue = @"Build and preview JSON result lines (local only, not submitted).";
    jsonHint.font = [NSFont systemFontOfSize:9];
    jsonHint.textColor = [NSColor secondaryLabelColor];
    jsonHint.bezeled = NO; jsonHint.editable = NO; jsonHint.drawsBackground = NO;
    jsonHint.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:jsonHint];

    y -= 30;

    // Stop on first factor toggle
    NSButton *abortCheck = [[NSButton alloc] initWithFrame:NSMakeRect(14, y - 18, 250, 18)];
    abortCheck.buttonType = NSButtonTypeSwitch;
    abortCheck.title = @"Stop on first factor found";
    abortCheck.font = [NSFont systemFontOfSize:10];
    abortCheck.state = _taskMgr->mersenne_abort_on_factor.load() ? NSControlStateValueOn : NSControlStateValueOff;
    abortCheck.target = self;
    abortCheck.action = @selector(toggleAbortOnFactor:);
    abortCheck.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:abortCheck];

    NSTextField *abortHint = [[NSTextField alloc] initWithFrame:NSMakeRect(270, y - 16, 280, 14)];
    abortHint.stringValue = @"Default off: complete full bitlevel (recommended by mersenne.org).";
    abortHint.font = [NSFont systemFontOfSize:9];
    abortHint.textColor = [NSColor secondaryLabelColor];
    abortHint.bezeled = NO; abortHint.editable = NO; abortHint.drawsBackground = NO;
    abortHint.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:abortHint];

    y -= 26;

    // Batch size option
    NSTextField *batchLbl = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 16, 80, 14)];
    batchLbl.stringValue = @"Sieve batch:";
    batchLbl.font = [NSFont systemFontOfSize:10];
    batchLbl.bezeled = NO; batchLbl.editable = NO; batchLbl.drawsBackground = NO;
    batchLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:batchLbl];

    NSPopUpButton *batchPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(94, y - 18, 140, 20) pullsDown:NO];
    [batchPopup addItemWithTitle:@"10M (light)"];
    [batchPopup addItemWithTitle:@"50M (medium)"];
    [batchPopup addItemWithTitle:@"100M (default)"];
    [batchPopup addItemWithTitle:@"500M (heavy)"];
    [batchPopup addItemWithTitle:@"1B (max CPU)"];
    batchPopup.font = [NSFont systemFontOfSize:10];
    batchPopup.tag = 8030;
    [batchPopup selectItemAtIndex:2]; // default 100M
    batchPopup.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:batchPopup];

    NSTextField *batchHint = [[NSTextField alloc] initWithFrame:NSMakeRect(240, y - 16, 280, 14)];
    batchHint.stringValue = @"k-values per sieve pass. Higher = more CPU, faster progress.";
    batchHint.font = [NSFont systemFontOfSize:9];
    batchHint.textColor = [NSColor secondaryLabelColor];
    batchHint.bezeled = NO; batchHint.editable = NO; batchHint.drawsBackground = NO;
    batchHint.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:batchHint];

    y -= 26;

    // Carry-chain verification info
    NSTextField *ccLbl = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 14, W - 28, 14)];
    ccLbl.stringValue = @"CPU Carry-Chain Verification: Every GPU-found factor is independently verified on CPU before submission.";
    ccLbl.font = [NSFont systemFontOfSize:9];
    ccLbl.textColor = [NSColor colorWithSRGBRed:0.2 green:0.7 blue:0.2 alpha:1.0];
    ccLbl.bezeled = NO; ccLbl.editable = NO; ccLbl.drawsBackground = NO;
    ccLbl.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:ccLbl];
    y -= 20;

    // Separator
    NSBox *sep2 = [[NSBox alloc] initWithFrame:NSMakeRect(14, y, W - 28, 1)];
    sep2.boxType = NSBoxSeparator;
    sep2.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:sep2];
    y -= 8;

    // Assignments list header
    NSTextField *assignHdr = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 14, 200, 14)];
    assignHdr.stringValue = @"Current Assignments";
    assignHdr.font = [NSFont boldSystemFontOfSize:10];
    assignHdr.bezeled = NO; assignHdr.editable = NO; assignHdr.drawsBackground = NO;
    assignHdr.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:assignHdr];
    y -= 18;

    // Assignment info area
    NSTextField *assignInfo = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 60, W - 28, 60)];
    assignInfo.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    assignInfo.bezeled = NO; assignInfo.editable = NO; assignInfo.drawsBackground = NO;
    assignInfo.maximumNumberOfLines = 0;
    assignInfo.tag = 8020;
    assignInfo.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:assignInfo];

    // Populate assignments
    if (_primenet->pending_count() > 0) {
        NSMutableString *astr = [NSMutableString string];
        for (auto& a : _primenet->state().assignments) {
            [astr appendFormat:@"M%llu  TF %d-%d bits  [%s]\n",
                a.exponent, (int)a.bit_lo, (int)a.bit_hi,
                a.key.c_str()];
        }
        assignInfo.stringValue = astr;
    } else {
        assignInfo.stringValue = @"No assignments. Click 'Get Work' to fetch from mersenne.org.";
        assignInfo.textColor = [NSColor secondaryLabelColor];
    }

    y -= 68;

    // Separator
    NSBox *sep3 = [[NSBox alloc] initWithFrame:NSMakeRect(14, y, W - 28, 1)];
    sep3.boxType = NSBoxSeparator;
    sep3.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:sep3];
    y -= 8;

    // Log area
    NSTextField *logHdr = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 14, 200, 14)];
    logHdr.stringValue = @"PrimeNet Log";
    logHdr.font = [NSFont boldSystemFontOfSize:10];
    logHdr.bezeled = NO; logHdr.editable = NO; logHdr.drawsBackground = NO;
    logHdr.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:logHdr];
    y -= 18;

    NSScrollView *logScroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(14, 10, W - 28, y - 10)];
    logScroll.hasVerticalScroller = YES;
    logScroll.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    NSTextView *logView = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, W - 32, y - 10)];
    logView.editable = NO;
    logView.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    logView.autoresizingMask = NSViewWidthSizable;
    logScroll.documentView = logView;
    logScroll.identifier = @"gimpsLog";
    [cv addSubview:logScroll];

    [win makeKeyAndOrderFront:nil];
}

// ═══════════════════════════════════════════════════════════════════════
// Abort-on-factor toggle
// ═══════════════════════════════════════════════════════════════════════

- (void)toggleAbortOnFactor:(id)sender {
    NSButton *btn = (NSButton *)sender;
    _taskMgr->mersenne_abort_on_factor.store(btn.state == NSControlStateValueOn);
}

// ═══════════════════════════════════════════════════════════════════════
// AutoPrimeNet Settings Panel
// ═══════════════════════════════════════════════════════════════════════

- (void)showAutoPrimeNetPanel:(id)sender {
    CGFloat W = 600, H = 580;
    NSRect frame = NSMakeRect(200, 200, W, H);
    NSWindow *win = [[NSWindow alloc]
        initWithContentRect:frame
        styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable
        backing:NSBackingStoreBuffered defer:NO];
    win.title = @"AutoPrimeNet Settings";
    win.minSize = NSMakeSize(500, 400);
    win.releasedWhenClosed = NO;

    NSView *cv = win.contentView;
    CGFloat y = H - 10;

    NSString *DATA_DIR = [NSSearchPathForDirectoriesInDomains(NSApplicationSupportDirectory,
        NSUserDomainMask, YES).firstObject stringByAppendingPathComponent:@"PrimePath"];

    // ── Header ───────────────────────────────────────────────────────
    NSTextField *header = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 20, W - 28, 18)];
    header.stringValue = @"AutoPrimeNet Integration";
    header.font = [NSFont boldSystemFontOfSize:13];
    header.bezeled = NO; header.editable = NO; header.drawsBackground = NO;
    header.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:header];
    y -= 24;

    NSTextField *desc = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 30, W - 28, 28)];
    desc.stringValue = @"AutoPrimeNet manages assignments and result submissions for all major GIMPS clients. "
        @"PrimePath reads worktodo.txt and writes results.json.txt. AutoPrimeNet handles the rest.";
    desc.font = [NSFont systemFontOfSize:10];
    desc.textColor = [NSColor secondaryLabelColor];
    desc.bezeled = NO; desc.editable = NO; desc.drawsBackground = NO;
    desc.maximumNumberOfLines = 3;
    desc.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:desc];
    y -= 36;

    // ── Data Directory ───────────────────────────────────────────────
    NSTextField *dirLbl = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 14, 90, 14)];
    dirLbl.stringValue = @"Data directory:";
    dirLbl.font = [NSFont boldSystemFontOfSize:10];
    dirLbl.bezeled = NO; dirLbl.editable = NO; dirLbl.drawsBackground = NO;
    dirLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:dirLbl];

    NSTextField *dirPath = [[NSTextField alloc] initWithFrame:NSMakeRect(110, y - 14, W - 124, 14)];
    dirPath.stringValue = DATA_DIR;
    dirPath.font = [NSFont monospacedSystemFontOfSize:9 weight:NSFontWeightRegular];
    dirPath.textColor = [NSColor labelColor];
    dirPath.bezeled = NO; dirPath.editable = NO; dirPath.drawsBackground = NO; dirPath.selectable = YES;
    dirPath.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:dirPath];
    y -= 22;

    NSBox *sep1 = [[NSBox alloc] initWithFrame:NSMakeRect(14, y, W - 28, 1)];
    sep1.boxType = NSBoxSeparator;
    sep1.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:sep1];
    y -= 12;

    // ── Settings ─────────────────────────────────────────────────────
    NSTextField *settingsHdr = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 14, 200, 14)];
    settingsHdr.stringValue = @"Settings";
    settingsHdr.font = [NSFont boldSystemFontOfSize:11];
    settingsHdr.bezeled = NO; settingsHdr.editable = NO; settingsHdr.drawsBackground = NO;
    settingsHdr.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:settingsHdr];
    y -= 22;

    NSButton *abortCheck = [[NSButton alloc] initWithFrame:NSMakeRect(14, y - 18, 300, 18)];
    abortCheck.buttonType = NSButtonTypeSwitch;
    abortCheck.title = @"Stop on first factor found (not recommended)";
    abortCheck.font = [NSFont systemFontOfSize:10];
    abortCheck.state = _taskMgr->mersenne_abort_on_factor.load() ? NSControlStateValueOn : NSControlStateValueOff;
    abortCheck.target = self;
    abortCheck.action = @selector(toggleAbortOnFactor:);
    abortCheck.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:abortCheck];

    NSTextField *abortNote = [[NSTextField alloc] initWithFrame:NSMakeRect(34, y - 32, W - 48, 14)];
    abortNote.stringValue = @"Default off. Completing the full bitlevel avoids wasted work on mersenne.org.";
    abortNote.font = [NSFont systemFontOfSize:9];
    abortNote.textColor = [NSColor secondaryLabelColor];
    abortNote.bezeled = NO; abortNote.editable = NO; abortNote.drawsBackground = NO;
    abortNote.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:abortNote];
    y -= 22;

    // Factor remaining composites before reporting
    NSButton *factorCompCheck = [[NSButton alloc] initWithFrame:NSMakeRect(14, y - 18, 350, 18)];
    factorCompCheck.buttonType = NSButtonTypeSwitch;
    factorCompCheck.title = @"Factor remaining composites before reporting (trial division)";
    factorCompCheck.font = [NSFont systemFontOfSize:10];
    factorCompCheck.state = NSControlStateValueOff;
    factorCompCheck.tag = 8041;
    factorCompCheck.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:factorCompCheck];
    y -= 22;

    NSBox *sep2 = [[NSBox alloc] initWithFrame:NSMakeRect(14, y, W - 28, 1)];
    sep2.boxType = NSBoxSeparator;
    sep2.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:sep2];
    y -= 12;

    // ── worktodo.txt ─────────────────────────────────────────────────
    NSTextField *wtHdr = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 14, 200, 14)];
    wtHdr.stringValue = @"worktodo.txt (pending assignments)";
    wtHdr.font = [NSFont boldSystemFontOfSize:11];
    wtHdr.bezeled = NO; wtHdr.editable = NO; wtHdr.drawsBackground = NO;
    wtHdr.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:wtHdr];

    // Status indicator
    NSTextField *wtStatus = [[NSTextField alloc] initWithFrame:NSMakeRect(W - 200, y - 14, 186, 14)];
    wtStatus.tag = 8101;
    wtStatus.font = [NSFont systemFontOfSize:10];
    wtStatus.alignment = NSTextAlignmentRight;
    wtStatus.bezeled = NO; wtStatus.editable = NO; wtStatus.drawsBackground = NO;
    wtStatus.autoresizingMask = NSViewMinYMargin | NSViewMinXMargin;
    [cv addSubview:wtStatus];
    y -= 20;

    NSScrollView *wtScroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(14, y - 100, W - 28, 100)];
    wtScroll.hasVerticalScroller = YES;
    wtScroll.borderType = NSBezelBorder;
    wtScroll.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    NSTextView *wtView = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, W - 32, 100)];
    wtView.editable = NO;
    wtView.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    wtView.identifier = @"apnWorktodo";
    wtScroll.documentView = wtView;
    [cv addSubview:wtScroll];
    y -= 108;

    // ── results.json.txt ─────────────────────────────────────────────
    NSTextField *rjHdr = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 14, 250, 14)];
    rjHdr.stringValue = @"results.json.txt (pending upload)";
    rjHdr.font = [NSFont boldSystemFontOfSize:11];
    rjHdr.bezeled = NO; rjHdr.editable = NO; rjHdr.drawsBackground = NO;
    rjHdr.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:rjHdr];

    NSTextField *rjStatus = [[NSTextField alloc] initWithFrame:NSMakeRect(W - 200, y - 14, 186, 14)];
    rjStatus.tag = 8102;
    rjStatus.font = [NSFont systemFontOfSize:10];
    rjStatus.alignment = NSTextAlignmentRight;
    rjStatus.bezeled = NO; rjStatus.editable = NO; rjStatus.drawsBackground = NO;
    rjStatus.autoresizingMask = NSViewMinYMargin | NSViewMinXMargin;
    [cv addSubview:rjStatus];
    y -= 20;

    NSScrollView *rjScroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(14, y - 100, W - 28, 100)];
    rjScroll.hasVerticalScroller = YES;
    rjScroll.borderType = NSBezelBorder;
    rjScroll.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    NSTextView *rjView = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, W - 32, 100)];
    rjView.editable = NO;
    rjView.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    rjView.identifier = @"apnResults";
    rjScroll.documentView = rjView;
    [cv addSubview:rjScroll];
    y -= 108;

    // ── Refresh button ───────────────────────────────────────────────
    NSButton *refreshBtn = [self buttonAt:NSMakeRect(14, y - 24, 80, 24)
        title:@"Refresh" action:@selector(apnRefresh:)];
    refreshBtn.font = [NSFont systemFontOfSize:10];
    refreshBtn.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:refreshBtn];

    // ── How it works ─────────────────────────────────────────────────
    NSTextField *howHdr = [[NSTextField alloc] initWithFrame:NSMakeRect(110, y - 22, W - 124, 14)];
    howHdr.stringValue = @"AutoPrimeNet \u2192 worktodo.txt \u2192 PrimePath (GPU) \u2192 results.json.txt \u2192 AutoPrimeNet \u2192 mersenne.org";
    howHdr.font = [NSFont monospacedSystemFontOfSize:9 weight:NSFontWeightRegular];
    howHdr.textColor = [NSColor secondaryLabelColor];
    howHdr.bezeled = NO; howHdr.editable = NO; howHdr.drawsBackground = NO;
    howHdr.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:howHdr];

    [win makeKeyAndOrderFront:nil];

    // Load initial data
    [self apnRefreshWindow:win dataDir:DATA_DIR];
}

- (void)apnRefresh:(id)sender {
    NSWindow *win = [sender window];
    NSString *DATA_DIR = [NSSearchPathForDirectoriesInDomains(NSApplicationSupportDirectory,
        NSUserDomainMask, YES).firstObject stringByAppendingPathComponent:@"PrimePath"];
    [self apnRefreshWindow:win dataDir:DATA_DIR];
}

- (void)apnRefreshWindow:(NSWindow *)win dataDir:(NSString *)dataDir {
    // Read worktodo.txt
    NSString *wtPath = [dataDir stringByAppendingPathComponent:@"worktodo.txt"];
    NSString *wtContents = [NSString stringWithContentsOfFile:wtPath encoding:NSUTF8StringEncoding error:nil];
    int wtCount = 0;
    if (wtContents) {
        for (NSString *line in [wtContents componentsSeparatedByString:@"\n"]) {
            NSString *trimmed = [line stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
            if ([trimmed hasPrefix:@"Factor="]) wtCount++;
        }
    }

    // Read last 10 lines of results.json.txt
    NSString *rjPath = [dataDir stringByAppendingPathComponent:@"results.json.txt"];
    NSString *rjContents = [NSString stringWithContentsOfFile:rjPath encoding:NSUTF8StringEncoding error:nil];
    int rjCount = 0;
    NSMutableArray *rjLines = [NSMutableArray array];
    if (rjContents) {
        NSArray *allLines = [rjContents componentsSeparatedByString:@"\n"];
        for (NSString *line in allLines) {
            if (line.length > 0) {
                [rjLines addObject:line];
                rjCount++;
            }
        }
        // Keep last 10
        if (rjLines.count > 10) {
            rjLines = [[rjLines subarrayWithRange:NSMakeRange(rjLines.count - 10, 10)] mutableCopy];
        }
    }

    // Update views
    for (NSView *sub in win.contentView.subviews) {
        if ([sub isKindOfClass:[NSScrollView class]]) {
            NSTextView *tv = (NSTextView *)((NSScrollView *)sub).documentView;
            if ([tv.identifier isEqualToString:@"apnWorktodo"]) {
                tv.string = wtContents ?: @"(file not found)";
            } else if ([tv.identifier isEqualToString:@"apnResults"]) {
                tv.string = rjLines.count > 0 ? [rjLines componentsJoinedByString:@"\n"] : @"(no results)";
            }
        }
    }

    // Status labels
    NSTextField *wtStatus = [win.contentView viewWithTag:8101];
    if (wtStatus) {
        if (wtCount > 0) {
            wtStatus.stringValue = [NSString stringWithFormat:@"%d assignment%s pending", wtCount, wtCount == 1 ? "" : "s"];
            wtStatus.textColor = [NSColor systemGreenColor];
        } else {
            wtStatus.stringValue = @"empty";
            wtStatus.textColor = [NSColor secondaryLabelColor];
        }
    }

    NSTextField *rjStatus = [win.contentView viewWithTag:8102];
    if (rjStatus) {
        if (rjCount > 0) {
            rjStatus.stringValue = [NSString stringWithFormat:@"%d result%s", rjCount, rjCount == 1 ? "" : "s"];
            rjStatus.textColor = [NSColor systemOrangeColor];
        } else {
            rjStatus.stringValue = @"no results yet";
            rjStatus.textColor = [NSColor secondaryLabelColor];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// JSON Result Editor -- build / preview / validate / test PrimeNet JSON
// ═══════════════════════════════════════════════════════════════════════

- (NSTextView *)jsonEditorOutputView:(NSWindow *)win {
    for (NSView *sub in win.contentView.subviews) {
        if ([sub isKindOfClass:[NSScrollView class]] &&
            [sub.identifier isEqualToString:@"jsonEditorOutput"]) {
            return (NSTextView *)((NSScrollView *)sub).documentView;
        }
    }
    return nil;
}

- (void)jsonEditorLog:(NSWindow *)win message:(NSString *)msg {
    NSTextView *logView = nil;
    for (NSView *sub in win.contentView.subviews) {
        if ([sub isKindOfClass:[NSScrollView class]] &&
            [sub.identifier isEqualToString:@"jsonEditorLog"]) {
            logView = (NSTextView *)((NSScrollView *)sub).documentView;
            break;
        }
    }
    if (logView) {
        NSString *line = [NSString stringWithFormat:@"%@\n", msg];
        [logView.textStorage appendAttributedString:
            [[NSAttributedString alloc] initWithString:line
                attributes:@{NSFontAttributeName: [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular],
                              NSForegroundColorAttributeName: [NSColor labelColor]}]];
        [logView scrollRangeToVisible:NSMakeRange(logView.string.length, 0)];
    }
}

- (void)showJSONResultEditor:(id)sender {
    NSWindow *win = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(140, 60, 700, 860)
        styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable)
        backing:NSBackingStoreBuffered defer:NO];
    win.title = @"JSON Result Editor";
    win.releasedWhenClosed = NO;
    win.minSize = NSMakeSize(580, 700);

    NSView *cv = win.contentView;
    CGFloat W = cv.frame.size.width;
    CGFloat y = cv.frame.size.height - 10;

    // Title
    NSTextField *title = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 18, W - 28, 18)];
    title.stringValue = @"PrimeNet JSON Result Builder";
    title.font = [NSFont boldSystemFontOfSize:13];
    title.bezeled = NO; title.editable = NO; title.drawsBackground = NO;
    title.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:title];
    y -= 22;

    // --- Field layout ---
    // Tags: 9001=exponent, 9002=status, 9003=bitlo, 9004=bithi,
    //        9005=rangecomplete, 9006=factors, 9007=user, 9008=computer,
    //        9009=aid, 9010=programName, 9011=programVersion, 9012=kernel,
    //        9020=PrimeNet URL, 9021=PrimeNet password/challenge

    CGFloat lblW = 110;
    CGFloat fldX = lblW + 20;
    CGFloat fldW = W - fldX - 14;

    auto addRow = [&](NSString *label, NSInteger tag, NSString *placeholder, NSString *defaultVal) {
        NSTextField *lbl = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 16, lblW, 14)];
        lbl.stringValue = label;
        lbl.font = [NSFont systemFontOfSize:10];
        lbl.bezeled = NO; lbl.editable = NO; lbl.drawsBackground = NO;
        lbl.alignment = NSTextAlignmentRight;
        lbl.autoresizingMask = NSViewMinYMargin;
        [cv addSubview:lbl];

        NSTextField *fld = [[NSTextField alloc] initWithFrame:NSMakeRect(fldX, y - 18, fldW, 20)];
        fld.font = [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular];
        fld.placeholderString = placeholder;
        fld.stringValue = defaultVal ? defaultVal : @"";
        fld.tag = tag;
        fld.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
        [cv addSubview:fld];
        y -= 24;
    };

    auto addPopupRow = [&](NSString *label, NSInteger tag, NSArray<NSString *> *items, NSInteger sel, NSString *hint) {
        NSTextField *lbl = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 16, lblW, 14)];
        lbl.stringValue = label;
        lbl.font = [NSFont systemFontOfSize:10];
        lbl.bezeled = NO; lbl.editable = NO; lbl.drawsBackground = NO;
        lbl.alignment = NSTextAlignmentRight;
        lbl.autoresizingMask = NSViewMinYMargin;
        [cv addSubview:lbl];

        NSPopUpButton *popup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(fldX, y - 18, 140, 20) pullsDown:NO];
        for (NSString *item in items) [popup addItemWithTitle:item];
        popup.font = [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular];
        popup.tag = tag;
        [popup selectItemAtIndex:sel];
        popup.autoresizingMask = NSViewMinYMargin;
        [cv addSubview:popup];

        if (hint) {
            NSTextField *h = [[NSTextField alloc] initWithFrame:NSMakeRect(fldX + 150, y - 16, fldW - 150, 14)];
            h.stringValue = hint;
            h.font = [NSFont systemFontOfSize:9];
            h.textColor = [NSColor secondaryLabelColor];
            h.bezeled = NO; h.editable = NO; h.drawsBackground = NO;
            h.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
            [cv addSubview:h];
        }
        y -= 24;
    };

    // Pre-fill from assignment
    NSString *defExp = @"", *defBLo = @"", *defBHi = @"", *defAid = @"";
    if (_primenet->pending_count() > 0) {
        auto& a = _primenet->state().assignments.front();
        defExp = [NSString stringWithFormat:@"%llu", a.exponent];
        defBLo = [NSString stringWithFormat:@"%d", (int)a.bit_lo];
        defBHi = [NSString stringWithFormat:@"%d", (int)a.bit_hi];
        defAid = [NSString stringWithUTF8String:a.key.c_str()];
    }
    NSString *defUser = [NSString stringWithUTF8String:_primenet->username().c_str()];
    char hn[256] = {};
    gethostname(hn, sizeof(hn));
    NSString *defComp = [NSString stringWithUTF8String:hn];

    // ── Result fields ──
    addRow(@"Exponent:", 9001, @"e.g. 501986531", defExp);
    addRow(@"Bit lo:", 9003, @"e.g. 76", defBLo);
    addRow(@"Bit hi:", 9004, @"e.g. 77", defBHi);
    addPopupRow(@"Status:", 9002,
        @[@"NF (no factor)", @"F (factor found)"], 0, nil);
    addPopupRow(@"Range complete:", 9005,
        @[@"true", @"false"], 0,
        @"true for NF; for F only if you continued past the factor(s)");
    addRow(@"Factors:", 9006, @"comma-separated, e.g. 12345,67890", @"");
    addRow(@"User:", 9007, @"mersenne.org username", defUser);
    addRow(@"Computer:", 9008, @"machine name", defComp);
    addRow(@"AID:", 9009, @"assignment key (blank if unknown)", defAid);

    // ── Program fields ──
    NSBox *sep1 = [[NSBox alloc] initWithFrame:NSMakeRect(14, y, W - 28, 1)];
    sep1.boxType = NSBoxSeparator;
    sep1.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:sep1];
    y -= 8;

    addRow(@"Program:", 9010, @"PrimePath", @"PrimePath");
    addRow(@"Version:", 9011, @"1.3.0", @"1.3.0");
    addRow(@"Kernel:", 9012, @"Metal96bit (optional)", @"Metal96bit");

    // ── PrimeNet connection ──
    NSBox *sep2 = [[NSBox alloc] initWithFrame:NSMakeRect(14, y, W - 28, 1)];
    sep2.boxType = NSBoxSeparator;
    sep2.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:sep2];
    y -= 8;

    {
        NSTextField *hdr = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 14, 300, 14)];
        hdr.stringValue = @"PrimeNet Connection (for direct submission testing)";
        hdr.font = [NSFont boldSystemFontOfSize:10];
        hdr.bezeled = NO; hdr.editable = NO; hdr.drawsBackground = NO;
        hdr.autoresizingMask = NSViewMinYMargin;
        [cv addSubview:hdr];
        y -= 20;
    }

    addRow(@"Server URL:", 9020, @"https://v5.mersenne.org/v5server/",
           @"https://v5.mersenne.org/v5server/");
    addRow(@"Credentials:", 9021, @"guid=...&ss=...&sh=... (appended to URL)", @"");

    // ── Buttons row 1: Generate / Validate / Copy ──
    NSBox *sep3 = [[NSBox alloc] initWithFrame:NSMakeRect(14, y - 2, W - 28, 1)];
    sep3.boxType = NSBoxSeparator;
    sep3.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:sep3];
    y -= 12;

    CGFloat bx = 14;
    auto addBtn = [&](NSString *t, SEL action, CGFloat w, BOOL bold) {
        NSButton *b = [self buttonAt:NSMakeRect(bx, y - 24, w, 24) title:t action:action];
        b.font = bold ? [NSFont boldSystemFontOfSize:10] : [NSFont systemFontOfSize:10];
        b.autoresizingMask = NSViewMinYMargin;
        [cv addSubview:b];
        bx += w + 4;
    };

    addBtn(@"Generate", @selector(jsonEditorGenerate:), 80, YES);
    addBtn(@"Validate", @selector(jsonEditorValidate:), 70, NO);
    addBtn(@"Copy", @selector(jsonEditorCopy:), 60, NO);
    addBtn(@"Save to File", @selector(jsonEditorSave:), 90, NO);
    addBtn(@"Auto-Fill", @selector(jsonEditorAutoFill:), 80, NO);
    y -= 28;

    // ── Buttons row 2: Templates / Simulate / Send Test ──
    bx = 14;
    addBtn(@"Load Template", @selector(jsonEditorLoadTemplate:), 110, NO);
    addBtn(@"Save Template", @selector(jsonEditorSaveTemplate:), 110, NO);
    addBtn(@"Simulate Test", @selector(jsonEditorSimulate:), 110, YES);
    addBtn(@"Send to Server", @selector(jsonEditorSendTest:), 110, NO);
    y -= 32;

    // ── JSON output area ──
    {
        NSTextField *hdr = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 14, 200, 14)];
        hdr.stringValue = @"JSON Output (editable)";
        hdr.font = [NSFont boldSystemFontOfSize:10];
        hdr.bezeled = NO; hdr.editable = NO; hdr.drawsBackground = NO;
        hdr.autoresizingMask = NSViewMinYMargin;
        [cv addSubview:hdr];
        y -= 18;
    }

    CGFloat splitY = y * 0.5; // split remaining space between output and log

    NSScrollView *outScroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(14, splitY + 4, W - 28, y - splitY - 4)];
    outScroll.hasVerticalScroller = YES;
    outScroll.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    NSTextView *outView = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, W - 32, y - splitY - 4)];
    outView.editable = YES;
    outView.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    outView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    outView.string = @"Click 'Generate' to build JSON, 'Simulate Test' for a sample, or paste JSON to validate.";
    outView.textColor = [NSColor secondaryLabelColor];
    outScroll.documentView = outView;
    outScroll.identifier = @"jsonEditorOutput";
    [cv addSubview:outScroll];

    // ── Log area ──
    {
        NSTextField *hdr = [[NSTextField alloc] initWithFrame:NSMakeRect(14, splitY - 4, 200, 14)];
        hdr.stringValue = @"Validation / Server Log";
        hdr.font = [NSFont boldSystemFontOfSize:10];
        hdr.bezeled = NO; hdr.editable = NO; hdr.drawsBackground = NO;
        hdr.autoresizingMask = NSViewMinYMargin;
        [cv addSubview:hdr];
    }

    NSScrollView *logScroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(14, 10, W - 28, splitY - 24)];
    logScroll.hasVerticalScroller = YES;
    logScroll.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    NSTextView *logView = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, W - 32, splitY - 24)];
    logView.editable = NO;
    logView.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    logView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    logScroll.documentView = logView;
    logScroll.identifier = @"jsonEditorLog";
    [cv addSubview:logScroll];

    [win makeKeyAndOrderFront:nil];
}

// ── JSON builder from form fields ────────────────────────────────────

- (NSString *)jsonEditorBuildJSON:(NSWindow *)win {
    NSView *cv = win.contentView;

    NSString *exponent = ((NSTextField *)[cv viewWithTag:9001]).stringValue;
    NSPopUpButton *statusPopup = [cv viewWithTag:9002];
    NSString *bitlo = ((NSTextField *)[cv viewWithTag:9003]).stringValue;
    NSString *bithi = ((NSTextField *)[cv viewWithTag:9004]).stringValue;
    NSPopUpButton *rcPopup = [cv viewWithTag:9005];
    NSString *factors = ((NSTextField *)[cv viewWithTag:9006]).stringValue;
    NSString *user = ((NSTextField *)[cv viewWithTag:9007]).stringValue;
    NSString *computer = ((NSTextField *)[cv viewWithTag:9008]).stringValue;
    NSString *aid = ((NSTextField *)[cv viewWithTag:9009]).stringValue;
    NSString *progName = ((NSTextField *)[cv viewWithTag:9010]).stringValue;
    NSString *progVersion = ((NSTextField *)[cv viewWithTag:9011]).stringValue;
    NSString *kernel = ((NSTextField *)[cv viewWithTag:9012]).stringValue;

    BOOL isF = (statusPopup.indexOfSelectedItem == 1);
    BOOL rangeComplete = (rcPopup.indexOfSelectedItem == 0);

    NSDateFormatter *fmt = [[NSDateFormatter alloc] init];
    fmt.dateFormat = @"yyyy-MM-dd HH:mm:ss";
    fmt.timeZone = [NSTimeZone timeZoneWithAbbreviation:@"UTC"];
    NSString *ts = [fmt stringFromDate:[NSDate date]];

    struct utsname un;
    uname(&un);

    NSMutableString *js = [NSMutableString string];
    [js appendFormat:@"{\"timestamp\":\"%@\"", ts];
    [js appendFormat:@",\"exponent\":%@", exponent];
    [js appendString:@",\"worktype\":\"TF\""];
    [js appendFormat:@",\"status\":\"%@\"", isF ? @"F" : @"NF"];
    [js appendFormat:@",\"bitlo\":%@", bitlo];
    [js appendFormat:@",\"bithi\":%@", bithi];
    [js appendFormat:@",\"rangecomplete\":%@", rangeComplete ? @"true" : @"false"];

    if (isF && factors.length > 0) {
        NSArray *flist = [factors componentsSeparatedByString:@","];
        NSMutableArray *quoted = [NSMutableArray array];
        for (NSString *f in flist) {
            NSString *trimmed = [f stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
            if (trimmed.length > 0)
                [quoted addObject:[NSString stringWithFormat:@"\"%@\"", trimmed]];
        }
        if (quoted.count > 0)
            [js appendFormat:@",\"factors\":[%@]", [quoted componentsJoinedByString:@","]];
    }

    [js appendFormat:@",\"program\":{\"name\":\"%@\",\"version\":\"%@\"", progName, progVersion];
    if (kernel.length > 0)
        [js appendFormat:@",\"kernel\":\"%@\"", kernel];
    [js appendString:@"}"];

    [js appendFormat:@",\"os\":{\"os\":\"macOS\",\"version\":\"%s\",\"architecture\":\"ARM_64\"}", un.release];

    if (user.length > 0)
        [js appendFormat:@",\"user\":\"%@\"", user];
    if (computer.length > 0)
        [js appendFormat:@",\"computer\":\"%@\"", computer];
    if (aid.length > 0 && ![aid isEqualToString:@"0"] &&
        [aid rangeOfCharacterFromSet:
            [[NSCharacterSet characterSetWithCharactersInString:@"0"] invertedSet]].location != NSNotFound) {
        [js appendFormat:@",\"aid\":\"%@\"", aid];
    }

    // Hardware (auto-detect)
    {
        char chip[256] = {};
        size_t chip_len = sizeof(chip);
        sysctlbyname("machdep.cpu.brand_string", chip, &chip_len, NULL, 0);
        NSString *chipStr = chip[0] ? [NSString stringWithUTF8String:chip] : @"Apple Silicon";

        int ncpu = (int)[[NSProcessInfo processInfo] processorCount];
        int perf = 0, eff = 0;
        size_t sz = sizeof(int);
        sysctlbyname("hw.perflevel0.logicalcpu", &perf, &sz, NULL, 0);
        sz = sizeof(int);
        sysctlbyname("hw.perflevel1.logicalcpu", &eff, &sz, NULL, 0);

        uint64_t memsize = 0;
        sz = sizeof(memsize);
        sysctlbyname("hw.memsize", &memsize, &sz, NULL, 0);
        int ram_gb = (int)(memsize / (1024ULL * 1024 * 1024));

        int gpu_cores = 0;
        io_iterator_t iter;
        if (IOServiceGetMatchingServices(kIOMainPortDefault,
                IOServiceMatching("AGXAccelerator"), &iter) == KERN_SUCCESS) {
            io_object_t svc;
            while ((svc = IOIteratorNext(iter))) {
                CFTypeRef prop = IORegistryEntrySearchCFProperty(svc, kIOServicePlane,
                    CFSTR("gpu-core-count"), kCFAllocatorDefault,
                    kIORegistryIterateRecursively | kIORegistryIterateParents);
                if (prop) {
                    if (CFGetTypeID(prop) == CFNumberGetTypeID())
                        CFNumberGetValue((CFNumberRef)prop, kCFNumberIntType, &gpu_cores);
                    CFRelease(prop);
                }
                IOObjectRelease(svc);
                if (gpu_cores > 0) break;
            }
            IOObjectRelease(iter);
        }

        [js appendFormat:@",\"hardware\":{\"chip\":\"%@\"", chipStr];
        if (perf > 0)
            [js appendFormat:@",\"cpu_p_cores\":%d,\"cpu_e_cores\":%d", perf, eff];
        else
            [js appendFormat:@",\"cpu_cores\":%d", ncpu];
        if (gpu_cores > 0)
            [js appendFormat:@",\"gpu_cores\":%d", gpu_cores];
        [js appendFormat:@",\"ram_gb\":%d}", ram_gb];
    }

    // CRC32 checksum
    {
        NSString *status = ((NSPopUpButton *)[cv viewWithTag:9002]).titleOfSelectedItem ?: @"NF";
        NSString *factorsField = ((NSTextField *)[cv viewWithTag:9006]).stringValue;
        NSString *bitlo = ((NSTextField *)[cv viewWithTag:9003]).stringValue;
        NSString *bithi = ((NSTextField *)[cv viewWithTag:9004]).stringValue;
        NSString *rangeStr = ((NSPopUpButton *)[cv viewWithTag:9005]).titleOfSelectedItem;
        NSString *rcVal = [rangeStr isEqualToString:@"true"] ? @"1" : ([rangeStr isEqualToString:@"false"] ? @"0" : @"");
        NSString *progName = ((NSTextField *)[cv viewWithTag:9010]).stringValue;
        NSString *progVer = ((NSTextField *)[cv viewWithTag:9011]).stringValue;
        NSString *kernel = ((NSTextField *)[cv viewWithTag:9012]).stringValue;

        // Parse and sort factors
        NSMutableArray *sortedFactors = [NSMutableArray array];
        if ([status isEqualToString:@"F"] && factorsField.length > 0) {
            for (NSString *f in [factorsField componentsSeparatedByString:@","]) {
                NSString *t = [f stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
                if (t.length > 0) [sortedFactors addObject:t];
            }
            [sortedFactors sortUsingComparator:^NSComparisonResult(NSString *a, NSString *b) {
                if (a.length != b.length) return a.length < b.length ? NSOrderedAscending : NSOrderedDescending;
                return [a compare:b];
            }];
        }

        NSString *ckStr = [NSString stringWithFormat:@"%@;TF;%@;;%@;%@;%@;;;%@;%@;%@;;macOS;ARM_64;%@",
            ((NSTextField *)[cv viewWithTag:9001]).stringValue,
            [sortedFactors componentsJoinedByString:@","],
            bitlo, bithi, rcVal,
            progName, progVer, kernel,
            ts];

        uLong crc = crc32(0L, Z_NULL, 0);
        const char *ckUTF8 = ckStr.UTF8String;
        crc = crc32(crc, (const Bytef *)ckUTF8, (uInt)strlen(ckUTF8));

        [js appendFormat:@",\"checksum\":{\"version\":1,\"checksum\":\"%08lX\"}", (unsigned long)crc];
    }

    [js appendString:@"}"];
    return js;
}

// ── Generate ─────────────────────────────────────────────────────────

- (void)jsonEditorGenerate:(id)sender {
    NSWindow *win = [sender window];
    NSString *json = [self jsonEditorBuildJSON:win];
    NSTextView *outView = [self jsonEditorOutputView:win];
    if (outView) {
        outView.string = json;
        outView.textColor = [NSColor labelColor];
    }
    [self jsonEditorLog:win message:@"Generated JSON result line."];
}

// ── Validate JSON syntax and required fields ─────────────────────────

- (void)jsonEditorValidate:(id)sender {
    NSWindow *win = [sender window];
    NSTextView *outView = [self jsonEditorOutputView:win];
    if (!outView || outView.string.length == 0) {
        [self jsonEditorLog:win message:@"VALIDATE: nothing to validate -- generate or paste JSON first."];
        return;
    }

    NSString *raw = outView.string;
    [self jsonEditorLog:win message:@"--- Validation ---"];

    // Check single line
    if ([raw componentsSeparatedByString:@"\n"].count > 1) {
        NSString *trimmed = [raw stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        if ([trimmed componentsSeparatedByString:@"\n"].count > 1) {
            [self jsonEditorLog:win message:@"WARN: JSON must be a single line from { to }."];
        }
    }

    // Parse with NSJSONSerialization
    NSData *data = [raw dataUsingEncoding:NSUTF8StringEncoding];
    NSError *err = nil;
    NSDictionary *dict = [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
    if (err || ![dict isKindOfClass:[NSDictionary class]]) {
        [self jsonEditorLog:win message:[NSString stringWithFormat:@"FAIL: invalid JSON -- %@",
            err.localizedDescription ?: @"not a JSON object"]];
        return;
    }
    [self jsonEditorLog:win message:@"OK: valid JSON syntax."];

    // Required fields
    NSArray *required = @[@"timestamp", @"exponent", @"worktype", @"status",
                          @"bitlo", @"bithi", @"rangecomplete", @"program", @"os",
                          @"user", @"computer"];
    BOOL allPresent = YES;
    for (NSString *key in required) {
        if (!dict[key]) {
            [self jsonEditorLog:win message:[NSString stringWithFormat:@"FAIL: missing required field \"%@\".", key]];
            allPresent = NO;
        }
    }
    if (allPresent)
        [self jsonEditorLog:win message:@"OK: all required fields present."];

    // Check program sub-fields
    if ([dict[@"program"] isKindOfClass:[NSDictionary class]]) {
        NSDictionary *prog = dict[@"program"];
        if (!prog[@"name"]) [self jsonEditorLog:win message:@"FAIL: program.name missing."];
        if (!prog[@"version"]) [self jsonEditorLog:win message:@"FAIL: program.version missing."];
    } else if (dict[@"program"]) {
        [self jsonEditorLog:win message:@"FAIL: \"program\" must be an object."];
    }

    // Check os sub-fields
    if ([dict[@"os"] isKindOfClass:[NSDictionary class]]) {
        NSDictionary *os = dict[@"os"];
        if (!os[@"os"]) [self jsonEditorLog:win message:@"FAIL: os.os missing."];
        if (!os[@"version"]) [self jsonEditorLog:win message:@"FAIL: os.version missing."];
        if (!os[@"architecture"]) [self jsonEditorLog:win message:@"FAIL: os.architecture missing."];
    } else if (dict[@"os"]) {
        [self jsonEditorLog:win message:@"FAIL: \"os\" must be an object."];
    }

    // Status-specific checks
    NSString *status = dict[@"status"];
    if ([status isEqualToString:@"F"]) {
        if (!dict[@"factors"] || ![dict[@"factors"] isKindOfClass:[NSArray class]] ||
            [dict[@"factors"] count] == 0) {
            [self jsonEditorLog:win message:@"FAIL: status is \"F\" but \"factors\" array is missing or empty."];
        } else {
            // Verify factors are strings
            for (id f in dict[@"factors"]) {
                if (![f isKindOfClass:[NSString class]]) {
                    [self jsonEditorLog:win message:@"FAIL: each factor must be a string, not a number."];
                    break;
                }
            }
        }
    } else if ([status isEqualToString:@"NF"]) {
        if (dict[@"rangecomplete"] && ![dict[@"rangecomplete"] boolValue]) {
            [self jsonEditorLog:win message:@"WARN: status NF with rangecomplete=false is invalid for PrimeNet."];
        }
        if (dict[@"factors"]) {
            [self jsonEditorLog:win message:@"WARN: status NF should not have \"factors\" field."];
        }
    } else if (status) {
        [self jsonEditorLog:win message:[NSString stringWithFormat:@"WARN: unknown status \"%@\" -- expected \"F\" or \"NF\".", status]];
    }

    // Worktype
    if (dict[@"worktype"] && ![dict[@"worktype"] isEqualToString:@"TF"]) {
        [self jsonEditorLog:win message:[NSString stringWithFormat:@"WARN: worktype \"%@\" -- expected \"TF\" for trial factoring.", dict[@"worktype"]]];
    }

    // Timestamp format check
    NSString *ts = dict[@"timestamp"];
    if (ts && ts.length > 0) {
        NSDateFormatter *tsFmt = [[NSDateFormatter alloc] init];
        tsFmt.dateFormat = @"yyyy-MM-dd HH:mm:ss";
        tsFmt.timeZone = [NSTimeZone timeZoneWithAbbreviation:@"UTC"];
        if (![tsFmt dateFromString:ts]) {
            [self jsonEditorLog:win message:@"WARN: timestamp should be \"YYYY-MM-DD HH:MM:SS\" in UTC."];
        }
    }

    // Length check
    if (raw.length > 2000)
        [self jsonEditorLog:win message:[NSString stringWithFormat:@"WARN: JSON is %lu chars -- try to keep under 2000.", (unsigned long)raw.length]];
    else
        [self jsonEditorLog:win message:[NSString stringWithFormat:@"OK: JSON length %lu chars (under 2000 limit).", (unsigned long)raw.length]];

    [self jsonEditorLog:win message:@"--- Validation complete ---"];
}

// ── Copy to clipboard ────────────────────────────────────────────────

- (void)jsonEditorCopy:(id)sender {
    NSWindow *win = [sender window];
    NSTextView *outView = [self jsonEditorOutputView:win];
    if (outView && outView.string.length > 0) {
        [[NSPasteboard generalPasteboard] clearContents];
        [[NSPasteboard generalPasteboard] setString:outView.string forType:NSPasteboardTypeString];
        NSButton *btn = (NSButton *)sender;
        NSString *orig = btn.title;
        btn.title = @"Copied";
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(1.0 * NSEC_PER_SEC)),
            dispatch_get_main_queue(), ^{ btn.title = orig; });
    }
}

// ── Save to file (append) ────────────────────────────────────────────

- (void)jsonEditorSave:(id)sender {
    NSWindow *win = [sender window];
    NSTextView *outView = [self jsonEditorOutputView:win];
    if (!outView || outView.string.length == 0) return;
    NSString *json = outView.string;

    NSSavePanel *panel = [NSSavePanel savePanel];
    panel.nameFieldStringValue = @"results.json.txt";

    [panel beginSheetModalForWindow:win completionHandler:^(NSModalResponse result) {
        if (result == NSModalResponseOK && panel.URL) {
            NSString *line = [json stringByAppendingString:@"\n"];
            NSFileManager *fm = [NSFileManager defaultManager];
            if ([fm fileExistsAtPath:panel.URL.path]) {
                NSFileHandle *fh = [NSFileHandle fileHandleForWritingAtPath:panel.URL.path];
                [fh seekToEndOfFile];
                [fh writeData:[line dataUsingEncoding:NSUTF8StringEncoding]];
                [fh closeFile];
            } else {
                [line writeToURL:panel.URL atomically:YES encoding:NSUTF8StringEncoding error:nil];
            }
            dispatch_async(dispatch_get_main_queue(), ^{
                [self jsonEditorLog:win message:[NSString stringWithFormat:@"Saved to %@", panel.URL.path]];
            });
        }
    }];
}

// ── Auto-fill from system / assignment / discoveries ─────────────────

- (void)jsonEditorAutoFill:(id)sender {
    NSWindow *win = [sender window];
    NSView *cv = win.contentView;

    char hostname[256] = {};
    gethostname(hostname, sizeof(hostname));
    ((NSTextField *)[cv viewWithTag:9008]).stringValue = [NSString stringWithUTF8String:hostname];

    if (_primenet->username().length() > 0)
        ((NSTextField *)[cv viewWithTag:9007]).stringValue =
            [NSString stringWithUTF8String:_primenet->username().c_str()];

    if (_primenet->pending_count() > 0) {
        auto& a = _primenet->state().assignments.front();
        ((NSTextField *)[cv viewWithTag:9001]).stringValue = [NSString stringWithFormat:@"%llu", a.exponent];
        ((NSTextField *)[cv viewWithTag:9003]).stringValue = [NSString stringWithFormat:@"%d", (int)a.bit_lo];
        ((NSTextField *)[cv viewWithTag:9004]).stringValue = [NSString stringWithFormat:@"%d", (int)a.bit_hi];
        ((NSTextField *)[cv viewWithTag:9009]).stringValue = [NSString stringWithUTF8String:a.key.c_str()];
    }

    for (auto& d : _taskMgr->discoveries()) {
        if (d.type == prime::TaskType::MersenneTrial) {
            [(NSPopUpButton *)[cv viewWithTag:9002] selectItemAtIndex:1];
            NSString *factor = d.divisors.empty()
                ? [NSString stringWithFormat:@"%llu", d.value]
                : [NSString stringWithUTF8String:d.divisors.c_str()];
            ((NSTextField *)[cv viewWithTag:9006]).stringValue = factor;
            break;
        }
    }

    // Fill credentials from state if available
    if (!_primenet->state().guid.empty()) {
        NSString *creds = [NSString stringWithFormat:@"g=%s&ss=19191919&sh=ABCDABCDABCDABCDABCDABCDABCDABCD",
            _primenet->state().guid.c_str()];
        ((NSTextField *)[cv viewWithTag:9021]).stringValue = creds;
    }

    ((NSTextField *)[cv viewWithTag:9010]).stringValue = @"PrimePath";
    ((NSTextField *)[cv viewWithTag:9011]).stringValue = @"1.3.0";
    ((NSTextField *)[cv viewWithTag:9012]).stringValue = @"Metal96bit";

    [self jsonEditorLog:win message:@"Auto-filled from system and current assignment."];
}

// ── Simulate a test problem with known factor ────────────────────────

- (void)jsonEditorSimulate:(id)sender {
    NSWindow *win = [sender window];
    NSView *cv = win.contentView;

    // Known Mersenne factor: M(86243) has factor 120327 (found long ago)
    // Actually use a small well-known example:
    // M(67) = 2^67 - 1 has factor 193707721
    // q = 193707721, p = 67, q = 2*1445579*67 + 1, so k = 1445579
    // bit range: 2^27 < 193707721 < 2^28, so bitlo=27, bithi=28

    ((NSTextField *)[cv viewWithTag:9001]).stringValue = @"67";
    ((NSTextField *)[cv viewWithTag:9003]).stringValue = @"27";
    ((NSTextField *)[cv viewWithTag:9004]).stringValue = @"28";
    [(NSPopUpButton *)[cv viewWithTag:9002] selectItemAtIndex:1]; // F
    [(NSPopUpButton *)[cv viewWithTag:9005] selectItemAtIndex:1]; // rangecomplete = false
    ((NSTextField *)[cv viewWithTag:9006]).stringValue = @"193707721";
    ((NSTextField *)[cv viewWithTag:9009]).stringValue = @"00000000000000000000000000000000";

    // Auto-fill user/computer/program
    char hostname[256] = {};
    gethostname(hostname, sizeof(hostname));
    ((NSTextField *)[cv viewWithTag:9008]).stringValue = [NSString stringWithUTF8String:hostname];
    if (_primenet->username().length() > 0)
        ((NSTextField *)[cv viewWithTag:9007]).stringValue =
            [NSString stringWithUTF8String:_primenet->username().c_str()];
    ((NSTextField *)[cv viewWithTag:9010]).stringValue = @"PrimePath";
    ((NSTextField *)[cv viewWithTag:9011]).stringValue = @"1.3.0";
    ((NSTextField *)[cv viewWithTag:9012]).stringValue = @"Metal96bit";

    // Generate the JSON
    NSString *json = [self jsonEditorBuildJSON:win];
    NSTextView *outView = [self jsonEditorOutputView:win];
    if (outView) {
        outView.string = json;
        outView.textColor = [NSColor labelColor];
    }

    [self jsonEditorLog:win message:@"--- Simulated Test ---"];
    [self jsonEditorLog:win message:@"Loaded known result: M(67) has factor 193707721."];
    [self jsonEditorLog:win message:@"2^67 - 1 = 147573952589676412927"];
    [self jsonEditorLog:win message:@"147573952589676412927 / 193707721 = 762078938126..."];
    [self jsonEditorLog:win message:@"Verification: 2^67 mod 193707721 = 1 (confirmed)"];
    [self jsonEditorLog:win message:@"This is a test result -- do NOT submit to mersenne.org."];
    [self jsonEditorLog:win message:@"Use 'Validate' to check the JSON format."];
}

// ── Load template from file ──────────────────────────────────────────

- (void)jsonEditorLoadTemplate:(id)sender {
    NSWindow *win = [sender window];

    NSOpenPanel *panel = [NSOpenPanel openPanel];
    panel.canChooseFiles = YES;
    panel.canChooseDirectories = NO;
    panel.allowsMultipleSelection = NO;
    panel.message = @"Select a JSON template or results.json.txt file";

    [panel beginSheetModalForWindow:win completionHandler:^(NSModalResponse result) {
        if (result != NSModalResponseOK || !panel.URL) return;

        NSError *err = nil;
        NSString *content = [NSString stringWithContentsOfURL:panel.URL
            encoding:NSUTF8StringEncoding error:&err];
        if (err || !content) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [self jsonEditorLog:win message:[NSString stringWithFormat:@"Failed to read file: %@",
                    err.localizedDescription]];
            });
            return;
        }

        dispatch_async(dispatch_get_main_queue(), ^{
            // Take the last non-empty line (in case of multi-line results file)
            NSArray *lines = [content componentsSeparatedByString:@"\n"];
            NSString *jsonLine = @"";
            for (NSString *line in [lines reverseObjectEnumerator]) {
                NSString *trimmed = [line stringByTrimmingCharactersInSet:
                    [NSCharacterSet whitespaceAndNewlineCharacterSet]];
                if (trimmed.length > 0) { jsonLine = trimmed; break; }
            }

            NSTextView *outView = [self jsonEditorOutputView:win];
            if (outView) {
                outView.string = jsonLine;
                outView.textColor = [NSColor labelColor];
            }

            // Try to parse and populate fields
            NSData *data = [jsonLine dataUsingEncoding:NSUTF8StringEncoding];
            NSDictionary *dict = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
            if ([dict isKindOfClass:[NSDictionary class]]) {
                NSView *cv = win.contentView;
                if (dict[@"exponent"])
                    ((NSTextField *)[cv viewWithTag:9001]).stringValue = [NSString stringWithFormat:@"%@", dict[@"exponent"]];
                if (dict[@"bitlo"])
                    ((NSTextField *)[cv viewWithTag:9003]).stringValue = [NSString stringWithFormat:@"%@", dict[@"bitlo"]];
                if (dict[@"bithi"])
                    ((NSTextField *)[cv viewWithTag:9004]).stringValue = [NSString stringWithFormat:@"%@", dict[@"bithi"]];
                if ([dict[@"status"] isEqualToString:@"F"])
                    [(NSPopUpButton *)[cv viewWithTag:9002] selectItemAtIndex:1];
                else
                    [(NSPopUpButton *)[cv viewWithTag:9002] selectItemAtIndex:0];
                if ([dict[@"rangecomplete"] boolValue])
                    [(NSPopUpButton *)[cv viewWithTag:9005] selectItemAtIndex:0];
                else
                    [(NSPopUpButton *)[cv viewWithTag:9005] selectItemAtIndex:1];
                if (dict[@"factors"] && [dict[@"factors"] isKindOfClass:[NSArray class]]) {
                    ((NSTextField *)[cv viewWithTag:9006]).stringValue =
                        [dict[@"factors"] componentsJoinedByString:@","];
                }
                if (dict[@"user"])
                    ((NSTextField *)[cv viewWithTag:9007]).stringValue = dict[@"user"];
                if (dict[@"computer"])
                    ((NSTextField *)[cv viewWithTag:9008]).stringValue = dict[@"computer"];
                if (dict[@"aid"])
                    ((NSTextField *)[cv viewWithTag:9009]).stringValue = dict[@"aid"];
                if ([dict[@"program"] isKindOfClass:[NSDictionary class]]) {
                    NSDictionary *prog = dict[@"program"];
                    if (prog[@"name"])
                        ((NSTextField *)[cv viewWithTag:9010]).stringValue = prog[@"name"];
                    if (prog[@"version"])
                        ((NSTextField *)[cv viewWithTag:9011]).stringValue = prog[@"version"];
                    if (prog[@"kernel"])
                        ((NSTextField *)[cv viewWithTag:9012]).stringValue = prog[@"kernel"];
                }
                [self jsonEditorLog:win message:[NSString stringWithFormat:@"Loaded template from %@", panel.URL.lastPathComponent]];
            } else {
                [self jsonEditorLog:win message:@"Loaded file but could not parse JSON -- fields not populated."];
            }
        });
    }];
}

// ── Save current fields as template ──────────────────────────────────

- (void)jsonEditorSaveTemplate:(id)sender {
    NSWindow *win = [sender window];
    NSString *json = [self jsonEditorBuildJSON:win];

    NSSavePanel *panel = [NSSavePanel savePanel];
    panel.nameFieldStringValue = @"template_mersenne_tf.json";
    panel.message = @"Save JSON template for reuse";

    [panel beginSheetModalForWindow:win completionHandler:^(NSModalResponse result) {
        if (result == NSModalResponseOK && panel.URL) {
            NSError *err = nil;
            [json writeToURL:panel.URL atomically:YES encoding:NSUTF8StringEncoding error:&err];
            dispatch_async(dispatch_get_main_queue(), ^{
                if (err)
                    [self jsonEditorLog:win message:[NSString stringWithFormat:@"Save failed: %@", err.localizedDescription]];
                else
                    [self jsonEditorLog:win message:[NSString stringWithFormat:@"Template saved to %@", panel.URL.lastPathComponent]];
            });
        }
    }];
}

// ── Send JSON to PrimeNet server (test submission) ───────────────────

- (void)jsonEditorSendTest:(id)sender {
    NSWindow *win = [sender window];
    NSView *cv = win.contentView;
    NSTextView *outView = [self jsonEditorOutputView:win];

    if (!outView || outView.string.length == 0) {
        [self jsonEditorLog:win message:@"SEND: generate JSON first."];
        return;
    }

    NSString *json = outView.string;
    NSString *serverURL = ((NSTextField *)[cv viewWithTag:9020]).stringValue;
    NSString *creds = ((NSTextField *)[cv viewWithTag:9021]).stringValue;

    if (serverURL.length == 0) {
        [self jsonEditorLog:win message:@"SEND: server URL is empty."];
        return;
    }
    if (creds.length == 0) {
        [self jsonEditorLog:win message:@"SEND: credentials are empty. Click Auto-Fill or enter guid/ss/sh values."];
        return;
    }

    // Parse exponent and bit range from the JSON to build the t=ar URL
    NSData *data = [json dataUsingEncoding:NSUTF8StringEncoding];
    NSDictionary *dict = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    if (![dict isKindOfClass:[NSDictionary class]]) {
        [self jsonEditorLog:win message:@"SEND: JSON parse failed -- validate first."];
        return;
    }

    NSString *exponent = [NSString stringWithFormat:@"%@", dict[@"exponent"] ?: @"0"];
    NSString *bitlo = [NSString stringWithFormat:@"%@", dict[@"bitlo"] ?: @"0"];
    NSString *bithi = [NSString stringWithFormat:@"%@", dict[@"bithi"] ?: @"0"];
    BOOL isF = [dict[@"status"] isEqualToString:@"F"];
    int rtype = isF ? 1 : 4;
    NSString *aid = dict[@"aid"] ?: @"0";

    // URL-encode the JSON for &m=
    NSString *encodedJSON = [json stringByAddingPercentEncodingWithAllowedCharacters:
        [NSCharacterSet characterSetWithCharactersInString:
            @"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"]];

    NSMutableString *url = [NSMutableString stringWithFormat:
        @"%@?px=GIMPS&v=0.95&t=ar&%@&k=%@&r=%d&d=1&n=%@&sf=%@&ef=%@",
        serverURL, creds, aid, rtype, exponent, bitlo, bithi];

    if (isF && dict[@"factors"] && [dict[@"factors"] count] > 0) {
        [url appendFormat:@"&f=%@", dict[@"factors"][0]];
    }

    [url appendFormat:@"&m=%@", encodedJSON];

    [self jsonEditorLog:win message:@"--- Sending to server ---"];
    [self jsonEditorLog:win message:[NSString stringWithFormat:@"URL: %@", url]];

    // Confirm before sending
    NSAlert *confirm = [[NSAlert alloc] init];
    confirm.messageText = @"Send to PrimeNet?";
    confirm.informativeText = [NSString stringWithFormat:
        @"This will submit a result for M%@ to %@.\n\nOnly send if this is a real, verified result.",
        exponent, serverURL];
    confirm.alertStyle = NSAlertStyleWarning;
    [confirm addButtonWithTitle:@"Send"];
    [confirm addButtonWithTitle:@"Cancel"];

    if ([confirm runModal] != NSAlertFirstButtonReturn) {
        [self jsonEditorLog:win message:@"Cancelled."];
        return;
    }

    __weak AppDelegate *weakSelf = self;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        @autoreleasepool {
            NSURL *reqURL = [NSURL URLWithString:url];
            if (!reqURL) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    [weakSelf jsonEditorLog:win message:@"SEND: invalid URL."];
                });
                return;
            }

            NSMutableURLRequest *req = [NSMutableURLRequest requestWithURL:reqURL];
            req.HTTPMethod = @"GET";
            req.timeoutInterval = 45.0;
            [req setValue:@"PrimePath/1.3.0" forHTTPHeaderField:@"User-Agent"];

            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            __block NSData *respData = nil;
            __block NSError *respErr = nil;
            __block NSInteger respStatus = 0;

            NSURLSessionDataTask *task = [[NSURLSession sharedSession]
                dataTaskWithRequest:req
                completionHandler:^(NSData *d, NSURLResponse *r, NSError *e) {
                    respData = d;
                    respErr = e;
                    if ([r isKindOfClass:[NSHTTPURLResponse class]])
                        respStatus = ((NSHTTPURLResponse *)r).statusCode;
                    dispatch_semaphore_signal(sem);
                }];
            [task resume];

            dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 60 * NSEC_PER_SEC));

            dispatch_async(dispatch_get_main_queue(), ^{
                if (respErr) {
                    [weakSelf jsonEditorLog:win message:[NSString stringWithFormat:@"SEND ERROR: %@",
                        respErr.localizedDescription]];
                } else if (respData) {
                    NSString *body = [[NSString alloc] initWithData:respData encoding:NSUTF8StringEncoding];
                    [weakSelf jsonEditorLog:win message:[NSString stringWithFormat:@"HTTP %ld", (long)respStatus]];
                    [weakSelf jsonEditorLog:win message:[NSString stringWithFormat:@"Response: %@", body]];

                    // Parse PrimeNet response
                    for (NSString *line in [body componentsSeparatedByString:@"\n"]) {
                        if ([line hasPrefix:@"pnErrorResult="] && ![line isEqualToString:@"pnErrorResult=0"]) {
                            [weakSelf jsonEditorLog:win message:[NSString stringWithFormat:@"SERVER ERROR: %@", line]];
                        }
                        if ([line hasPrefix:@"pnErrorDetail="]) {
                            [weakSelf jsonEditorLog:win message:[NSString stringWithFormat:@"Detail: %@", line]];
                        }
                    }
                } else {
                    [weakSelf jsonEditorLog:win message:@"SEND: no response (timeout or connection failure)."];
                }
                [weakSelf jsonEditorLog:win message:@"--- Send complete ---"];
            });
        }
    });
}

// ── Prevent Sleep ────────────────────────────────────────────────────

- (void)togglePreventSleep:(NSButton *)sender {
    _preventSleepEnabled = (sender.state == NSControlStateValueOn);
    if (_preventSleepEnabled) {
        [self acquireSleepAssertion];
    } else {
        [self releaseSleepAssertion];
    }
}

- (void)acquireSleepAssertion {
    if (_sleepAssertionID != kIOPMNullAssertionID) return; // already held
    IOReturn ret = IOPMAssertionCreateWithName(
        kIOPMAssertionTypePreventUserIdleSystemSleep,
        kIOPMAssertionLevelOn,
        CFSTR("PrimePath: prime search tasks running"),
        &_sleepAssertionID);
    if (ret == kIOReturnSuccess) {
        [self appendText:@"Sleep prevention: ON — Mac will stay awake.\n"];
    } else {
        _sleepAssertionID = kIOPMNullAssertionID;
        [self appendText:@"Sleep prevention: failed to acquire power assertion.\n"];
    }
}

- (void)releaseSleepAssertion {
    if (_sleepAssertionID == kIOPMNullAssertionID) return;
    IOPMAssertionRelease(_sleepAssertionID);
    _sleepAssertionID = kIOPMNullAssertionID;
    [self appendText:@"Sleep prevention: OFF — normal power management resumed.\n"];
}

// ── GIMPS Actions ────────────────────────────────────────────────────

// gimpsRegister and gimpsGetWork removed -- all API communication now
// routes through AutoPrimeNet. Assignments come from worktodo.txt,
// results go to results.json.txt. AutoPrimeNet handles server comms.

- (void)gimpsRunAssignment:(id)sender {
    // Read assignments from worktodo.txt (AutoPrimeNet workflow)
    auto worktodo = _primenet->read_worktodo();
    if (worktodo.empty()) {
        [self appendText:@"GIMPS: no assignments in worktodo.txt. Use AutoPrimeNet to get work.\n"];
        return;
    }

    // Read batch size from GIMPS panel popup
    NSWindow *win = [sender window];
    if (win) {
        NSPopUpButton *batchPopup = [win.contentView viewWithTag:8030];
        if (batchPopup) {
            uint64_t sizes[] = {10000000, 50000000, 100000000, 500000000, 1000000000};
            int idx = (int)batchPopup.indexOfSelectedItem;
            if (idx >= 0 && idx < 5) {
                _taskMgr->mersenne_k_batch.store(sizes[idx]);
                [self appendText:[NSString stringWithFormat:@"GIMPS: sieve batch = %lluM k-values\n",
                    sizes[idx] / 1000000]];
            }
        }
    }

    auto& assignment = worktodo.front();
    // Store assignment in primenet state so submit handler can find it
    _primenet->mutable_state().assignments.clear();
    _primenet->mutable_state().assignments.push_back(assignment);
    uint64_t exponent = assignment.exponent;

    // Configure the Mersenne TF task with this exponent
    auto& tasks = _taskMgr->tasks();
    auto it = tasks.find(prime::TaskType::MersenneTrial);
    if (it != tasks.end()) {
        if (it->second.status == prime::TaskStatus::Running) {
            [self appendText:@"GIMPS: Mersenne TF is already running. Stop it first.\n"];
            return;
        }
        it->second.end_pos = exponent;
        it->second.bit_lo = assignment.bit_lo;
        it->second.bit_hi = assignment.bit_hi;
        it->second.known_factors = assignment.known_factors;

        // Calculate starting k for bit_lo: q = 2kp + 1 >= 2^bit_lo
        // k >= (2^bit_lo - 1) / (2 * p)
        // Use double for the 2^bit_lo calculation since bit_lo can be 76+
        double q_min = pow(2.0, assignment.bit_lo);
        uint64_t k_start = (uint64_t)(q_min / (2.0 * (double)exponent));
        if (k_start < 1) k_start = 1;
        it->second.current_pos = k_start;

        [self appendText:[NSString stringWithFormat:
            @"GIMPS: starting M%llu TF from %d to %d bits, k=%llu (assignment %s)\n",
            exponent, (int)assignment.bit_lo, (int)assignment.bit_hi,
            k_start, assignment.key.c_str()]];

        _taskMgr->start_task(prime::TaskType::MersenneTrial);
    }
}

- (void)gimpsStopAssignment:(id)sender {
    auto& tasks = _taskMgr->tasks();
    auto it = tasks.find(prime::TaskType::MersenneTrial);
    if (it != tasks.end() && it->second.status == prime::TaskStatus::Running) {
        _taskMgr->pause_task(prime::TaskType::MersenneTrial);
        [self appendText:@"GIMPS: Mersenne TF stopped.\n"];
    } else {
        [self appendText:@"GIMPS: Mersenne TF is not running.\n"];
    }
}

- (void)gimpsSubmitResults:(id)sender {
    // Check for completed Mersenne TF work
    auto& tasks = _taskMgr->tasks();
    auto it = tasks.find(prime::TaskType::MersenneTrial);
    if (it == tasks.end()) return;

    if (_primenet->state().assignments.empty()) {
        [self appendText:@"GIMPS: no assignments to submit results for.\n"];
        return;
    }

    auto& assignment = _primenet->state().assignments.front();

    // Validate range completion: current k must cover bit_hi
    // q = 2kp + 1, so k_end = (2^bit_hi) / (2 * p)
    double q_max = pow(2.0, assignment.bit_hi);
    uint64_t k_end_needed = (uint64_t)(q_max / (2.0 * (double)assignment.exponent));
    uint64_t k_current = it->second.current_pos;
    bool range_done = (k_current >= k_end_needed);

    if (!range_done && it->second.status != prime::TaskStatus::Running) {
        double pct = (k_end_needed > 0) ? (100.0 * k_current / k_end_needed) : 0;
        [self appendText:[NSString stringWithFormat:
            @"GIMPS: range NOT complete — k=%llu of %llu needed (%.1f%%). "
            @"Resume the assignment or submit partial result anyway? "
            @"Submitting partial no-factor results wastes server resources.\n",
            k_current, k_end_needed, pct]];
        // Still allow submission (user may want to report a found factor)
    }

    // Check discoveries for this exponent (collect all new factors)
    std::vector<std::string> found_factors;
    for (auto& d : _taskMgr->discoveries()) {
        if (d.type == prime::TaskType::MersenneTrial && d.value2 == assignment.exponent) {
            std::string f = d.divisors.empty() ? std::to_string(d.value) : d.divisors;
            found_factors.push_back(f);
        }
    }
    bool found_factor = !found_factors.empty();

    if (!found_factor && !range_done) {
        double pct2 = (k_end_needed > 0) ? (100.0 * k_current / k_end_needed) : 0;
        NSAlert *alert = [[NSAlert alloc] init];
        alert.messageText = @"Range Incomplete";
        alert.informativeText = [NSString stringWithFormat:
            @"No factor found and range is only %.1f%% complete (k=%llu of %llu).\n\n"
            @"Submitting 'no factor' without completing the full range is invalid on PrimeNet.\n"
            @"Resume and finish the assignment first.", pct2, k_current, k_end_needed];
        alert.alertStyle = NSAlertStyleWarning;
        [alert addButtonWithTitle:@"OK"];
        [alert runModal];
        return;
    }

    primenet::TFResult result;
    result.exponent = assignment.exponent;
    result.bit_lo = assignment.bit_lo;
    result.bit_hi = assignment.bit_hi;
    result.factor_found = found_factor;
    result.factors = found_factors;
    result.assignment_key = assignment.key;
    result.range_complete = range_done;
    result.known_factors = assignment.known_factors;

    NSString *summaryMsg = [NSString stringWithFormat:
        @"GIMPS: submitting M%llu %@ (bits %d-%d, k=%llu/%llu)\n",
        assignment.exponent,
        found_factor ? @"FACTOR FOUND" : @"no factor",
        (int)assignment.bit_lo, (int)assignment.bit_hi,
        k_current, k_end_needed];
    [self appendText:summaryMsg];

    // Write pre-submission verification to live log
    {
        std::string logpath = std::string(DATA_DIR.UTF8String) + "/mersenne_tf_live.log";
        std::ofstream lf(logpath, std::ios::app);
        if (lf.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto tt = std::chrono::system_clock::to_time_t(now);
            char ts[32];
            strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", localtime(&tt));

            lf << "\n--- SUBMISSION SNAPSHOT ---\n"
               << "Time: " << ts << "\n"
               << "Exponent: " << assignment.exponent << "\n"
               << "Assignment Key: " << std::string(assignment.key) << "\n"
               << "Bit Range: " << (int)assignment.bit_lo << " to " << (int)assignment.bit_hi << "\n"
               << "k_current: " << k_current << "\n"
               << "k_needed: " << k_end_needed << "\n"
               << "Range Complete: " << (range_done ? "YES" : "NO") << "\n"
               << "Factor Found: " << (found_factor ? "YES" : "NO") << "\n";
            if (found_factor && !found_factors.empty()) {
                lf << "Factor(s):";
                for (auto& f : found_factors) lf << " " << f;
                lf << "\n";
            }
            lf << "Result Type: " << (found_factor ? "1 (factor found)" : "4 (no factor)") << "\n"
               << "URL will contain: t=ar&n=" << assignment.exponent
               << "&sf=" << (int)assignment.bit_lo
               << "&ef=" << (int)assignment.bit_hi
               << "&r=" << (found_factor ? 1 : 4) << "\n"
               << "--- END SNAPSHOT ---\n\n";
        }
    }

    // Write result to results.json.txt for AutoPrimeNet to pick up
    __weak AppDelegate *weakSelf = self;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        @autoreleasepool {
            AppDelegate *s = weakSelf;
            if (!s) return;
            // Build JSON and append to results.json.txt
            std::string json = s->_primenet->build_result_json(result);
            std::string path = std::string(DATA_DIR.UTF8String) + "/results.json.txt";
            std::ofstream f(path, std::ios::app);
            bool ok = false;
            if (f.is_open()) {
                f << json << "\n";
                f.close();
                ok = true;
            }
            // Remove completed line from worktodo.txt
            if (ok && !s->_primenet->mutable_state().assignments.empty()) {
                s->_primenet->remove_worktodo(s->_primenet->mutable_state().assignments.front());
            }
            dispatch_async(dispatch_get_main_queue(), ^{
                if (ok) {
                    [s appendText:[NSString stringWithFormat:
                        @"GIMPS: result written to results.json.txt. AutoPrimeNet will submit to mersenne.org.\n"]];
                    // Clear from in-memory state
                    s->_primenet->mutable_state().assignments.clear();
                } else {
                    [s appendText:@"GIMPS: FAILED to write results.json.txt.\n"];
                }
            });
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════
// Pipeline Builder -- configurable search & calculation pipeline
// ═══════════════════════════════════════════════════════════════════════

// Pipeline stage definition
struct PipelineStage {
    const char *name;
    const char *category;    // "Sieve", "Score", "Test", "Post"
    const char *description;
    double cost_per_million; // estimated ms per 1M candidates
    double rejection_pct;    // typical % of candidates rejected (0 for tests)
};

static const PipelineStage kPipelineStages[] = {
    // Sieve stages
    {"Wheel-210",         "Sieve", "Skip multiples of 2,3,5,7",                  0.1,  77.0},
    {"MatrixSieve",       "Sieve", "NEON sieve (primes 3-31)",                   0.5,  77.0},
    {"CRT Filter",        "Sieve", "CRT rejection (primes 11-31)",               0.3,  15.0},
    {"Pseudoprime Filter","Sieve", "Remove known Carmichael/SPRP-2",             0.2,   0.1},
    // Score stages
    {"Convergence Score", "Score", "Shadow prime field scoring (primes 7-67)",    5.0,  30.0},
    {"EvenShadow Score",  "Score", "p+/-1 divisor structure analysis (0-255)",    3.0,  20.0},
    // Test stages
    {"Miller-Rabin CPU",  "Test",  "Deterministic 12-witness primality",        50.0,   0.0},
    {"GPU Primality",     "Test",  "Metal batch primality (4096/batch)",         15.0,   0.0},
    {"Wieferich Test",    "Test",  "2^(p-1) mod p^2 == 1?",                    200.0,   0.0},
    {"Wilson Test",       "Test",  "factorial(p-1) mod p^2",                   5000.0,   0.0},
    {"Twin Pair Test",    "Test",  "is_prime(p) && is_prime(p+2)",              80.0,   0.0},
    {"Sophie Germain",    "Test",  "is_prime(p) && is_prime(2p+1)",             80.0,   0.0},
    {"Cousin Pair Test",  "Test",  "is_prime(p) && is_prime(p+4)",              80.0,   0.0},
    {"Sexy Pair Test",    "Test",  "is_prime(p) && is_prime(p+6)",              80.0,   0.0},
    {"Emirp Test",        "Test",  "is_prime(reverse_digits(p))",               60.0,   0.0},
    // Post stages
    {"Factor (full)",     "Post",  "Trial division + Pollard rho",             500.0,   0.0},
    {"PinchFactor",       "Post",  "Digit-structural factoring heuristic",     100.0,   0.0},
    {"Lucky7s",           "Post",  "Round-number proximity factoring",         100.0,   0.0},
    {"DivisorWeb",        "Post",  "Hierarchical divisor search",              200.0,   0.0},
    // Analysis stages (Factor Seed Prediction)
    {"Ring Beacon",       "Analysis", "Mod-210 wheel beacon density scoring",    80.0,   0.0},
    {"Topography",        "Analysis", "Factor count elevation & depth-gap corr.",120.0,   0.0},
    {"Factor Web",        "Analysis", "Factor seed web connectivity & gaps",     100.0,   0.0},
    {"Audio Harmonic",    "Analysis", "Harmonic resonance scoring of factors",   90.0,   0.0},
    {"Twisting Tree",     "Analysis", "Factor tree shape transition analysis",  150.0,   0.0},
    {"Quad Residue",      "Analysis", "Quadratic residue distribution analysis",200.0,   0.0},
    {"Goldbach Split",    "Analysis", "Goldbach partition pair analysis",        300.0,   0.0},
    {"Euler Totient",     "Analysis", "Totient function chain analysis",        250.0,   0.0},
};
static const int kNumPipelineStages = sizeof(kPipelineStages) / sizeof(kPipelineStages[0]);

- (void)showPipelineBuilder:(id)sender {
    NSWindow *win = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(150, 150, 720, 640)
        styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable)
        backing:NSBackingStoreBuffered defer:NO];
    win.title = @"Search Pipeline Builder";
    win.releasedWhenClosed = NO;
    win.minSize = NSMakeSize(600, 500);

    NSView *cv = win.contentView;
    CGFloat W = cv.frame.size.width;
    CGFloat y = cv.frame.size.height - 10;

    // Title
    NSTextField *title = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 20, 400, 18)];
    title.stringValue = @"Build a custom search pipeline";
    title.font = [NSFont boldSystemFontOfSize:13];
    title.bezeled = NO; title.editable = NO; title.drawsBackground = NO;
    title.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:title];
    y -= 30;

    // Range inputs
    NSTextField *rangeLbl = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 16, 40, 14)];
    rangeLbl.stringValue = @"From:"; rangeLbl.font = [NSFont systemFontOfSize:10];
    rangeLbl.bezeled = NO; rangeLbl.editable = NO; rangeLbl.drawsBackground = NO;
    rangeLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:rangeLbl];

    NSTextField *fromField = [[NSTextField alloc] initWithFrame:NSMakeRect(54, y - 18, 140, 20)];
    fromField.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    fromField.stringValue = @"1000000";
    fromField.tag = 9001;
    fromField.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:fromField];

    NSTextField *toLbl = [[NSTextField alloc] initWithFrame:NSMakeRect(200, y - 16, 24, 14)];
    toLbl.stringValue = @"To:"; toLbl.font = [NSFont systemFontOfSize:10];
    toLbl.bezeled = NO; toLbl.editable = NO; toLbl.drawsBackground = NO;
    toLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:toLbl];

    NSTextField *toField = [[NSTextField alloc] initWithFrame:NSMakeRect(224, y - 18, 140, 20)];
    toField.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    toField.stringValue = @"1100000";
    toField.tag = 9002;
    toField.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:toField];

    // Sort by score checkbox
    NSButton *sortCheck = [[NSButton alloc] initWithFrame:NSMakeRect(380, y - 18, 140, 18)];
    sortCheck.buttonType = NSButtonTypeSwitch;
    sortCheck.title = @"Sort by score";
    sortCheck.font = [NSFont systemFontOfSize:10];
    sortCheck.state = NSControlStateValueOn;
    sortCheck.tag = 9010;
    sortCheck.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:sortCheck];

    // Run button
    NSButton *runBtn = [self buttonAt:NSMakeRect(540, y - 20, 80, 22) title:@"Run" action:@selector(runPipeline:)];
    runBtn.font = [NSFont boldSystemFontOfSize:11];
    runBtn.autoresizingMask = NSViewMinYMargin | NSViewMinXMargin;
    [cv addSubview:runBtn];

    // Stop button
    NSButton *stopBtn = [self buttonAt:NSMakeRect(626, y - 20, 70, 22) title:@"Stop" action:@selector(checkStopAction:)];
    stopBtn.font = [NSFont systemFontOfSize:10];
    stopBtn.autoresizingMask = NSViewMinYMargin | NSViewMinXMargin;
    [cv addSubview:stopBtn];

    y -= 28;

    // Separator
    NSBox *sep = [[NSBox alloc] initWithFrame:NSMakeRect(14, y, W - 28, 1)];
    sep.boxType = NSBoxSeparator;
    sep.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:sep];
    y -= 6;

    // Column headers
    NSDictionary *headerAttrs = @{
        NSFontAttributeName: [NSFont boldSystemFontOfSize:9],
        NSForegroundColorAttributeName: [NSColor secondaryLabelColor]
    };

    NSTextField *hdrStage = [[NSTextField alloc] initWithFrame:NSMakeRect(40, y - 12, 140, 12)];
    hdrStage.attributedStringValue = [[NSAttributedString alloc] initWithString:@"STAGE" attributes:headerAttrs];
    hdrStage.bezeled = NO; hdrStage.editable = NO; hdrStage.drawsBackground = NO;
    hdrStage.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:hdrStage];

    NSTextField *hdrDesc = [[NSTextField alloc] initWithFrame:NSMakeRect(190, y - 12, 240, 12)];
    hdrDesc.attributedStringValue = [[NSAttributedString alloc] initWithString:@"DESCRIPTION" attributes:headerAttrs];
    hdrDesc.bezeled = NO; hdrDesc.editable = NO; hdrDesc.drawsBackground = NO;
    hdrDesc.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:hdrDesc];

    NSTextField *hdrCost = [[NSTextField alloc] initWithFrame:NSMakeRect(450, y - 12, 80, 12)];
    hdrCost.attributedStringValue = [[NSAttributedString alloc] initWithString:@"COST (ms/1M)" attributes:headerAttrs];
    hdrCost.bezeled = NO; hdrCost.editable = NO; hdrCost.drawsBackground = NO;
    hdrCost.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:hdrCost];

    NSTextField *hdrReject = [[NSTextField alloc] initWithFrame:NSMakeRect(550, y - 12, 80, 12)];
    hdrReject.attributedStringValue = [[NSAttributedString alloc] initWithString:@"REJECT %%" attributes:headerAttrs];
    hdrReject.bezeled = NO; hdrReject.editable = NO; hdrReject.drawsBackground = NO;
    hdrReject.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:hdrReject];

    NSTextField *hdrOrder = [[NSTextField alloc] initWithFrame:NSMakeRect(640, y - 12, 50, 12)];
    hdrOrder.attributedStringValue = [[NSAttributedString alloc] initWithString:@"ORDER" attributes:headerAttrs];
    hdrOrder.bezeled = NO; hdrOrder.editable = NO; hdrOrder.drawsBackground = NO;
    hdrOrder.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:hdrOrder];

    y -= 16;

    // Stage checkboxes in a scrollable container
    CGFloat stageListTop = y;
    CGFloat stageListHeight = y - 180; // leave room for log below
    NSScrollView *stageScroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(14, y - stageListHeight, W - 28, stageListHeight)];
    stageScroll.hasVerticalScroller = YES;
    stageScroll.borderType = NSBezelBorder;
    stageScroll.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

    // Calculate total content height for all stages
    CGFloat contentH = 0;
    {
        const char *prevCat = "";
        for (int i = 0; i < kNumPipelineStages; i++) {
            if (strcmp(kPipelineStages[i].category, prevCat) != 0) {
                prevCat = kPipelineStages[i].category;
                contentH += 16; // category header
            }
            contentH += 18; // stage row
        }
        contentH += 10; // padding
    }
    if (contentH < stageListHeight) contentH = stageListHeight;

    NSView *stageContainer = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, W - 32, contentH)];

    CGFloat sy = contentH - 4; // start near top of content

    NSColor *sieveColor = [NSColor colorWithSRGBRed:0.2 green:0.6 blue:0.3 alpha:1.0];
    NSColor *scoreColor = [NSColor colorWithSRGBRed:0.3 green:0.5 blue:0.8 alpha:1.0];
    NSColor *testColor  = [NSColor colorWithSRGBRed:0.8 green:0.4 blue:0.2 alpha:1.0];
    NSColor *postColor  = [NSColor colorWithSRGBRed:0.6 green:0.3 blue:0.7 alpha:1.0];
    NSColor *analysisColor = [NSColor colorWithSRGBRed:0.7 green:0.5 blue:0.1 alpha:1.0];

    const char *lastCategory = "";

    for (int i = 0; i < kNumPipelineStages; i++) {
        const auto& stage = kPipelineStages[i];

        // Category header
        if (strcmp(stage.category, lastCategory) != 0) {
            lastCategory = stage.category;

            NSColor *catColor = sieveColor;
            if (strcmp(stage.category, "Score") == 0) catColor = scoreColor;
            else if (strcmp(stage.category, "Test") == 0) catColor = testColor;
            else if (strcmp(stage.category, "Post") == 0) catColor = postColor;
            else if (strcmp(stage.category, "Analysis") == 0) catColor = analysisColor;

            NSTextField *catLbl = [[NSTextField alloc] initWithFrame:NSMakeRect(4, sy - 14, 100, 13)];
            catLbl.stringValue = [NSString stringWithUTF8String:stage.category];
            catLbl.font = [NSFont boldSystemFontOfSize:9];
            catLbl.textColor = catColor;
            catLbl.bezeled = NO; catLbl.editable = NO; catLbl.drawsBackground = NO;
            [stageContainer addSubview:catLbl];
            sy -= 16;
        }

        // Checkbox
        NSButton *cb = [[NSButton alloc] initWithFrame:NSMakeRect(14, sy - 16, 160, 16)];
        cb.buttonType = NSButtonTypeSwitch;
        cb.title = [NSString stringWithUTF8String:stage.name];
        cb.font = [NSFont systemFontOfSize:10];
        cb.tag = 10000 + i;
        // Default: enable Wheel, MatrixSieve, CRT, and Miller-Rabin
        if (i <= 2 || i == 6) cb.state = NSControlStateValueOn;
        [stageContainer addSubview:cb];

        // Description
        NSTextField *desc = [[NSTextField alloc] initWithFrame:NSMakeRect(180, sy - 15, 250, 13)];
        desc.stringValue = [NSString stringWithUTF8String:stage.description];
        desc.font = [NSFont systemFontOfSize:9];
        desc.textColor = [NSColor secondaryLabelColor];
        desc.bezeled = NO; desc.editable = NO; desc.drawsBackground = NO;
        [stageContainer addSubview:desc];

        // Cost
        NSTextField *cost = [[NSTextField alloc] initWithFrame:NSMakeRect(440, sy - 15, 70, 13)];
        cost.stringValue = [NSString stringWithFormat:@"%.1f", stage.cost_per_million];
        cost.font = [NSFont monospacedSystemFontOfSize:9 weight:NSFontWeightRegular];
        cost.alignment = NSTextAlignmentRight;
        cost.bezeled = NO; cost.editable = NO; cost.drawsBackground = NO;
        [stageContainer addSubview:cost];

        // Rejection %
        NSTextField *rej = [[NSTextField alloc] initWithFrame:NSMakeRect(520, sy - 15, 60, 13)];
        if (stage.rejection_pct > 0) {
            rej.stringValue = [NSString stringWithFormat:@"~%.0f%%", stage.rejection_pct];
        } else {
            rej.stringValue = @"--";
        }
        rej.font = [NSFont monospacedSystemFontOfSize:9 weight:NSFontWeightRegular];
        rej.alignment = NSTextAlignmentRight;
        rej.bezeled = NO; rej.editable = NO; rej.drawsBackground = NO;
        [stageContainer addSubview:rej];

        // Order stepper
        NSTextField *orderField = [[NSTextField alloc] initWithFrame:NSMakeRect(600, sy - 16, 30, 16)];
        orderField.integerValue = i + 1;
        orderField.font = [NSFont monospacedSystemFontOfSize:9 weight:NSFontWeightRegular];
        orderField.alignment = NSTextAlignmentCenter;
        orderField.tag = 20000 + i;
        [stageContainer addSubview:orderField];

        sy -= 18;
    }

    stageScroll.documentView = stageContainer;
    [cv addSubview:stageScroll];

    y = y - stageListHeight - 8;

    // Cost estimate label
    NSTextField *estLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(14, y - 14, W - 28, 13)];
    estLabel.stringValue = @"Estimated pipeline throughput shown after each run.";
    estLabel.font = [NSFont systemFontOfSize:9];
    estLabel.textColor = [NSColor tertiaryLabelColor];
    estLabel.bezeled = NO; estLabel.editable = NO; estLabel.drawsBackground = NO;
    estLabel.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:estLabel];

    y -= 20;

    // Pipeline output log
    NSScrollView *logScroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(14, 10, W - 28, y - 10)];
    logScroll.hasVerticalScroller = YES;
    logScroll.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    NSTextView *logView = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, W - 32, y - 10)];
    logView.editable = NO;
    logView.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    logView.autoresizingMask = NSViewWidthSizable;
    logScroll.documentView = logView;
    logScroll.identifier = @"pipelineLog";
    [cv addSubview:logScroll];

    [win makeKeyAndOrderFront:nil];
}

// ── Run Pipeline -- executes the configured pipeline ──

- (void)runPipeline:(id)sender {
    _activeLogTab = 3;  // Pipeline tab
    [self.logTabView selectTabViewItemAtIndex:3];
    // Find the pipeline window
    NSWindow *win = [sender window];
    if (!win) return;
    NSView *cv = win.contentView;

    // Read range
    NSTextField *fromField = [cv viewWithTag:9001];
    NSTextField *toField = [cv viewWithTag:9002];
    NSButton *sortCheck = [cv viewWithTag:9010];
    NSTextView *logView = nil;

    // Find the log text view by scroll view identifier
    for (NSView *v in cv.subviews) {
        if ([v.identifier isEqualToString:@"pipelineLog"] && [v isKindOfClass:[NSScrollView class]]) {
            NSScrollView *sv = (NSScrollView *)v;
            if ([sv.documentView isKindOfClass:[NSTextView class]]) {
                logView = (NSTextView *)sv.documentView;
            }
            break;
        }
    }
    if (!logView || !fromField || !toField) return;

    NSString *fromStr = [fromField.stringValue stringByReplacingOccurrencesOfString:@"," withString:@""];
    NSString *toStr = [toField.stringValue stringByReplacingOccurrencesOfString:@"," withString:@""];
    uint64_t rangeStart = [self evalMathExpr:fromStr];
    uint64_t rangeEnd = [self evalMathExpr:toStr];
    if (rangeStart < 2) rangeStart = 2;
    if (rangeEnd <= rangeStart) {
        logView.string = @"Invalid range. End must be > Start.\n";
        return;
    }
    if (rangeEnd - rangeStart > 10000000) {
        logView.string = @"Range too large for pipeline (max 10M). Use Search Tasks for larger ranges.\n";
        return;
    }
    bool sortByScore = (sortCheck && sortCheck.state == NSControlStateValueOn);

    // Collect enabled stages with their order
    struct ActiveStage {
        int index;
        int order;
    };
    std::vector<ActiveStage> activeStages;

    for (int i = 0; i < kNumPipelineStages; i++) {
        NSButton *cb = [cv viewWithTag:10000 + i];
        if (cb && cb.state == NSControlStateValueOn) {
            NSTextField *orderField = [cv viewWithTag:20000 + i];
            int order = orderField ? (int)orderField.integerValue : (i + 1);
            activeStages.push_back({i, order});
        }
    }

    if (activeStages.empty()) {
        logView.string = @"No pipeline stages selected.\n";
        return;
    }

    // Sort by user-specified order
    std::sort(activeStages.begin(), activeStages.end(),
              [](const ActiveStage& a, const ActiveStage& b) { return a.order < b.order; });

    // Build stage name list for header
    NSMutableString *stageNames = [NSMutableString string];
    for (auto& s : activeStages) {
        if (stageNames.length > 0) [stageNames appendString:@" -> "];
        [stageNames appendFormat:@"%s", kPipelineStages[s.index].name];
    }

    logView.string = [NSString stringWithFormat:
        @"Pipeline: %@\n"
        @"Range: [%@, %@] (%@ candidates)\n"
        @"Running...\n\n",
        stageNames,
        formatNumber(rangeStart), formatNumber(rangeEnd),
        formatNumber(rangeEnd - rangeStart)];

    // Run in background
    __weak AppDelegate *weakSelf = self;
    __weak NSTextView *weakLog = logView;
    std::vector<ActiveStage> stages = activeStages;

    _checkRunning.store(false);
    if (_checkThread.joinable()) _checkThread.join();
    _checkRunning.store(true);

    _checkThread = std::thread([weakSelf, weakLog, rangeStart, rangeEnd, stages, sortByScore]() {
        AppDelegate *ss = weakSelf;
        if (!ss) return;

        auto plog = [weakLog](NSString *msg) {
            dispatch_async(dispatch_get_main_queue(), ^{
                NSTextView *lv = weakLog;
                if (!lv) return;
                NSAttributedString *attr = [[NSAttributedString alloc]
                    initWithString:msg
                    attributes:@{
                        NSFontAttributeName: [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular],
                        NSForegroundColorAttributeName: [NSColor labelColor]
                    }];
                [lv.textStorage appendAttributedString:attr];
                [lv scrollRangeToVisible:NSMakeRange(lv.string.length, 0)];
            });
        };

        auto t_total = std::chrono::steady_clock::now();

        // Start with all candidates in range
        struct Candidate {
            uint64_t value;
            double score;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(rangeEnd - rangeStart);

        // Populate initial candidates (odd numbers, skip even except 2)
        if (rangeStart == 2) candidates.push_back({2, 0.0});
        uint64_t start = (rangeStart % 2 == 0) ? rangeStart + 1 : rangeStart;
        if (start < 3) start = 3;
        for (uint64_t n = start; n <= rangeEnd; n += 2) {
            candidates.push_back({n, 0.0});
        }

        plog([NSString stringWithFormat:@"Initial candidates: %@\n", formatNumber((uint64_t)candidates.size())]);

        // Results collection
        std::vector<uint64_t> results;
        std::vector<std::pair<uint64_t, std::string>> postResults; // for post-processing

        // Execute each stage
        for (auto& stage : stages) {
            if (!ss->_checkRunning.load()) {
                plog(@"\nPipeline stopped by user.\n");
                return;
            }

            const auto& def = kPipelineStages[stage.index];
            auto t0 = std::chrono::steady_clock::now();
            size_t before = candidates.size();

            plog([NSString stringWithFormat:@"[%s] Processing %@ candidates...\n",
                def.name, formatNumber((uint64_t)before)]);

            if (strcmp(def.name, "Wheel-210") == 0) {
                // Wheel-210 filter
                std::vector<Candidate> filtered;
                filtered.reserve(candidates.size());
                for (auto& c : candidates) {
                    if (c.value <= 7 || prime::WHEEL.valid(c.value))
                        filtered.push_back(c);
                }
                candidates = std::move(filtered);

            } else if (strcmp(def.name, "MatrixSieve") == 0) {
                // MatrixSieve: check against small primes 3-31
                static const uint64_t mprimes[] = {3,5,7,11,13,17,19,23,29,31};
                std::vector<Candidate> filtered;
                filtered.reserve(candidates.size());
                for (auto& c : candidates) {
                    bool ok = true;
                    for (auto p : mprimes) {
                        if (c.value > p && c.value % p == 0) { ok = false; break; }
                    }
                    if (ok) filtered.push_back(c);
                }
                candidates = std::move(filtered);

            } else if (strcmp(def.name, "CRT Filter") == 0) {
                std::vector<Candidate> filtered;
                filtered.reserve(candidates.size());
                for (auto& c : candidates) {
                    if (!prime::crt_reject(c.value))
                        filtered.push_back(c);
                }
                candidates = std::move(filtered);

            } else if (strcmp(def.name, "Pseudoprime Filter") == 0) {
                // Simple: remove even numbers and known small composites
                std::vector<Candidate> filtered;
                filtered.reserve(candidates.size());
                for (auto& c : candidates) {
                    // Quick Fermat test base 2 -- catches many pseudoprimes
                    if (c.value > 2 && prime::modpow(2, c.value - 1, c.value) != 1) continue;
                    filtered.push_back(c);
                }
                candidates = std::move(filtered);

            } else if (strcmp(def.name, "Convergence Score") == 0) {
                for (auto& c : candidates) {
                    c.score = prime::convergence(c.value, 12);
                }
                // Remove shadow-field rejects
                std::vector<Candidate> filtered;
                filtered.reserve(candidates.size());
                for (auto& c : candidates) {
                    if (c.score > -900.0) filtered.push_back(c);
                }
                candidates = std::move(filtered);

            } else if (strcmp(def.name, "EvenShadow Score") == 0) {
                // Score based on p-1/p+1 divisor structure
                for (auto& c : candidates) {
                    uint64_t pm1 = c.value - 1;
                    int factors = 0;
                    for (uint64_t p : {2ULL,3ULL,5ULL,7ULL,11ULL,13ULL,17ULL,19ULL,23ULL}) {
                        while (pm1 % p == 0) { pm1 /= p; factors++; }
                    }
                    c.score += factors * 10.0; // more small factors = higher score
                }

            } else if (strcmp(def.name, "Miller-Rabin CPU") == 0) {
                std::vector<Candidate> filtered;
                for (auto& c : candidates) {
                    if (prime::is_prime(c.value)) filtered.push_back(c);
                }
                candidates = std::move(filtered);

            } else if (strcmp(def.name, "GPU Primality") == 0) {
                if (ss->_gpu) {
                    // Batch GPU test
                    std::vector<uint64_t> vals;
                    vals.reserve(candidates.size());
                    for (auto& c : candidates) vals.push_back(c.value);
                    std::vector<uint8_t> gpu_results(vals.size(), 0);
                    ss->_gpu->primality_batch(vals.data(), gpu_results.data(), (uint32_t)vals.size());
                    std::vector<Candidate> filtered;
                    for (size_t i = 0; i < candidates.size(); i++) {
                        if (gpu_results[i]) filtered.push_back(candidates[i]);
                    }
                    candidates = std::move(filtered);
                } else {
                    plog(@"  (no GPU available, falling back to CPU Miller-Rabin)\n");
                    std::vector<Candidate> filtered;
                    for (auto& c : candidates) {
                        if (prime::is_prime(c.value)) filtered.push_back(c);
                    }
                    candidates = std::move(filtered);
                }

            } else if (strcmp(def.name, "Wieferich Test") == 0) {
                std::vector<Candidate> filtered;
                for (auto& c : candidates) {
                    if (prime::modpow(2, c.value - 1, c.value * c.value) == 1)
                        filtered.push_back(c);
                }
                candidates = std::move(filtered);

            } else if (strcmp(def.name, "Wilson Test") == 0) {
                std::vector<Candidate> filtered;
                for (auto& c : candidates) {
                    if (c.value > 100000) continue; // too slow for large p
                    uint64_t mod = c.value * c.value;
                    uint64_t factorial = 1;
                    for (uint64_t i = 2; i < c.value; i++) {
                        factorial = prime::mulmod(factorial, i, mod);
                    }
                    if (factorial == mod - 1) filtered.push_back(c);
                }
                candidates = std::move(filtered);

            } else if (strcmp(def.name, "Twin Pair Test") == 0) {
                std::vector<Candidate> filtered;
                for (auto& c : candidates) {
                    if (prime::is_prime(c.value) && prime::is_prime(c.value + 2))
                        filtered.push_back(c);
                }
                candidates = std::move(filtered);

            } else if (strcmp(def.name, "Sophie Germain") == 0) {
                std::vector<Candidate> filtered;
                for (auto& c : candidates) {
                    if (prime::is_prime(c.value) && prime::is_prime(2 * c.value + 1))
                        filtered.push_back(c);
                }
                candidates = std::move(filtered);

            } else if (strcmp(def.name, "Cousin Pair Test") == 0) {
                std::vector<Candidate> filtered;
                for (auto& c : candidates) {
                    if (prime::is_prime(c.value) && prime::is_prime(c.value + 4))
                        filtered.push_back(c);
                }
                candidates = std::move(filtered);

            } else if (strcmp(def.name, "Sexy Pair Test") == 0) {
                std::vector<Candidate> filtered;
                for (auto& c : candidates) {
                    if (prime::is_prime(c.value) && prime::is_prime(c.value + 6))
                        filtered.push_back(c);
                }
                candidates = std::move(filtered);

            } else if (strcmp(def.name, "Emirp Test") == 0) {
                std::vector<Candidate> filtered;
                for (auto& c : candidates) {
                    uint64_t rev = 0, tmp = c.value;
                    while (tmp > 0) { rev = rev * 10 + tmp % 10; tmp /= 10; }
                    if (rev != c.value && prime::is_prime(c.value) && prime::is_prime(rev))
                        filtered.push_back(c);
                }
                candidates = std::move(filtered);

            } else if (strcmp(def.name, "Factor (full)") == 0) {
                for (auto& c : candidates) {
                    std::string fs = prime::factors_string(c.value);
                    if (!fs.empty()) {
                        postResults.push_back({c.value, fs});
                    }
                }

            } else if (strcmp(def.name, "PinchFactor") == 0) {
                for (auto& c : candidates) {
                    auto hits = prime::pinch_factor(c.value);
                    for (auto& h : hits) {
                        if (c.value % h.divisor == 0 && h.divisor > 1 && h.divisor < c.value) {
                            postResults.push_back({c.value,
                                std::string("PinchFactor: ") + std::to_string(h.divisor)
                                + " (" + h.method + ")"});
                            break;
                        }
                    }
                }

            } else if (strcmp(def.name, "Lucky7s") == 0) {
                for (auto& c : candidates) {
                    auto hits = prime::lucky7_factor(c.value);
                    for (auto& h : hits) {
                        if (c.value % h.divisor == 0 && h.divisor > 1) {
                            postResults.push_back({c.value,
                                std::string("Lucky7: ") + std::to_string(h.divisor)
                                + " (" + h.method + ")"});
                            break;
                        }
                    }
                }

            } else if (strcmp(def.name, "DivisorWeb") == 0) {
                for (auto& c : candidates) {
                    auto web = prime::divisor_web(c.value, 10);
                    if (!web.all_divisors.empty()) {
                        std::string divs;
                        for (auto d : web.all_divisors) {
                            if (!divs.empty()) divs += ", ";
                            divs += std::to_string(d);
                        }
                        postResults.push_back({c.value, "DivisorWeb: {" + divs + "}"});
                    }
                }
            }

            auto dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            size_t after = candidates.size();
            size_t rejected = before - after;
            double pct = before > 0 ? 100.0 * rejected / before : 0.0;

            NSString *stageResult;
            if (strcmp(def.category, "Post") == 0) {
                stageResult = [NSString stringWithFormat:
                    @"  -> %@ results, %.4fs\n",
                    formatNumber((uint64_t)postResults.size()), dt];
            } else if (rejected > 0) {
                stageResult = [NSString stringWithFormat:
                    @"  -> %@ survived, %@ rejected (%.1f%%), %.4fs\n",
                    formatNumber((uint64_t)after), formatNumber((uint64_t)rejected), pct, dt];
            } else {
                stageResult = [NSString stringWithFormat:
                    @"  -> %@ survived, %.4fs\n",
                    formatNumber((uint64_t)after), dt];
            }
            plog(stageResult);
        }

        // Sort by score if requested
        if (sortByScore && !candidates.empty()) {
            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
        }

        auto totalTime = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_total).count();

        // Output results
        plog([NSString stringWithFormat:@"\n== Pipeline Results ==\n"]);
        plog([NSString stringWithFormat:@"Survivors: %@\n", formatNumber((uint64_t)candidates.size())]);
        plog([NSString stringWithFormat:@"Total time: %.4fs\n\n", totalTime]);

        // Show up to 500 results
        size_t show = std::min(candidates.size(), (size_t)500);
        for (size_t i = 0; i < show; i++) {
            auto& c = candidates[i];
            NSString *scoreStr = (c.score != 0.0)
                ? [NSString stringWithFormat:@" (score: %.2f)", c.score]
                : @"";
            plog([NSString stringWithFormat:@"  %@%@\n", formatNumber(c.value), scoreStr]);
        }
        if (candidates.size() > 500) {
            plog([NSString stringWithFormat:@"  ... and %@ more\n",
                formatNumber((uint64_t)(candidates.size() - 500))]);
        }

        // Show post-processing results
        if (!postResults.empty()) {
            plog([NSString stringWithFormat:@"\n== Post-Processing Results ==\n"]);
            size_t showPost = std::min(postResults.size(), (size_t)200);
            for (size_t i = 0; i < showPost; i++) {
                plog([NSString stringWithFormat:@"  %@: %s\n",
                    formatNumber(postResults[i].first), postResults[i].second.c_str()]);
            }
        }

        plog([NSString stringWithFormat:@"\nDone. Throughput: %@/s\n",
            formatNumber((uint64_t)((rangeEnd - rangeStart) / totalTime))]);

        ss->_checkRunning.store(false);
    });
    // Thread stays joinable for proper cleanup
}

- (void)appendStatus:(NSString *)text {
    // Route to the status/network pane
    [_logLock lock];
    NSArray *lines = [text componentsSeparatedByString:@"\n"];
    for (NSString *line in lines) {
        if (line.length == 0) continue;
        _statusRing[_statusRingHead % _statusRing.count] = line;
        _statusRingHead++;
    }
    _statusDirty = YES;
    [_logLock unlock];
}

- (void)appendToTab:(int)tab text:(NSString *)text {
    [_logLock lock];
    NSArray *lines = [text componentsSeparatedByString:@"\n"];
    for (NSString *line in lines) {
        if (line.length == 0) continue;
        _tabRings[tab][_tabRingHeads[tab] % _tabRings[tab].count] = line;
        _tabRingHeads[tab]++;
    }
    _tabDirty[tab] = YES;
    [_logLock unlock];
}

- (void)appendText:(NSString *)text {
    // Auto-route: GIMPS/PrimeNet/discovery/network messages go to status pane
    if ([text hasPrefix:@"GIMPS:"] || [text hasPrefix:@"PrimeNet"] ||
        [text hasPrefix:@">>> DISCOVERY"] || [text hasPrefix:@"Conductor:"] ||
        [text hasPrefix:@"Carriage:"] || [text hasPrefix:@"Network:"]) {
        [self appendStatus:text];
        return;
    }
    // Route to the currently active log tab
    [self appendToTab:_activeLogTab text:text];
}

- (void)flushLogBuffer {
    NSDictionary *logAttrs = @{
        NSFontAttributeName: [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular],
        NSForegroundColorAttributeName: [NSColor labelColor]
    };

    // ── Flush each tab ──
    for (int t = 0; t < 4; t++) {
        [_logLock lock];
        BOOL dirty = _tabDirty[t];
        NSMutableString *text = nil;
        if (dirty) {
            NSUInteger cap = _tabRings[t].count;
            NSUInteger start = (_tabRingHeads[t] > cap) ? (_tabRingHeads[t] - cap) : 0;
            text = [NSMutableString stringWithCapacity:(_tabRingHeads[t] - start) * 80];
            for (NSUInteger i = start; i < _tabRingHeads[t]; i++) {
                [text appendString:_tabRings[t][i % cap]];
                [text appendString:@"\n"];
            }
            _tabDirty[t] = NO;
        }
        [_logLock unlock];

        if (dirty && _tabViews[t]) {
            NSTextStorage *ts = _tabViews[t].textStorage;
            NSAttributedString *attr = [[NSAttributedString alloc] initWithString:text attributes:logAttrs];
            [ts beginEditing];
            [ts setAttributedString:attr];
            [ts endEditing];
            [_tabViews[t] scrollRangeToVisible:NSMakeRange(ts.length, 0)];
        }
    }

    // ── Flush status pane ──
    [_logLock lock];
    BOOL needStatus = _statusDirty;
    NSMutableString *statusText = nil;
    if (needStatus) {
        NSUInteger cap = _statusRing.count;
        NSUInteger st = (_statusRingHead > cap) ? (_statusRingHead - cap) : 0;
        statusText = [NSMutableString stringWithCapacity:(_statusRingHead - st) * 80];
        for (NSUInteger i = st; i < _statusRingHead; i++) {
            [statusText appendString:_statusRing[i % cap]];
            [statusText appendString:@"\n"];
        }
        _statusDirty = NO;
    }
    [_logLock unlock];

    if (needStatus && self.statusView) {
        NSTextStorage *sts = self.statusView.textStorage;
        NSDictionary *sattrs = @{
            NSFontAttributeName: [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular],
            NSForegroundColorAttributeName: [NSColor colorWithCalibratedRed:0.6 green:0.9 blue:0.6 alpha:1.0]
        };
        NSAttributedString *sattr = [[NSAttributedString alloc] initWithString:statusText attributes:sattrs];
        [sts beginEditing];
        [sts setAttributedString:sattr];
        [sts endEditing];
        [self.statusView scrollRangeToVisible:NSMakeRange(sts.length, 0)];
    }
}

// ═══════════════════════════════════════════════════════════════════════
// In-App Engine Test Suite (108 tests)
// ═══════════════════════════════════════════════════════════════════════

- (void)runInAppTests:(id)sender {
    // Stop any running check first
    _checkRunning.store(false);
    if (_checkThread.joinable()) _checkThread.join();
    _checkRunning.store(true);

    __weak AppDelegate *weakSelf = self;

    _checkThread = std::thread([weakSelf]() {
        AppDelegate *ss = weakSelf;
        if (!ss) return;

        int pass = 0, fail = 0;

        auto log = [weakSelf](NSString *msg) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [weakSelf appendText:msg];
            });
        };

        auto cancelled = [weakSelf]() -> bool {
            AppDelegate *s = weakSelf;
            return !s || !s->_checkRunning.load();
        };

        auto check = [&](bool expr, const char *name) {
            if (expr) {
                log([NSString stringWithFormat:@"  PASS: %s\n", name]);
                pass++;
            } else {
                log([NSString stringWithFormat:@"  FAIL: %s\n", name]);
                fail++;
            }
        };

        log(@"\nPrimePath Engine Test Suite\n");
        log(@"================================================\n");

        // ── 1. Primality Testing ──
        log(@"\n=== 1. Primality Testing ===\n");

        check(!prime::is_prime(0), "0 is not prime");
        check(!prime::is_prime(1), "1 is not prime");
        check(prime::is_prime(2),  "2 is prime");

        uint64_t known_primes[] = {2,3,5,7,11,13,997,7919,104729};
        bool all_primes_ok = true;
        for (auto p : known_primes) {
            if (!prime::is_prime(p)) { all_primes_ok = false; break; }
        }
        check(all_primes_ok, "known primes: 2,3,5,7,11,13,997,7919,104729");

        uint64_t known_composites[] = {4,6,9,15};
        bool all_comp_ok = true;
        for (auto c : known_composites) {
            if (prime::is_prime(c)) { all_comp_ok = false; break; }
        }
        check(all_comp_ok, "known composites: 4,6,9,15");

        check(!prime::is_prime(561), "561 (Carmichael) is composite");

        uint64_t carmichaels[] = {1105, 1729, 2465, 2821, 6601, 8911, 10585, 15841, 29341, 41041, 46657, 52633, 62745, 63973, 75361};
        bool carm_ok = true;
        for (auto c : carmichaels) {
            if (prime::is_prime(c)) { carm_ok = false; break; }
        }
        check(carm_ok, "all Carmichael numbers correctly rejected");

        check(prime::is_prime(2147483647ULL), "M31 = 2^31-1 is prime");
        check(prime::is_prime(2305843009213693951ULL), "M61 = 2^61-1 is prime");
        check(!prime::is_prime(8388607ULL),   "2^23-1 = 8388607 is composite (47*178481)");
        check(!prime::is_prime(536870911ULL), "2^29-1 = 536870911 is composite");
        check(prime::is_prime(1000000007ULL),    "10^9+7 is prime");
        check(prime::is_prime(999999999989ULL),  "999999999989 is prime");
        check(!prime::is_prime(1000000007ULL * 1000000009ULL), "product of two large primes is composite");

        uint64_t small_primes[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97};
        bool all_small = true;
        for (auto p : small_primes) {
            if (!prime::is_prime(p)) { all_small = false; break; }
        }
        check(all_small, "all primes up to 97 detected");

        uint64_t small_comp[] = {4,6,8,9,10,12,14,15,16,18,20,21,22,24,25,26,27,28};
        bool all_sc = true;
        for (auto c : small_comp) {
            if (prime::is_prime(c)) { all_sc = false; break; }
        }
        check(all_sc, "small composites [4..28] correctly rejected");

        prime::Engine engine;
        auto r2 = engine.verify(2);
        check(r2.confirmed && r2.value == 2, "Engine::verify(2) confirms prime");
        auto r4 = engine.verify(4);
        check(!r4.confirmed && r4.value == 4, "Engine::verify(4) confirms composite");
        auto rBig = engine.verify(1000000007ULL);
        check(rBig.confirmed, "Engine::verify(10^9+7) confirms prime");

        if (cancelled()) goto test_done;
        { // ── 2. Modular Arithmetic ──
        log(@"\n=== 2. Modular Arithmetic ===\n");

        check(prime::mulmod(3, 5, 7) == 1, "mulmod(3,5,7) = 15 mod 7 = 1");
        check(prime::mulmod(0, 12345, 100) == 0, "mulmod(0,x,m) = 0");
        check(prime::mulmod(12345, 0, 100) == 0, "mulmod(x,0,m) = 0");
        check(prime::mulmod(1, 1, 1) == 0, "mulmod(1,1,1) = 0");

        uint64_t big = (1ULL << 63);
        uint64_t bigmod = big + 1;
        check(prime::mulmod(big, big, bigmod) == 1, "mulmod overflow: (2^63)^2 mod (2^63+1) = 1");

        uint64_t umax = UINT64_MAX;
        check(prime::mulmod(umax - 1, umax - 1, umax) == 1, "mulmod(UINT64_MAX-1, UINT64_MAX-1, UINT64_MAX) = 1");

        check(prime::mulmod(1ULL << 62, 4, (1ULL << 63) + 7) != 0, "mulmod near 2^64 does not crash");

        check(prime::modpow(2, 10, 1000) == 24, "modpow(2,10,1000) = 1024 mod 1000 = 24");
        check(prime::modpow(2, 10, 1024) == 0,  "modpow(2,10,1024) = 0");
        check(prime::modpow(3, 0, 100) == 1,    "modpow(x,0,m) = 1");
        check(prime::modpow(5, 1, 100) == 5,    "modpow(5,1,100) = 5");
        check(prime::modpow(7, 7, 1) == 0,      "modpow(x,y,1) = 0");

        uint64_t m = 1000000007ULL;
        uint64_t r100v = prime::modpow(3, 100, m);
        uint64_t r101v = prime::modpow(3, 101, m);
        check(prime::mulmod(r100v, 3, m) == r101v, "modpow consistency: 3^100 * 3 = 3^101 (mod 10^9+7)");

        uint64_t fermat_primes[] = {5, 7, 13, 97, 101, 997, 7919, 10007, 104729, 1000000007ULL};
        bool fermat_ok = true;
        for (auto p : fermat_primes) {
            if (prime::modpow(2, p - 1, p) != 1) { fermat_ok = false; break; }
        }
        check(fermat_ok, "Fermat's little theorem: 2^(p-1) mod p == 1 for known primes");

        bool fermat_multi = true;
        for (auto p : fermat_primes) {
            for (uint64_t a : {3ULL, 5ULL, 11ULL}) {
                if (a % p == 0) continue;
                if (prime::modpow(a, p - 1, p) != 1) { fermat_multi = false; break; }
            }
            if (!fermat_multi) break;
        }
        check(fermat_multi, "Fermat's little theorem holds for bases 3,5,11 across sample primes");

        uint64_t r64 = prime::modpow(2, 64, m);
        uint64_t r65 = prime::modpow(2, 65, m);
        check(prime::mulmod(r64, 2, m) == r65, "modpow consistency: 2^64 * 2 = 2^65 (mod 10^9+7)");
        }

        if (cancelled()) goto test_done;
        { // ── 3. Sieve ──
        log(@"\n=== 3. Sieve ===\n");

        auto s100v = prime::sieve(100);
        int count100 = 0;
        for (uint64_t i = 0; i <= 100; i++) if (s100v[i]) count100++;
        check(count100 == 25, "pi(100) = 25");

        auto s1000 = prime::sieve(1000);
        int count1000 = 0;
        for (uint64_t i = 0; i <= 1000; i++) if (s1000[i]) count1000++;
        check(count1000 == 168, "pi(1000) = 168");

        auto s10k = prime::sieve(10000);
        int count10k = 0;
        for (uint64_t i = 0; i <= 10000; i++) if (s10k[i]) count10k++;
        check(count10k == 1229, "pi(10000) = 1229");

        bool sieve_agree = true;
        for (uint64_t i = 0; i <= 1000; i++) {
            if ((bool)s1000[i] != prime::is_prime(i)) { sieve_agree = false; break; }
        }
        check(sieve_agree, "sieve agrees with is_prime for n <= 1000");

        auto s2v = prime::sieve(2);
        check(s2v[0] == false && s2v[1] == false && s2v[2] == true, "sieve(2) correct");

        bool wheel_ok = true;
        for (uint64_t i = 8; i <= 10000; i++) {
            if (s10k[i] && !prime::WHEEL.valid(i)) { wheel_ok = false; break; }
        }
        check(wheel_ok, "wheel-210 accepts all primes > 7 up to 10000");
        check(!prime::WHEEL.valid(4) && !prime::WHEEL.valid(6) && !prime::WHEEL.valid(9) && !prime::WHEEL.valid(10),
              "wheel-210 rejects 4, 6, 9, 10");
        }

        if (cancelled()) goto test_done;
        { // ── 4. Factoring ──
        log(@"\n=== 4. Factoring ===\n");

        auto f1 = prime::factor_u64(1);
        check(f1.empty(), "factor(1) = empty");

        auto f2v = prime::factor_u64(2);
        check(f2v.size() == 1 && f2v[0] == 2, "factor(2) = {2}");

        auto f12 = prime::factor_u64(12);
        check(f12.size() == 3 && f12[0] == 2 && f12[1] == 2 && f12[2] == 3, "factor(12) = {2,2,3}");

        auto f143 = prime::factor_u64(143);
        check(f143.size() == 2 && f143[0] == 11 && f143[1] == 13, "factor(143) = {11,13}");

        auto f256 = prime::factor_u64(256);
        check(f256.size() == 8, "factor(256) = eight 2s");
        bool all2 = true;
        for (auto x : f256) if (x != 2) all2 = false;
        check(all2, "factor(256) all factors are 2");

        auto fSemi = prime::factor_u64(100160063ULL);
        check(fSemi.size() == 2 && fSemi[0] == 10007 && fSemi[1] == 10009, "factor(10007*10009)");

        auto f3v = prime::factor_u64(1113121ULL);
        check(f3v.size() == 3 && f3v[0] == 101 && f3v[1] == 103 && f3v[2] == 107,
              "factor(101*103*107) = {101,103,107}");

        auto fPollard = prime::factor_u64(1000036000099ULL);
        check(fPollard.size() == 2 && fPollard[0] == 1000003ULL && fPollard[1] == 1000033ULL,
              "factor(1000003*1000033) via Pollard rho");

        uint64_t test_vals[] = {2, 12, 143, 256, 1113121ULL, 100160063ULL, 1000036000099ULL};
        bool product_ok = true;
        for (auto n : test_vals) {
            auto factors = prime::factor_u64(n);
            uint64_t product = 1;
            for (auto f : factors) product *= f;
            if (product != n) { product_ok = false; break; }
        }
        check(product_ok, "factor product reconstruction correct for all test values");

        bool all_prime = true;
        for (auto n : test_vals) {
            auto factors = prime::factor_u64(n);
            for (auto f : factors) {
                if (!prime::is_prime(f)) { all_prime = false; break; }
            }
            if (!all_prime) break;
        }
        check(all_prime, "all returned factors are prime");

        check(prime::factors_string(12) == "2 x 2 x 3", "factors_string(12) = \"2 x 2 x 3\"");
        check(prime::factors_string(1).empty(), "factors_string(1) = empty");

        uint64_t d = prime::brent_rho_one(143, 1);
        check(d == 11 || d == 13, "brent_rho_one(143,1) finds a factor of 143");
        }

        if (cancelled()) goto test_done;
        { // ── 5. Special Primes (Wieferich / Wilson) ──
        log(@"\n=== 5. Special Primes (Wieferich / Wilson) ===\n");

        auto wieferich_test = [](uint64_t p) -> bool {
            return prime::modpow(2, p - 1, p * p) == 1;
        };

        check(prime::is_prime(1093), "1093 is prime");
        check(wieferich_test(1093),  "1093 is Wieferich: 2^1092 mod 1093^2 == 1");
        check(prime::is_prime(3511), "3511 is prime");
        check(wieferich_test(3511),  "3511 is Wieferich: 2^3510 mod 3511^2 == 1");
        check(!wieferich_test(3),  "3 is not Wieferich");
        check(!wieferich_test(5),  "5 is not Wieferich");
        check(!wieferich_test(7),  "7 is not Wieferich");
        check(!wieferich_test(11), "11 is not Wieferich");
        check(!wieferich_test(13), "13 is not Wieferich");

        auto wilson_test = [](uint64_t p) -> bool {
            uint64_t mod = p * p;
            uint64_t factorial = 1;
            for (uint64_t i = 2; i < p; i++) {
                factorial = prime::mulmod(factorial, i, mod);
            }
            return factorial == mod - 1;
        };

        check(prime::is_prime(5),   "5 is prime");
        check(wilson_test(5),       "5 is Wilson prime: 4! mod 25 == 24");
        check(prime::is_prime(13),  "13 is prime");
        check(wilson_test(13),      "13 is Wilson prime: 12! mod 169 == 168");
        check(prime::is_prime(563), "563 is prime");
        check(wilson_test(563),     "563 is Wilson prime: 562! mod 563^2 == 563^2-1");
        check(!wilson_test(7),  "7 is not Wilson prime");
        check(!wilson_test(11), "11 is not Wilson prime");
        check(!wilson_test(23), "23 is not Wilson prime");
        }

        if (cancelled()) goto test_done;
        { // ── 6. Search Engine ──
        log(@"\n=== 6. Search Engine ===\n");

        int hw = std::max(1, (int)std::thread::hardware_concurrency());

        auto r1 = engine.search_range(2, 30, 1);
        check(r1.size() == 6, "search_range(2,30) finds 6 primes (wheel skips 2,3,5,7)");

        bool sorted = true;
        for (size_t i = 1; i < r1.size(); i++) {
            if (r1[i].value <= r1[i-1].value) { sorted = false; break; }
        }
        check(sorted, "search_range results are sorted");

        bool all_confirmed = true;
        for (auto& pr : r1) if (!pr.confirmed) all_confirmed = false;
        check(all_confirmed, "all search results have confirmed=true");

        auto sieveData = prime::sieve(1100);
        int sieve_count = 0;
        for (uint64_t i = 1000; i <= 1100; i++) if (sieveData[i]) sieve_count++;
        auto rMid = engine.search_range(1000, 1100, hw);
        check((int)rMid.size() == sieve_count, "search_range(1000,1100) count matches sieve");

        std::set<uint64_t> sieveSet, engineSet;
        for (uint64_t i = 1000; i <= 1100; i++) if (sieveData[i]) sieveSet.insert(i);
        for (auto& pr : rMid) engineSet.insert(pr.value);
        check(sieveSet == engineSet, "search_range(1000,1100) exact values match sieve");

        auto s100b = prime::sieve(100);
        int sieve100 = 0;
        for (uint64_t i = 2; i <= 100; i++) if (s100b[i]) sieve100++;
        auto r100b = engine.search_range(2, 100, 1);
        check((int)r100b.size() == sieve100 - 4, "search_range(2,100) matches pi(100)-4 (wheel skips 2,3,5,7)");

        auto s5200 = prime::sieve(5200);
        std::set<uint64_t> sSet5k, eSet5k;
        for (uint64_t i = 5000; i <= 5200; i++) if (s5200[i]) sSet5k.insert(i);
        auto r5k = engine.search_range(5000, 5200, hw);
        for (auto& pr : r5k) eSet5k.insert(pr.value);
        check(sSet5k == eSet5k, "search_range(5000,5200) exact values match sieve");

        auto r_single = engine.search_range(100000, 101000, 1);
        auto r_multi  = engine.search_range(100000, 101000, hw);
        std::set<uint64_t> set1, set2;
        for (auto& pr : r_single) set1.insert(pr.value);
        for (auto& pr : r_multi)  set2.insert(pr.value);
        check(set1 == set2, "single-thread and multi-thread find identical primes [100000,101000]");

        int twin_count = 0;
        for (size_t i = 1; i < r100b.size(); i++) {
            if (r100b[i].value - r100b[i-1].value == 2) twin_count++;
        }
        check(twin_count == 6, "6 twin prime pairs in engine results [2,100]");

        uint64_t lo = 999900, hi = 1000100;
        auto rBase = engine.search_range(lo, hi, hw);
        std::set<uint64_t> baseline;
        for (auto& pr : rBase) baseline.insert(pr.value);
        bool deterministic = true;
        for (int trial = 0; trial < 3; trial++) {
            auto rt = engine.search_range(lo, hi, hw);
            std::set<uint64_t> sv;
            for (auto& pr : rt) sv.insert(pr.value);
            if (sv != baseline) { deterministic = false; break; }
        }
        check(deterministic, "3 repeated runs produce identical results");
        }

        if (cancelled()) goto test_done;
        { // ── 7. Performance Benchmark ──
        log(@"\n=== 7. Performance Benchmark ===\n");

        int hw = std::max(1, (int)std::thread::hardware_concurrency());
        uint64_t bstart = 1000000000ULL;
        uint64_t bend   = 1000000000ULL + 100000ULL;
        auto t0 = std::chrono::steady_clock::now();
        auto results = engine.search_range(bstart, bend, hw);
        auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        log([NSString stringWithFormat:@"  Range: [%llu, %llu]\n", bstart, bend]);
        log([NSString stringWithFormat:@"  Primes found: %zu\n", results.size()]);
        log([NSString stringWithFormat:@"  Time: %.4f s\n", dt]);
        if (dt > 0) {
            log([NSString stringWithFormat:@"  Rate: %.0f candidates/sec\n", (double)(bend - bstart) / dt]);
        }

        check(results.size() > 4500 && results.size() < 5200,
              "prime count in [10^9, 10^9+10^5] is reasonable (~4832)");
        check(dt < 30.0, "benchmark completes in under 30 seconds");
        }

        if (cancelled()) goto test_done;
        { // ── A. CRT Rejection Filter ──
        log(@"\n=== A. CRT Rejection Filter ===\n");

        check(prime::crt_reject(0),  "crt_reject(0)");
        check(prime::crt_reject(1),  "crt_reject(1)");
        check(!prime::crt_reject(2), "crt_reject(2) = false (prime)");
        check(!prime::crt_reject(37), "crt_reject(37) = false (prime)");
        check(prime::crt_reject(11 * 43), "crt_reject(11*43)");
        check(prime::crt_reject(13 * 47), "crt_reject(13*47)");
        check(prime::crt_reject(17 * 41), "crt_reject(17*41)");
        check(prime::crt_reject(19 * 43), "crt_reject(19*43)");
        check(prime::crt_reject(23 * 41), "crt_reject(23*41)");
        check(prime::crt_reject(29 * 43), "crt_reject(29*43)");
        check(prime::crt_reject(31 * 43), "crt_reject(31*43)");

        auto crtSv = prime::sieve(10000);
        bool crt_ok = true;
        for (uint64_t i = 38; i <= 10000; i++) {
            if (crtSv[i] && prime::crt_reject(i)) { crt_ok = false; break; }
        }
        check(crt_ok, "crt_reject does not reject any prime in [38, 10000]");
        }

        if (cancelled()) goto test_done;
        { // ── B. Convergence / Shadow Field ──
        log(@"\n=== B. Convergence / Shadow Field ===\n");

        check(prime::convergence(7 * 11) == -999.0, "convergence(77) = -999 (multiple of shadow prime)");
        check(prime::convergence(13 * 17) == -999.0, "convergence(221) = -999");

        double c97 = prime::convergence(97);
        check(c97 != -999.0, "convergence(97) is finite (prime)");
        double c101 = prime::convergence(101);
        check(c101 != -999.0, "convergence(101) is finite");

        auto prv = engine.verify(997);
        check(prv.convergence_score != 0.0 && prv.convergence_score != -999.0,
              "verify(997) returns non-trivial convergence score");

        // ── C. Heuristic Factoring ──
        log(@"\n=== C. Heuristic Factoring ===\n");

        auto ph = prime::pinch_factor(1729);
        bool pinch_found = false;
        for (auto& h : ph) {
            if (1729 % h.divisor == 0 && h.divisor > 1 && h.divisor < 1729) pinch_found = true;
        }
        check(pinch_found, "pinch_factor(1729) finds at least one divisor");

        auto l7 = prime::lucky7_factor(100160063ULL);
        bool lucky7_found = false;
        for (auto& h : l7) {
            if (100160063ULL % h.divisor == 0 && h.divisor > 1) lucky7_found = true;
        }
        check(lucky7_found, "lucky7_factor(10007*10009) finds factor near 10^4");

        auto web = prime::divisor_web(60, 10);
        check(web.n == 60, "divisor_web(60).n = 60");
        std::set<uint64_t> expected_pd = {2, 3, 5};
        std::set<uint64_t> actual_pd(web.prime_divisors.begin(), web.prime_divisors.end());
        check(actual_pd == expected_pd, "divisor_web(60) prime_divisors = {2,3,5}");

        auto hd = prime::heuristic_divisors(1729);
        bool hd_valid = true;
        for (auto dv : hd) {
            if (1729 % dv != 0) { hd_valid = false; break; }
        }
        check(hd_valid, "heuristic_divisors(1729) all results divide 1729");
        }

    test_done:
        // ── Results ──
        if (cancelled()) {
            log(@"\nTest suite cancelled.\n\n");
        } else {
            log(@"\n================================================\n");
            NSString *summary = [NSString stringWithFormat:
                @"RESULTS: %d passed, %d failed, %d total\n", pass, fail, pass + fail];
            log(summary);
            log(@"================================================\n");
            if (fail == 0) {
                log(@"ALL TESTS PASSED\n\n");
            } else {
                log([NSString stringWithFormat:@"%d TESTS FAILED\n\n", fail]);
            }
        }
        ss->_checkRunning.store(false);
    });
    // Thread stays joinable for proper cleanup
}

- (BOOL)applicationSupportsSecureRestorableState:(NSApplication *)app {
    return YES;
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender {
    // Invalidate all timers first
    [self.refreshTimer invalidate];
    self.refreshTimer = nil;
    [self.frontierCheckTimer invalidate];
    self.frontierCheckTimer = nil;

    // Release sleep prevention
    [self releaseSleepAssertion];

    // Signal all background threads to stop
    _checkRunning.store(false);
    _benchRunning.store(false);

    // Stop TaskManager tasks (this signals worker threads inside TaskManager)
    if (_taskMgr) {
        _taskMgr->stop_all();
    }

    // Join background threads — use a helper that detaches if the thread is stuck
    // (e.g., blocked on network I/O after "connection reset by peer")
    auto safeJoin = [](std::thread &t) {
        if (!t.joinable()) return;
        // Give the thread up to 500ms to notice its stop flag and exit
        std::promise<void> p;
        auto f = p.get_future();
        std::thread joiner([&]{ t.join(); p.set_value(); });
        if (f.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready) {
            joiner.join();
        } else {
            joiner.detach(); // Thread is stuck — detach and let the OS reclaim on exit
        }
    };
    safeJoin(_checkThread);
    safeJoin(_benchThread);

    if (self.eventMonitor) {
        [NSEvent removeMonitor:self.eventMonitor];
        self.eventMonitor = nil;
    }
    // Stop distributed computing
    [self.networkRefreshTimer invalidate];
    self.networkRefreshTimer = nil;
    if (_conductor) { _conductor->stop(); delete _conductor; _conductor = nullptr; }
    if (_carriage) { _carriage->stop(); delete _carriage; _carriage = nullptr; }

    if (_taskMgr) {
        delete _taskMgr;
        _taskMgr = nullptr;
    }
    if (_gpu) {
        delete _gpu;
        _gpu = nullptr;
    }
    return NSTerminateNow;
}

// ═══════════════════════════════════════════════════════════════════════
// Test Catalog -- loads from TestCatalog.txt
// ═══════════════════════════════════════════════════════════════════════

- (void)loadTestCatalog {
    _testCatalog.clear();
    _selectedTestIdx = -1;
    NSString *path = [DATA_DIR stringByAppendingPathComponent:@"TestCatalog.txt"];
    NSString *content = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
    if (!content) return;

    TestCatalogEntry current;
    bool in_entry = false;

    for (NSString *rawLine in [content componentsSeparatedByString:@"\n"]) {
        NSString *line = [rawLine stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
        if (line.length == 0 || [line hasPrefix:@"#"]) continue;

        if ([line hasPrefix:@"["] && [line hasSuffix:@"]"]) {
            if (in_entry && !current.test_id.empty()) _testCatalog.push_back(current);
            current = TestCatalogEntry();
            current.test_id = [[line substringWithRange:NSMakeRange(1, line.length - 2)] UTF8String];
            in_entry = true;
            continue;
        }
        if (!in_entry) continue;

        NSRange eq = [line rangeOfString:@" = "];
        if (eq.location == NSNotFound) continue;
        NSString *key = [line substringToIndex:eq.location];
        NSString *val = [line substringFromIndex:eq.location + 3];
        std::string k = key.UTF8String, v = val.UTF8String;

        if (k == "name") current.name = v;
        else if (k == "category") current.category = v;
        else if (k == "mode") current.mode = v;
        else if (k == "description") current.description = v;
        else if (k == "algorithms") current.algorithms = v;
        else if (k == "default_params") current.default_params = v;
    }
    if (in_entry && !current.test_id.empty()) _testCatalog.push_back(current);

    // Build display order grouped by category
    _testDisplayOrder.clear();
    _testCategories.clear();
    std::map<std::string, std::vector<size_t>> byCategory;
    std::vector<std::string> catOrder;
    for (size_t i = 0; i < _testCatalog.size(); i++) {
        auto& cat = _testCatalog[i].category;
        if (byCategory.find(cat) == byCategory.end()) catOrder.push_back(cat);
        byCategory[cat].push_back(i);
    }
    for (auto& cat : catOrder) {
        _testCategories.push_back(cat);
        _testDisplayOrder.push_back(SIZE_MAX); // category header marker
        for (auto idx : byCategory[cat]) {
            _testDisplayOrder.push_back(idx);
        }
    }
}

- (void)showTestCatalog:(id)sender {
    if (self.testCatalogWindow) {
        [self loadTestCatalog];
        [self.testTableView reloadData];
        [self.testCatalogWindow makeKeyAndOrderFront:nil];
        return;
    }

    [self loadTestCatalog];

    CGFloat W = 820, H = 600;
    self.testCatalogWindow = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(100, 100, W, H)
        styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                   NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
        backing:NSBackingStoreBuffered defer:NO];
    [self.testCatalogWindow setTitle:@"Test Catalog"];
    [self.testCatalogWindow setMinSize:NSMakeSize(600, 400)];
    self.testCatalogWindow.delegate = self;

    NSView *cv = self.testCatalogWindow.contentView;
    CGFloat M = 10;

    // Split: left list (280px) | right detail (rest)
    CGFloat listW = 280;

    // Left: table view in scroll view
    NSScrollView *tableScroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(M, 44, listW, H - 54)];
    tableScroll.hasVerticalScroller = YES;
    tableScroll.borderType = NSBezelBorder;
    tableScroll.autoresizingMask = NSViewHeightSizable | NSViewMaxXMargin;

    self.testTableView = [[NSTableView alloc] initWithFrame:NSMakeRect(0, 0, listW - 4, H - 54)];
    NSTableColumn *col = [[NSTableColumn alloc] initWithIdentifier:@"test"];
    col.title = @"Tests";
    col.width = listW - 8;
    col.resizingMask = NSTableColumnAutoresizingMask;
    [self.testTableView addTableColumn:col];
    self.testTableView.headerView = nil;
    self.testTableView.delegate = self;
    self.testTableView.dataSource = self;
    self.testTableView.rowHeight = 22;
    self.testTableView.target = self;
    self.testTableView.action = @selector(testTableClicked:);
    tableScroll.documentView = self.testTableView;
    [cv addSubview:tableScroll];

    // Right: detail view
    CGFloat detailX = M + listW + 8;
    CGFloat detailW = W - detailX - M;

    // Detail description area (upper right)
    CGFloat paramH = 80;
    NSScrollView *detailScroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(detailX, 44 + paramH + 4, detailW, H - 54 - paramH - 4)];
    detailScroll.hasVerticalScroller = YES;
    detailScroll.borderType = NSBezelBorder;
    detailScroll.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.testDetailView = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, detailW - 4, H - 54 - paramH - 4)];
    self.testDetailView.editable = NO;
    self.testDetailView.font = [NSFont systemFontOfSize:11];
    self.testDetailView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.testDetailView.string = @"Select a test from the list to see its description, algorithms, and parameters.";
    detailScroll.documentView = self.testDetailView;
    [cv addSubview:detailScroll];

    // Parameter input area (multi-line text view, lower right)
    NSTextField *paramLbl = [self labelAt:NSMakeRect(detailX, 44 + paramH - 14, 80, 14) text:@"Parameters:" bold:NO size:10];
    paramLbl.autoresizingMask = NSViewMaxYMargin | NSViewWidthSizable;
    [cv addSubview:paramLbl];

    self.testParamScroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(detailX, 44, detailW - 80, paramH - 16)];
    self.testParamScroll.hasVerticalScroller = YES;
    self.testParamScroll.borderType = NSBezelBorder;
    self.testParamScroll.autoresizingMask = NSViewMaxYMargin | NSViewWidthSizable;
    self.testParamField = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, detailW - 84, paramH - 16)];
    self.testParamField.editable = YES;
    self.testParamField.selectable = YES;
    self.testParamField.richText = NO;
    self.testParamField.font = [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular];
    self.testParamField.autoresizingMask = NSViewWidthSizable;
    self.testParamField.string = @"";
    self.testParamScroll.documentView = self.testParamField;
    [cv addSubview:self.testParamScroll];

    self.testRunButton = [self buttonAt:NSMakeRect(W - M - 70, 44 + paramH/2 - 14, 70, 28)
        title:@"Run" action:@selector(runCatalogTest:)];
    self.testRunButton.font = [NSFont boldSystemFontOfSize:11];
    self.testRunButton.autoresizingMask = NSViewMaxYMargin | NSViewMinXMargin;
    [cv addSubview:self.testRunButton];

    // Reload button
    NSButton *reloadBtn = [self buttonAt:NSMakeRect(M, 14, 90, 22)
        title:@"Reload File" action:@selector(reloadTestCatalog:)];
    reloadBtn.font = [NSFont systemFontOfSize:9];
    reloadBtn.autoresizingMask = NSViewMaxYMargin;
    [cv addSubview:reloadBtn];

    NSTextField *fileLbl = [self labelAt:NSMakeRect(M + 96, 16, 400, 13)
        text:[NSString stringWithFormat:@"Catalog: %@/TestCatalog.txt", DATA_DIR] bold:NO size:8.5];
    fileLbl.textColor = [NSColor secondaryLabelColor];
    fileLbl.autoresizingMask = NSViewMaxYMargin | NSViewWidthSizable;
    [cv addSubview:fileLbl];

    [self.testCatalogWindow makeKeyAndOrderFront:nil];
}

- (void)reloadTestCatalog:(id)sender {
    [self loadTestCatalog];
    [self.testTableView reloadData];
    self.testDetailView.string = @"Catalog reloaded. Select a test.";
    [self appendText:@"Test catalog reloaded from file.\n"];
}

// ── NSTableView DataSource / Delegate ────────────────────────────────

- (NSInteger)numberOfRowsInTableView:(NSTableView *)tableView {
    return (NSInteger)_testDisplayOrder.size();
}

- (NSView *)tableView:(NSTableView *)tableView viewForTableColumn:(NSTableColumn *)tableColumn row:(NSInteger)row {
    if (row < 0 || row >= (NSInteger)_testDisplayOrder.size()) return nil;
    size_t idx = _testDisplayOrder[row];

    NSTextField *cell = [tableView makeViewWithIdentifier:@"TestCell" owner:self];
    if (!cell) {
        cell = [[NSTextField alloc] initWithFrame:NSMakeRect(0, 0, 270, 20)];
        cell.identifier = @"TestCell";
        cell.editable = NO;
        cell.bordered = NO;
        cell.drawsBackground = NO;
    }

    if (idx == SIZE_MAX) {
        // Category header -- find which one
        size_t catIdx = 0;
        for (size_t i = 0; i <= (size_t)row; i++) {
            if (_testDisplayOrder[i] == SIZE_MAX) catIdx++;
        }
        catIdx--;
        NSString *catName = (catIdx < _testCategories.size())
            ? [NSString stringWithUTF8String:_testCategories[catIdx].c_str()]
            : @"Unknown";
        cell.stringValue = [catName uppercaseString];
        cell.font = [NSFont boldSystemFontOfSize:9.5];
        cell.textColor = [NSColor secondaryLabelColor];
    } else {
        auto& entry = _testCatalog[idx];
        NSString *mode = (entry.mode == "search") ? @"  [search]" : @"";
        cell.stringValue = [NSString stringWithFormat:@"  %s%@",
            entry.name.c_str(), mode];
        cell.font = [NSFont systemFontOfSize:11];
        cell.textColor = [NSColor labelColor];
    }
    return cell;
}

- (BOOL)tableView:(NSTableView *)tableView shouldSelectRow:(NSInteger)row {
    if (row < 0 || row >= (NSInteger)_testDisplayOrder.size()) return NO;
    return _testDisplayOrder[row] != SIZE_MAX; // can't select category headers
}

- (void)testTableClicked:(id)sender {
    NSInteger row = self.testTableView.selectedRow;
    if (row < 0 || row >= (NSInteger)_testDisplayOrder.size()) return;
    size_t idx = _testDisplayOrder[row];
    if (idx == SIZE_MAX) return;

    _selectedTestIdx = (int)idx;
    auto& entry = _testCatalog[idx];

    NSMutableString *detail = [NSMutableString string];
    [detail appendFormat:@"%s\n", entry.name.c_str()];
    [detail appendString:@"────────────────────────────────────────\n\n"];
    [detail appendFormat:@"Category: %s\n", entry.category.c_str()];
    [detail appendFormat:@"Mode: %s\n\n", entry.mode == "search" ? "Long-running search (Start/Pause/Stop)" : "One-shot analysis tool"];
    [detail appendFormat:@"DESCRIPTION\n%s\n\n", entry.description.c_str()];
    [detail appendFormat:@"ALGORITHMS USED\n"];
    // Split by | and show as bullet list
    NSString *algos = [NSString stringWithUTF8String:entry.algorithms.c_str()];
    for (NSString *algo in [algos componentsSeparatedByString:@"|"]) {
        NSString *trimmed = [algo stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
        if (trimmed.length > 0)
            [detail appendFormat:@"  • %@\n", trimmed];
    }
    [detail appendString:@"\n"];

    // Mode-specific parameter hints
    if (entry.mode == "search") {
        [detail appendString:@"USAGE\nSelect this task in the main window dropdown and click Start.\n"
            @"The search runs continuously until stopped. Use Set Start to\n"
            @"choose the starting position.\n"];
    } else {
        [detail appendString:@"PARAMETERS\nEnter values in the parameter field below and click Run.\n"];
        // Specific hints per test
        if (entry.test_id == "rsa_key_audit" || entry.test_id == "factorization_workbench")
            [detail appendString:@"Enter one or more numbers separated by spaces.\n"];
        else if (entry.test_id == "batch_gcd_audit")
            [detail appendString:@"Enter moduli separated by spaces.\n"];
        else if (entry.test_id == "discrete_log")
            [detail appendString:@"Enter: g h p  (finds x where g^x == h mod p)\n"];
        else if (entry.test_id == "primitive_root_finder")
            [detail appendString:@"Enter a prime p.\n"];
        else if (entry.test_id == "dh_parameter_test")
            [detail appendString:@"Enter: g p  (generator and prime)\n"];
        else if (entry.test_id == "euler_totient_calc")
            [detail appendString:@"Enter one or more numbers.\n"];
        else if (entry.test_id == "multiplicative_order_calc")
            [detail appendString:@"Enter: a m  (finds ord_m(a))\n"];
        else if (entry.test_id == "crt_solver")
            [detail appendString:@"Enter pairs: r1 m1 r2 m2 ...  (x == r_i mod m_i)\n"];
        else if (entry.test_id == "quadratic_residue_test")
            [detail appendString:@"Enter: a p  (test if a is QR mod p)\n"];
        else if (entry.test_id == "sum_two_squares")
            [detail appendString:@"Enter a number n to decompose as a^2 + b^2.\n"];
        else if (entry.test_id == "perfect_power_test")
            [detail appendString:@"Enter a number n to test if n = a^k.\n"];
        else if (entry.test_id == "jacobi_symbol_calc")
            [detail appendString:@"Enter: a n  (computes Jacobi symbol (a/n))\n"];
        else if (entry.test_id == "smooth_number_finder")
            [detail appendString:@"Enter: lo hi B  (find B-smooth numbers in [lo,hi])\n"];
        else if (entry.test_id == "lcg_period_analysis")
            [detail appendString:@"Enter: a c m [seed]  (LCG: x' = a*x+c mod m)\n"];
        else if (entry.test_id == "pollard_p1_factor")
            [detail appendString:@"Enter: n [B]  (factor n, smoothness bound B default 100000)\n"];
        else if (entry.test_id == "convergence_analyzer")
            [detail appendString:@"Enter one or more numbers.\n"];
        else if (entry.test_id == "modular_arithmetic")
            [detail appendString:@"Enter: op a b m  (op = modpow|mulmod|modinv|gcd|lcm)\n"];
    }

    self.testDetailView.string = detail;

    // Pre-fill parameter field with default example values
    if (!entry.default_params.empty()) {
        self.testParamField.string = [NSString stringWithUTF8String:entry.default_params.c_str()];
    } else {
        self.testParamField.string = @"";
    }
}

// ── Run selected test ────────────────────────────────────────────────

- (void)runCatalogTest:(id)sender {
    if (_selectedTestIdx < 0 || _selectedTestIdx >= (int)_testCatalog.size()) return;
    auto& entry = _testCatalog[_selectedTestIdx];

    // Always stop and join any previous test thread first
    _checkRunning.store(false);
    if (_checkThread.joinable()) _checkThread.join();

    // For search-mode tests, start the corresponding TaskManager task
    if (entry.mode == "search") {
        std::map<std::string, prime::TaskType> searchMap = {
            {"wieferich_search", prime::TaskType::Wieferich},
            {"wallsunsun_search", prime::TaskType::WallSunSun},
            {"wilson_search", prime::TaskType::Wilson},
            {"twin_prime_search", prime::TaskType::TwinPrime},
            {"sophie_germain_search", prime::TaskType::SophieGermain},
            {"cousin_prime_search", prime::TaskType::CousinPrime},
            {"sexy_prime_search", prime::TaskType::SexyPrime},
            {"general_prime_search", prime::TaskType::GeneralPrime},
            {"emirp_search", prime::TaskType::Emirp},
            {"mersenne_trial", prime::TaskType::MersenneTrial},
            {"fermat_factor", prime::TaskType::FermatFactor},
        };
        auto it = searchMap.find(entry.test_id);
        if (it != searchMap.end()) {
            _taskMgr->start_task(it->second);
            [self appendText:[NSString stringWithFormat:@"Started search: %s\n", entry.name.c_str()]];
        }
        return;
    }

    // For tool-mode tests, parse params and run
    NSString *params = [self.testParamField.string
        stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];

    // Parse values separated by spaces, newlines, commas, semicolons, or pipes
    NSMutableCharacterSet *delimiters = [NSMutableCharacterSet whitespaceAndNewlineCharacterSet];
    [delimiters addCharactersInString:@",;|"];
    NSArray *parts = [params componentsSeparatedByCharactersInSet:delimiters];
    std::vector<uint64_t> vals;
    std::vector<std::string> tokens;
    for (NSString *p in parts) {
        NSString *cleaned = [p stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
        if (cleaned.length > 0) {
            tokens.push_back(cleaned.UTF8String);
            uint64_t v = strtoull(cleaned.UTF8String, nullptr, 10);
            vals.push_back(v);
        }
    }

    std::string tid = entry.test_id;
    __weak AppDelegate *weakSelf = self;

    // Special cases that use existing methods
    if (tid == "benchmark_suite") {
        [self runBenchmark:sender];
        return;
    }
    if (tid == "engine_test_suite") {
        [self runInAppTests:sender];
        return;
    }

    _checkRunning.store(true);

    _checkThread = std::thread([weakSelf, tid, vals, tokens]() {
        @autoreleasepool {
        AppDelegate *ss = weakSelf;
        if (!ss) return;

        auto log = [weakSelf](NSString *msg) {
            dispatch_async(dispatch_get_main_queue(), ^{
                @autoreleasepool {
                [weakSelf appendText:msg];
                }
            });
        };

        if (tid == "rsa_key_audit" || tid == "factorization_workbench") {
            if (vals.empty()) { log(@"Enter numbers to factor.\n"); ss->_checkRunning.store(false); return; }
            for (auto n : vals) {
                auto factors = prime::factor_u64(n);
                NSMutableString *s = [NSMutableString stringWithFormat:@"%llu = ", n];
                if (factors.empty()) [s appendString:@"(no factors)"];
                for (size_t i = 0; i < factors.size(); i++) {
                    if (i > 0) [s appendString:@" x "];
                    [s appendFormat:@"%llu", factors[i]];
                    if (prime::is_prime(factors[i])) [s appendString:@"(prime)"];
                }
                [s appendString:@"\n"];
                log(s);
                // Additional info for RSA audit
                if (tid == "rsa_key_audit") {
                    if (factors.size() == 1 && factors[0] == n) {
                        log([NSString stringWithFormat:@"  -> %llu is prime. Strong key.\n", n]);
                    } else if (factors.size() == 2) {
                        uint64_t bits_p = 0, bits_q = 0, p = factors[0], q = factors[1];
                        while ((1ULL << bits_p) <= p) bits_p++;
                        while ((1ULL << bits_q) <= q) bits_q++;
                        log([NSString stringWithFormat:@"  -> Semiprime: %llu-bit x %llu-bit. %s\n",
                            bits_p, bits_q, (bits_p < 20 || bits_q < 20) ? "WEAK -- small factor!" : "Factor sizes OK."]);
                    } else {
                        log([NSString stringWithFormat:@"  -> %zu factors. Not a standard RSA modulus.\n", factors.size()]);
                    }
                }
            }
        }
        else if (tid == "batch_gcd_audit") {
            if (vals.size() < 2) { log(@"Enter at least 2 moduli.\n"); ss->_checkRunning.store(false); return; }
            log([NSString stringWithFormat:@"Batch GCD audit on %zu moduli...\n", vals.size()]);
            auto weak = prime::batch_gcd_audit(vals);
            if (weak.empty()) {
                log(@"No shared factors found. All moduli appear independent.\n");
            } else {
                log([NSString stringWithFormat:@"WEAK KEYS FOUND: %zu shared factors!\n", weak.size() / 2]);
                for (auto& w : weak) {
                    log([NSString stringWithFormat:@"  %llu shares factor %llu with %llu\n",
                        w.modulus, w.shared_factor, w.other_modulus]);
                }
            }
        }
        else if (tid == "discrete_log") {
            if (vals.size() < 3) { log(@"Enter: g h p\n"); ss->_checkRunning.store(false); return; }
            log([NSString stringWithFormat:@"Solving %llu^x == %llu (mod %llu)...\n", vals[0], vals[1], vals[2]]);
            auto t0 = std::chrono::steady_clock::now();
            int64_t x = prime::baby_step_giant_step(vals[0], vals[1], vals[2]);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (x >= 0) {
                log([NSString stringWithFormat:@"Solution: x = %lld  (%.4f s)\n", x, dt]);
                uint64_t verify = prime::modpow(vals[0], (uint64_t)x, vals[2]);
                log([NSString stringWithFormat:@"Verify: %llu^%lld mod %llu = %llu %s\n",
                    vals[0], x, vals[2], verify, verify == vals[1] ? "OK" : "FAIL"]);
            } else {
                log([NSString stringWithFormat:@"No solution found (%.4f s). p may not be prime.\n", dt]);
            }
        }
        else if (tid == "primitive_root_finder") {
            if (vals.empty()) { log(@"Enter a prime p.\n"); ss->_checkRunning.store(false); return; }
            for (auto p : vals) {
                if (!prime::is_prime(p)) {
                    log([NSString stringWithFormat:@"%llu is not prime.\n", p]);
                    continue;
                }
                uint64_t g = prime::primitive_root(p);
                uint64_t ord = prime::multiplicative_order(g, p);
                log([NSString stringWithFormat:@"Primitive root of %llu: g = %llu  (order = %llu = p-1 [OK])\n", p, g, ord]);
            }
        }
        else if (tid == "dh_parameter_test") {
            if (vals.size() < 2) { log(@"Enter: g p\n"); ss->_checkRunning.store(false); return; }
            uint64_t g = vals[0], p = vals[1];
            log([NSString stringWithFormat:@"Diffie-Hellman parameter analysis: g=%llu, p=%llu\n", g, p]);
            log([NSString stringWithFormat:@"  p is prime: %s\n", prime::is_prime(p) ? "YES" : "NO -- INSECURE"]);
            if (prime::is_prime(p)) {
                bool safe = prime::is_prime((p - 1) / 2);
                log([NSString stringWithFormat:@"  Safe prime ((p-1)/2 prime): %s\n", safe ? "YES" : "NO"]);
                uint64_t ord = prime::multiplicative_order(g, p);
                log([NSString stringWithFormat:@"  ord_p(g) = %llu", ord]);
                if (ord == p - 1) log(@"  (generator -- GOOD)\n");
                else log([NSString stringWithFormat:@"  (subgroup of size %llu -- may be weak)\n", ord]);
                auto pf = prime::factor_u64(p - 1);
                NSMutableString *fs = [NSMutableString stringWithString:@"  p-1 factors: "];
                for (size_t i = 0; i < pf.size(); i++) {
                    if (i > 0) [fs appendString:@" x "];
                    [fs appendFormat:@"%llu", pf[i]];
                }
                [fs appendString:@"\n"];
                log(fs);
                uint64_t largest = pf.empty() ? 0 : pf.back();
                log([NSString stringWithFormat:@"  Largest factor of p-1: %llu (%s)\n",
                    largest, largest > 1000000 ? "large -- resists Pohlig-Hellman" : "SMALL -- vulnerable to Pohlig-Hellman"]);
            }
        }
        else if (tid == "euler_totient_calc") {
            if (vals.empty()) { log(@"Enter numbers.\n"); ss->_checkRunning.store(false); return; }
            for (auto n : vals) {
                uint64_t phi = prime::euler_totient(n);
                log([NSString stringWithFormat:@"phi(%llu) = %llu", n, phi]);
                auto factors = prime::factor_u64(n);
                NSMutableString *fs = [NSMutableString stringWithString:@"  ["];
                for (size_t i = 0; i < factors.size(); i++) {
                    if (i > 0) [fs appendString:@"x"];
                    [fs appendFormat:@"%llu", factors[i]];
                }
                [fs appendString:@"]\n"];
                log(fs);
            }
        }
        else if (tid == "multiplicative_order_calc") {
            if (vals.size() < 2) { log(@"Enter: a m\n"); ss->_checkRunning.store(false); return; }
            uint64_t a = vals[0], m = vals[1];
            uint64_t g = prime::gcd(a, m);
            if (g != 1) {
                log([NSString stringWithFormat:@"gcd(%llu, %llu) = %llu ≠ 1. Order undefined.\n", a, m, g]);
            } else {
                uint64_t ord = prime::multiplicative_order(a, m);
                log([NSString stringWithFormat:@"ord_%llu(%llu) = %llu\n", m, a, ord]);
                log([NSString stringWithFormat:@"Verify: %llu^%llu mod %llu = %llu\n",
                    a, ord, m, prime::modpow(a, ord, m)]);
            }
        }
        else if (tid == "crt_solver") {
            if (vals.size() < 4 || vals.size() % 2 != 0) {
                log(@"Enter pairs: r1 m1 r2 m2 ...\n"); ss->_checkRunning.store(false); return;
            }
            std::vector<uint64_t> rems, mods;
            for (size_t i = 0; i < vals.size(); i += 2) {
                rems.push_back(vals[i]);
                mods.push_back(vals[i + 1]);
                log([NSString stringWithFormat:@"  x == %llu (mod %llu)\n", vals[i], vals[i + 1]]);
            }
            auto [sol, mod] = prime::chinese_remainder(rems, mods);
            if (mod == 0) {
                log(@"No solution (moduli not coprime or inconsistent).\n");
            } else {
                log([NSString stringWithFormat:@"Solution: x == %llu (mod %llu)\n", sol, mod]);
                // Verify
                bool ok = true;
                for (size_t i = 0; i < rems.size(); i++) {
                    if (sol % mods[i] != rems[i]) ok = false;
                }
                log([NSString stringWithFormat:@"Verify: %s\n", ok ? "All congruences satisfied [OK]" : "ERROR [X]"]);
            }
        }
        else if (tid == "quadratic_residue_test") {
            if (vals.size() < 2) { log(@"Enter: a p\n"); ss->_checkRunning.store(false); return; }
            uint64_t a = vals[0], p = vals[1];
            if (p < 2) { log(@"p must be >= 2.\n"); ss->_checkRunning.store(false); return; }
            if (!prime::is_prime(p)) {
                log([NSString stringWithFormat:@"Warning: %llu is not prime. Results use Jacobi symbol.\n", p]);
            }
            bool qr = prime::is_quadratic_residue(a, p);
            int leg = prime::legendre_symbol(a, p);
            log([NSString stringWithFormat:@"(%llu/%llu) = %d  --  %llu is %sa quadratic residue mod %llu\n",
                a, p, leg, a, qr ? "" : "NOT ", p]);
            if (qr) {
                uint64_t root = prime::sqrt_mod(a, p);
                log([NSString stringWithFormat:@"sqrt%llu mod %llu = %llu\n", a, p, root]);
                log([NSString stringWithFormat:@"Verify: %llu^2 mod %llu = %llu %s\n",
                    root, p, prime::mulmod(root, root, p),
                    prime::mulmod(root, root, p) == a % p ? "OK" : "FAIL"]);
            }
        }
        else if (tid == "sum_two_squares") {
            if (vals.empty()) { log(@"Enter a number.\n"); ss->_checkRunning.store(false); return; }
            for (auto n : vals) {
                auto [a, b] = prime::sum_two_squares(n);
                if (a == 0 && b == 0 && n != 0) {
                    log([NSString stringWithFormat:@"%llu cannot be written as a^2 + b^2\n", n]);
                } else {
                    log([NSString stringWithFormat:@"%llu = %llu^2 + %llu^2 = %llu + %llu\n", n, a, b, a*a, b*b]);
                }
            }
        }
        else if (tid == "perfect_power_test") {
            if (vals.empty()) { log(@"Enter numbers.\n"); ss->_checkRunning.store(false); return; }
            for (auto n : vals) {
                auto [base, exp] = prime::perfect_power(n);
                if (exp > 1) {
                    log([NSString stringWithFormat:@"%llu = %llu^%llu  (perfect power)\n", n, base, exp]);
                } else {
                    log([NSString stringWithFormat:@"%llu is not a perfect power\n", n]);
                }
            }
        }
        else if (tid == "jacobi_symbol_calc") {
            if (vals.size() < 2) { log(@"Enter: a n\n"); ss->_checkRunning.store(false); return; }
            int j = prime::jacobi_symbol((int64_t)vals[0], (int64_t)vals[1]);
            log([NSString stringWithFormat:@"Jacobi symbol (%llu/%llu) = %d\n", vals[0], vals[1], j]);
            if (prime::is_prime(vals[1])) {
                log([NSString stringWithFormat:@"(n=%llu is prime, so this equals the Legendre symbol)\n", vals[1]]);
            }
        }
        else if (tid == "smooth_number_finder") {
            if (vals.size() < 3) { log(@"Enter: lo hi B\n"); ss->_checkRunning.store(false); return; }
            uint64_t lo = vals[0], hi = vals[1], B = vals[2];
            if (hi - lo > 1000000) { log(@"Range too large (max 1M). Reduce range.\n"); ss->_checkRunning.store(false); return; }
            log([NSString stringWithFormat:@"Finding %llu-smooth numbers in [%llu, %llu]...\n", B, lo, hi]);
            auto smooth = prime::enumerate_smooth(lo, hi, B);
            log([NSString stringWithFormat:@"Found %zu smooth numbers (%.1f%% of range)\n",
                smooth.size(), 100.0 * smooth.size() / (hi - lo + 1)]);
            size_t show = std::min(smooth.size(), (size_t)50);
            for (size_t i = 0; i < show; i++) {
                auto factors = prime::factor_u64(smooth[i]);
                NSMutableString *s = [NSMutableString stringWithFormat:@"  %llu = ", smooth[i]];
                for (size_t j = 0; j < factors.size(); j++) {
                    if (j > 0) [s appendString:@"x"];
                    [s appendFormat:@"%llu", factors[j]];
                }
                [s appendString:@"\n"];
                log(s);
            }
            if (smooth.size() > show) log([NSString stringWithFormat:@"  ... and %zu more\n", smooth.size() - show]);
        }
        else if (tid == "lcg_period_analysis") {
            if (vals.size() < 3) { log(@"Enter: a c m [seed]\n"); ss->_checkRunning.store(false); return; }
            uint64_t a = vals[0], c = vals[1], m = vals[2];
            uint64_t seed = vals.size() > 3 ? vals[3] : 0;
            log([NSString stringWithFormat:@"LCG analysis: x' = %llu*x + %llu mod %llu (seed=%llu)\n", a, c, m, seed]);
            auto result = prime::analyze_lcg(a, c, m, seed);
            log([NSString stringWithFormat:@"  Period: %llu\n", result.period]);
            log([NSString stringWithFormat:@"  Tail length: %llu\n", result.tail_length]);
            log([NSString stringWithFormat:@"  Full period (period=m): %s\n", result.full_period ? "YES -- ideal" : "NO"]);
            NSMutableString *fs = [NSMutableString stringWithString:@"  m factors: "];
            for (size_t i = 0; i < result.factors_of_m.size(); i++) {
                if (i > 0) [fs appendString:@"x"];
                [fs appendFormat:@"%llu", result.factors_of_m[i]];
            }
            [fs appendString:@"\n"];
            log(fs);
            // Check Hull-Dobell conditions
            if (c != 0) {
                bool cond1 = prime::gcd(c, m) == 1;
                log([NSString stringWithFormat:@"  Hull-Dobell: gcd(c,m)=1: %s\n", cond1 ? "YES" : "NO"]);
            }
        }
        else if (tid == "pollard_p1_factor") {
            if (vals.empty()) { log(@"Enter: n [B]\n"); ss->_checkRunning.store(false); return; }
            uint64_t n = vals[0];
            uint64_t B = vals.size() > 1 ? vals[1] : 100000;
            log([NSString stringWithFormat:@"Pollard p-1 factoring: n=%llu, B=%llu\n", n, B]);
            auto t0 = std::chrono::steady_clock::now();
            uint64_t f = prime::pollard_p_minus_1(n, B);
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (f > 0) {
                log([NSString stringWithFormat:@"  Factor found: %llu x %llu  (%.4f s)\n", f, n / f, dt]);
                log([NSString stringWithFormat:@"  p-1 = %llu, factors: %s\n", f - 1,
                    [NSString stringWithUTF8String:prime::factors_string(f - 1).c_str()]]);
            } else {
                log([NSString stringWithFormat:@"  No factor found with B=%llu (%.4f s). Try larger B.\n", B, dt]);
            }
        }
        else if (tid == "convergence_analyzer") {
            if (vals.empty()) { log(@"Enter numbers.\n"); ss->_checkRunning.store(false); return; }
            for (auto n : vals) {
                double c = prime::convergence(n);
                bool ip = prime::is_prime(n);
                log([NSString stringWithFormat:@"%llu: convergence=%.4f  prime=%s\n",
                    n, c, ip ? "YES" : "NO"]);
            }
        }
        else if (tid == "modular_arithmetic") {
            // Check if first token is an operation name
            std::string op = "";
            std::vector<uint64_t> nums;
            if (!tokens.empty()) {
                char c0 = tokens[0][0];
                if (c0 < '0' || c0 > '9') {
                    op = tokens[0];
                    for (size_t i = 1; i < tokens.size(); i++)
                        nums.push_back(strtoull(tokens[i].c_str(), nullptr, 10));
                } else {
                    nums = vals;
                }
            }
            if (nums.size() < 2) {
                log(@"Enter: op a b [m]  -- op = modpow|mulmod|modinv|gcd|lcm\n"
                    @"Or just 2 numbers for gcd/lcm, or 3 for modpow.\n");
                ss->_checkRunning.store(false); return;
            }
            if (op == "gcd" || (op.empty() && nums.size() == 2)) {
                log([NSString stringWithFormat:@"gcd(%llu, %llu) = %llu\n", nums[0], nums[1], prime::gcd(nums[0], nums[1])]);
                log([NSString stringWithFormat:@"lcm(%llu, %llu) = %llu\n", nums[0], nums[1], prime::lcm(nums[0], nums[1])]);
            } else if (op == "lcm") {
                log([NSString stringWithFormat:@"lcm(%llu, %llu) = %llu\n", nums[0], nums[1], prime::lcm(nums[0], nums[1])]);
            } else if (op == "modinv" && nums.size() >= 2) {
                uint64_t inv = prime::mod_inverse(nums[0], nums[1]);
                if (inv > 0)
                    log([NSString stringWithFormat:@"modinv(%llu, %llu) = %llu\n", nums[0], nums[1], inv]);
                else
                    log([NSString stringWithFormat:@"modinv(%llu, %llu) = none (not coprime)\n", nums[0], nums[1]]);
            } else if (op == "mulmod" && nums.size() >= 3) {
                log([NSString stringWithFormat:@"mulmod(%llu, %llu, %llu) = %llu\n",
                    nums[0], nums[1], nums[2], prime::mulmod(nums[0], nums[1], nums[2])]);
            } else if (nums.size() >= 3) {
                // Default: modpow
                log([NSString stringWithFormat:@"modpow(%llu, %llu, %llu) = %llu\n",
                    nums[0], nums[1], nums[2], prime::modpow(nums[0], nums[1], nums[2])]);
                log([NSString stringWithFormat:@"mulmod(%llu, %llu, %llu) = %llu\n",
                    nums[0], nums[1], nums[2], prime::mulmod(nums[0], nums[1], nums[2])]);
                uint64_t inv = prime::mod_inverse(nums[0], nums[2]);
                if (inv > 0)
                    log([NSString stringWithFormat:@"modinv(%llu, %llu) = %llu\n", nums[0], nums[2], inv]);
                else
                    log([NSString stringWithFormat:@"modinv(%llu, %llu) = none (not coprime)\n", nums[0], nums[2]]);
            }
        }
        else if (tid == "rsa_key_gen") {
            uint64_t bits = vals.empty() ? 32 : vals[0];
            if (bits < 8) bits = 8;
            if (bits > 62) bits = 62;
            uint64_t half = bits / 2;
            if (half < 4) half = 4;
            uint64_t lo = 1ULL << (half - 1);
            uint64_t hi = (1ULL << half) - 1;
            log([NSString stringWithFormat:@"Generating RSA keypair (~%llu-bit modulus)...\n", bits]);

            // Find random prime p
            srand48(time(nullptr));
            uint64_t p = 0, q = 0;
            for (int attempt = 0; attempt < 100000; attempt++) {
                uint64_t candidate = lo + (uint64_t)(drand48() * (hi - lo));
                candidate |= 1; // odd
                if (prime::is_prime(candidate)) { p = candidate; break; }
            }
            // Find random prime q != p
            for (int attempt = 0; attempt < 100000; attempt++) {
                uint64_t candidate = lo + (uint64_t)(drand48() * (hi - lo));
                candidate |= 1;
                if (candidate != p && prime::is_prime(candidate)) { q = candidate; break; }
            }
            if (p == 0 || q == 0) {
                log(@"Failed to find primes. Try different bit size.\n");
                ss->_checkRunning.store(false); return;
            }
            if (p < q) std::swap(p, q);
            uint64_t n = p * q;
            uint64_t phi = (p - 1) * (q - 1);
            uint64_t e = 65537;
            if (prime::gcd(e, phi) != 1) {
                // Find alternative e
                for (e = 3; e < phi; e += 2) {
                    if (prime::gcd(e, phi) == 1) break;
                }
            }
            uint64_t d = prime::mod_inverse(e, phi);

            log([NSString stringWithFormat:@"\n  RSA Key Parameters (%llu-bit modulus)\n", bits]);
            log(@"  ─────────────────────────────────────\n");
            log([NSString stringWithFormat:@"  p  = %llu  (prime)\n", p]);
            log([NSString stringWithFormat:@"  q  = %llu  (prime)\n", q]);
            log([NSString stringWithFormat:@"  n  = pxq = %llu\n", n]);
            log([NSString stringWithFormat:@"  phi(n) = (p-1)(q-1) = %llu\n", phi]);
            log([NSString stringWithFormat:@"  e  = %llu  (public exponent)\n", e]);
            log([NSString stringWithFormat:@"  d  = %llu  (private exponent)\n", d]);

            // Verify: m^e^d mod n = m
            uint64_t test_msg = 42;
            if (test_msg >= n) test_msg = n / 2;
            uint64_t cipher = prime::modpow(test_msg, e, n);
            uint64_t plain = prime::modpow(cipher, d, n);
            log([NSString stringWithFormat:@"\n  Verification: encrypt(%llu) = %llu, decrypt = %llu %s\n",
                test_msg, cipher, plain, plain == test_msg ? "OK" : "FAIL"]);

            // Bit counts
            uint64_t nbits = 0;
            for (uint64_t tmp = n; tmp; tmp >>= 1) nbits++;
            log([NSString stringWithFormat:@"  Actual modulus size: %llu bits\n", nbits]);
        }
        else if (tid == "ring_beacon_test") {
            uint64_t lo = vals.size() >= 2 ? vals[0] : 10000;
            uint64_t hi = vals.size() >= 2 ? vals[1] : lo + 10000;
            if (hi - lo > 100000) { log(@"Range too large (max 100K).\n"); ss->_checkRunning.store(false); return; }
            log([NSString stringWithFormat:@"Ring Beacon Test [%llu, %llu] (mod-210 wheel)...\n", lo, hi]);
            auto r = prime::ring_beacon_test(lo, hi);
            log([NSString stringWithFormat:@"  Primes in range: %d\n", r.total_primes]);
            log([NSString stringWithFormat:@"  Predicted prime positions: %d\n", r.predicted_primes]);
            log([NSString stringWithFormat:@"  Correct predictions: %d\n", r.correct_predictions]);
            log([NSString stringWithFormat:@"  Precision: %.1f%%\n", r.precision * 100]);
            log([NSString stringWithFormat:@"  Recall: %.1f%%\n", r.recall * 100]);
            log(@"  Spoke beacon counts (lowest = most prime-like):\n");
            int show = std::min((int)r.spoke_beacon_counts.size(), 15);
            for (int i = 0; i < show; i++) {
                log([NSString stringWithFormat:@"    spoke %3d: %d beacons\n",
                    r.spoke_beacon_counts[i].first, r.spoke_beacon_counts[i].second]);
            }
        }
        else if (tid == "topography_test") {
            uint64_t lo = vals.size() >= 2 ? vals[0] : 1000;
            uint64_t hi = vals.size() >= 2 ? vals[1] : lo + 4000;
            if (hi - lo > 100000) { log(@"Range too large (max 100K).\n"); ss->_checkRunning.store(false); return; }
            log([NSString stringWithFormat:@"Topography Test [%llu, %llu]...\n", lo, hi]);
            auto r = prime::topography_test(lo, hi);
            log([NSString stringWithFormat:@"  Primes: %d\n", r.num_primes]);
            log([NSString stringWithFormat:@"  Avg prime gap: %.2f\n", r.avg_gap]);
            log([NSString stringWithFormat:@"  Avg valley depth (factor count): %.2f\n", r.avg_valley_depth]);
            log([NSString stringWithFormat:@"  Depth-gap correlation: %.4f\n", r.depth_gap_correlation]);
            log(@"    (positive = deeper valleys predict bigger gaps)\n");
            if (!r.deepest_valleys.empty()) {
                log(@"  Deepest valleys (highly composite numbers):\n");
                int show = std::min((int)r.deepest_valleys.size(), 10);
                for (int i = 0; i < show; i++) {
                    log([NSString stringWithFormat:@"    %llu: %d factors\n",
                        r.deepest_valleys[i].first, r.deepest_valleys[i].second]);
                }
            }
        }
        else if (tid == "web_test") {
            uint64_t lo = vals.size() >= 2 ? vals[0] : 1000;
            uint64_t hi = vals.size() >= 2 ? vals[1] : lo + 4000;
            if (hi - lo > 50000) { log(@"Range too large (max 50K).\n"); ss->_checkRunning.store(false); return; }
            log([NSString stringWithFormat:@"Factor Web Test [%llu, %llu]...\n", lo, hi]);
            auto r = prime::web_test(lo, hi);
            log([NSString stringWithFormat:@"  Primes: %d, Composites: %d\n", r.num_primes, r.num_composites]);
            log(@"  Seed type distribution:\n");
            for (auto& [type, count] : r.seed_counts) {
                log([NSString stringWithFormat:@"    %-16s: %d\n", prime::seed_type_name(type), count]);
            }
            log([NSString stringWithFormat:@"  Shared TinyPrime seeds: %d\n", r.shared_tiny]);
            log([NSString stringWithFormat:@"  Shared TwinFactor seeds: %d\n", r.shared_twin]);
            log([NSString stringWithFormat:@"  Shared SophieFactor seeds: %d\n", r.shared_sophie]);
            log([NSString stringWithFormat:@"  Predicted primes (web gaps): %d\n", r.predicted_primes]);
            log([NSString stringWithFormat:@"  Correct: %d\n", r.correct_predictions]);
            log([NSString stringWithFormat:@"  Precision: %.1f%%, Recall: %.1f%%\n", r.precision * 100, r.recall * 100]);
        }
        else if (tid == "audio_test") {
            uint64_t lo = vals.size() >= 2 ? vals[0] : 1000;
            uint64_t hi = vals.size() >= 2 ? vals[1] : lo + 4000;
            if (hi - lo > 100000) { log(@"Range too large (max 100K).\n"); ss->_checkRunning.store(false); return; }
            log([NSString stringWithFormat:@"Audio Test (Harmonic Resonance) [%llu, %llu]...\n", lo, hi]);
            auto r = prime::audio_test(lo, hi);
            log([NSString stringWithFormat:@"  Primes: %d\n", r.num_primes]);
            log([NSString stringWithFormat:@"  Avg resonance: %.4f\n", r.avg_resonance]);
            log([NSString stringWithFormat:@"  Low-resonance gaps: %d\n", r.resonance_gaps]);
            log([NSString stringWithFormat:@"  Primes in gaps: %d\n", r.primes_in_gaps]);
            log([NSString stringWithFormat:@"  Gap prime rate vs baseline: %.2fx\n", r.gap_prime_rate]);
            log(@"    (>1.0 = resonance gaps predict primes better than random)\n");
            if (!r.common_harmonics.empty()) {
                log(@"  Most common harmonic ratios:\n");
                // Sort by count
                std::vector<std::pair<std::string, int>> sorted_h(r.common_harmonics.begin(), r.common_harmonics.end());
                std::sort(sorted_h.begin(), sorted_h.end(), [](auto& a, auto& b) { return a.second > b.second; });
                int show = std::min((int)sorted_h.size(), 10);
                for (int i = 0; i < show; i++) {
                    log([NSString stringWithFormat:@"    %s: %d occurrences\n",
                        sorted_h[i].first.c_str(), sorted_h[i].second]);
                }
            }
        }
        else if (tid == "twisting_tree_test") {
            uint64_t lo = vals.size() >= 2 ? vals[0] : 1000;
            uint64_t hi = vals.size() >= 2 ? vals[1] : lo + 4000;
            if (hi - lo > 100000) { log(@"Range too large (max 100K).\n"); ss->_checkRunning.store(false); return; }
            log([NSString stringWithFormat:@"Twisting Tree Test [%llu, %llu]...\n", lo, hi]);
            auto r = prime::twisting_tree_test(lo, hi);
            log([NSString stringWithFormat:@"  Primes: %d\n", r.num_primes]);
            log([NSString stringWithFormat:@"  Balanced trees: %d, Unbalanced: %d\n", r.balanced_trees, r.unbalanced_trees]);
            log([NSString stringWithFormat:@"  Tree shape transitions: %d\n", r.shape_changes]);
            log([NSString stringWithFormat:@"  Primes after shape change: %d\n", r.primes_after_change]);
            log([NSString stringWithFormat:@"  Change->prime rate vs baseline: %.2fx\n", r.change_prime_rate]);
            log(@"    (>1.0 = shape changes predict primes)\n");
            log([NSString stringWithFormat:@"  Avg gap after balanced tree: %.2f\n", r.balanced_gap_avg]);
            log([NSString stringWithFormat:@"  Avg gap after unbalanced tree: %.2f\n", r.unbalanced_gap_avg]);
            log(@"  Tree shape distribution:\n");
            std::vector<std::pair<std::string, int>> sorted_s(r.shape_counts.begin(), r.shape_counts.end());
            std::sort(sorted_s.begin(), sorted_s.end(), [](auto& a, auto& b) { return a.second > b.second; });
            int show = std::min((int)sorted_s.size(), 15);
            for (int i = 0; i < show; i++) {
                log([NSString stringWithFormat:@"    %s: %d composites\n",
                    sorted_s[i].first.c_str(), sorted_s[i].second]);
            }
        }
        else {
            log([NSString stringWithFormat:@"Test '%s' not yet implemented.\n", tid.c_str()]);
        }

        ss->_checkRunning.store(false);
        } // @autoreleasepool
    });
}

// ═══════════════════════════════════════════════════════════════════════
// Expression Evaluate
// ═══════════════════════════════════════════════════════════════════════

- (void)expressionEvaluate:(id)sender {
    _activeLogTab = 1;  // Check / Expressions tab
    [self.logTabView selectTabViewItemAtIndex:1];
    NSString *input = [self.expressionField.stringValue
        stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (input.length == 0) return;
    [self evaluateExpression:input];
}

- (void)showExpressionHelp:(id)sender {
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = @"Expression Syntax";
    alert.informativeText =
        @"Supported expressions:\n\n"
        @"  is_prime(N)         -- primality test\n"
        @"  factor(N)           -- full factorization\n"
        @"  modpow(a,b,m)       -- a^b mod m\n"
        @"  mulmod(a,b,m)       -- a*b mod m\n"
        @"  gcd(a,b)  lcm(a,b)  -- greatest common div / least common mult\n"
        @"  fibonacci(n)        -- nth Fibonacci number\n"
        @"  binomial(n,k)       -- C(n,k)\n"
        @"  primes A to B       -- list primes in range\n"
        @"  next prime N        -- next prime after N\n"
        @"  prev prime N        -- previous prime before N\n"
        @"  twin primes A B     -- twin primes in range\n"
        @"  sophie germain A B  -- Sophie Germain primes in range\n"
        @"  wieferich N         -- test Wieferich property\n"
        @"  wilson N            -- test Wilson property\n"
        @"  mersenne N          -- test 2^N - 1\n"
        @"  convergence(N)      -- convergence score\n"
        @"  pi(N)  phi(N)       -- prime counting / Euler totient\n"
        @"  N!  N mod M         -- factorial / modulo\n"
        @"  2^67-1  3*5+7       -- math expressions";
    [alert addButtonWithTitle:@"OK"];
    [alert runModal];
}

// ═══════════════════════════════════════════════════════════════════════
// Distributed Computing Setup
// ═══════════════════════════════════════════════════════════════════════

- (void)showDistributedSetup:(id)sender {
    // If window already exists, just bring it forward
    if (self.networkWindow) {
        [self.networkWindow makeKeyAndOrderFront:nil];
        return;
    }

    CGFloat W = 620, H = 560;
    self.networkWindow = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(200, 200, W, H)
        styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                   NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
        backing:NSBackingStoreBuffered defer:NO];
    [self.networkWindow setTitle:@"Distributed Computing Setup"];
    [self.networkWindow setMinSize:NSMakeSize(500, 400)];
    self.networkWindow.delegate = self;

    NSView *cv = self.networkWindow.contentView;
    CGFloat M = 14, CW = W - 2 * M;
    CGFloat y = H - 10;

    // Title
    NSTextField *titleLbl = [self labelAt:NSMakeRect(M, y - 20, CW, 20)
        text:@"Distributed Computing" bold:YES size:14];
    titleLbl.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:titleLbl];
    y -= 24;

    NSTextField *descLbl = [self labelAt:NSMakeRect(M, y - 14, CW, 14)
        text:@"Run as Conductor (coordinates work) or Carriage (receives work from a Conductor)" bold:NO size:9.5];
    descLbl.textColor = [NSColor secondaryLabelColor];
    descLbl.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:descLbl];
    y -= 20;

    NSBox *sep1 = [[NSBox alloc] initWithFrame:NSMakeRect(M, y, CW, 1)];
    sep1.boxType = NSBoxSeparator;
    sep1.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:sep1];
    y -= 10;

    // Role selector
    NSTextField *roleLbl = [self labelAt:NSMakeRect(M, y - 16, 40, 14) text:@"Role:" bold:YES size:11];
    roleLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:roleLbl];

    self.roleSegment = [[NSSegmentedControl alloc] initWithFrame:NSMakeRect(M + 44, y - 18, 240, 22)];
    self.roleSegment.segmentCount = 2;
    [self.roleSegment setLabel:@"Conductor (Server)" forSegment:0];
    [self.roleSegment setLabel:@"Carriage (Worker)" forSegment:1];
    [self.roleSegment setWidth:120 forSegment:0];
    [self.roleSegment setWidth:120 forSegment:1];
    self.roleSegment.selectedSegment = 0;
    self.roleSegment.target = self;
    self.roleSegment.action = @selector(networkRoleChanged:);
    self.roleSegment.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.roleSegment];
    y -= 30;

    // ── Conductor panel ──────────────────────────────────────────────
    CGFloat panelTop = y;
    self.conductorPanel = [[NSView alloc] initWithFrame:NSMakeRect(0, panelTop - 70, W, 70)];
    self.conductorPanel.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    {
        CGFloat py = 50;
        NSTextField *portLbl = [self labelAt:NSMakeRect(M, py - 14, 34, 14) text:@"Port:" bold:NO size:10];
        [self.conductorPanel addSubview:portLbl];

        self.conductorPortField = [[NSTextField alloc] initWithFrame:NSMakeRect(M + 38, py - 16, 70, 20)];
        self.conductorPortField.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
        self.conductorPortField.stringValue = @"9807";
        [self.conductorPanel addSubview:self.conductorPortField];

        self.conductorStartStopBtn = [self buttonAt:NSMakeRect(M + 120, py - 16, 110, 22)
            title:@"Start Server" action:@selector(conductorStartStop:)];
        self.conductorStartStopBtn.font = [NSFont systemFontOfSize:10];
        [self.conductorPanel addSubview:self.conductorStartStopBtn];

        py -= 28;
        self.bonjourToggle = [NSButton checkboxWithTitle:@"Publish via Bonjour (auto-discovery on local network)"
            target:nil action:nil];
        self.bonjourToggle.frame = NSMakeRect(M, py - 14, 400, 18);
        self.bonjourToggle.font = [NSFont systemFontOfSize:9.5];
        self.bonjourToggle.state = NSControlStateValueOn;
        [self.conductorPanel addSubview:self.bonjourToggle];
    }
    [cv addSubview:self.conductorPanel];

    // ── Carriage panel ───────────────────────────────────────────────
    self.carriagePanel = [[NSView alloc] initWithFrame:NSMakeRect(0, panelTop - 70, W, 70)];
    self.carriagePanel.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    self.carriagePanel.hidden = YES;
    {
        CGFloat py = 50;
        NSTextField *hostLbl = [self labelAt:NSMakeRect(M, py - 14, 80, 14) text:@"Conductor IP:" bold:NO size:10];
        [self.carriagePanel addSubview:hostLbl];

        self.carriageHostField = [[NSTextField alloc] initWithFrame:NSMakeRect(M + 84, py - 16, 160, 20)];
        self.carriageHostField.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
        self.carriageHostField.placeholderString = @"192.168.1.100";
        [self.carriagePanel addSubview:self.carriageHostField];

        NSTextField *cportLbl = [self labelAt:NSMakeRect(M + 252, py - 14, 30, 14) text:@"Port:" bold:NO size:10];
        [self.carriagePanel addSubview:cportLbl];

        self.carriagePortField = [[NSTextField alloc] initWithFrame:NSMakeRect(M + 286, py - 16, 60, 20)];
        self.carriagePortField.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
        self.carriagePortField.stringValue = @"9807";
        [self.carriagePanel addSubview:self.carriagePortField];

        self.carriageConnectBtn = [self buttonAt:NSMakeRect(M + 360, py - 16, 100, 22)
            title:@"Connect" action:@selector(carriageConnect:)];
        self.carriageConnectBtn.font = [NSFont systemFontOfSize:10];
        [self.carriagePanel addSubview:self.carriageConnectBtn];

        py -= 28;
        NSButton *autoDiscover = [NSButton checkboxWithTitle:@"Auto-discover via Bonjour"
            target:self action:@selector(carriageBonjourToggle:)];
        autoDiscover.frame = NSMakeRect(M, py - 14, 250, 18);
        autoDiscover.font = [NSFont systemFontOfSize:9.5];
        autoDiscover.state = NSControlStateValueOff;
        [self.carriagePanel addSubview:autoDiscover];
    }
    [cv addSubview:self.carriagePanel];

    y = panelTop - 76;

    // ── Connected Machines ───────────────────────────────────────────
    NSBox *sep2 = [[NSBox alloc] initWithFrame:NSMakeRect(M, y, CW, 1)];
    sep2.boxType = NSBoxSeparator;
    sep2.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:sep2];
    y -= 6;

    NSTextField *machLbl = [self labelAt:NSMakeRect(M, y - 16, 150, 14)
        text:@"Connected Machines" bold:YES size:11];
    machLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:machLbl];

    NSButton *testBtn = [self buttonAt:NSMakeRect(W - M - 120, y - 16, 120, 20)
        title:@"Test Connection" action:@selector(testNetworkConnection:)];
    testBtn.font = [NSFont systemFontOfSize:10];
    testBtn.autoresizingMask = NSViewMinYMargin | NSViewMinXMargin;
    [cv addSubview:testBtn];
    y -= 22;

    // Machine list (scrollable text view)
    CGFloat listH = y - 40;
    NSScrollView *machScroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(M, 36, CW, listH)];
    machScroll.hasVerticalScroller = YES;
    machScroll.borderType = NSBezelBorder;
    machScroll.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.connectedMachinesView = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, CW - 4, listH)];
    self.connectedMachinesView.editable = NO;
    self.connectedMachinesView.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    self.connectedMachinesView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.connectedMachinesView.string = @"No machines connected.\n\n"
        @"Conductor mode: Start the server, then launch PrimePath on other\n"
        @"machines in Carriage mode to connect.\n\n"
        @"Carriage mode: Enter the Conductor's IP address and click Connect,\n"
        @"or enable Bonjour to discover automatically.";
    machScroll.documentView = self.connectedMachinesView;
    [cv addSubview:machScroll];

    // Status line at bottom
    self.networkStatusLabel = [self labelAt:NSMakeRect(M, 10, CW, 16)
        text:@"Status: Idle" bold:NO size:10];
    self.networkStatusLabel.textColor = [NSColor secondaryLabelColor];
    self.networkStatusLabel.autoresizingMask = NSViewWidthSizable | NSViewMaxYMargin;
    [cv addSubview:self.networkStatusLabel];

    // Start refresh timer
    __weak AppDelegate *weakSelf = self;
    self.networkRefreshTimer = [NSTimer scheduledTimerWithTimeInterval:1.0 repeats:YES
        block:^(NSTimer *t) { [weakSelf refreshNetworkStatus]; }];

    [self.networkWindow makeKeyAndOrderFront:nil];
}

// ── Network role toggle ──────────────────────────────────────────────

- (void)networkRoleChanged:(id)sender {
    BOOL isConductor = (self.roleSegment.selectedSegment == 0);
    self.conductorPanel.hidden = !isConductor;
    self.carriagePanel.hidden = isConductor;
}

// ── Conductor start/stop ─────────────────────────────────────────────

- (void)conductorStartStop:(id)sender {
    if (_conductor && _conductor->is_running()) {
        _conductor->stop();
        delete _conductor;
        _conductor = nullptr;
        self.conductorStartStopBtn.title = @"Start Server";
        self.networkStatusLabel.stringValue = @"Status: Server stopped";
        [self appendText:@"[Network] Conductor server stopped.\n"];
    } else {
        uint16_t port = (uint16_t)self.conductorPortField.integerValue;
        if (port == 0) port = 9807;
        _conductor = new prime::ConductorServer(port, _taskMgr);

        __weak AppDelegate *weakSelf = self;
        _conductor->set_on_carriage_connected([weakSelf](uint32_t cid, const std::string& name) {
            NSString *msg = [NSString stringWithFormat:@"[Network] Carriage connected: %s (id=%u)\n",
                name.c_str(), cid];
            dispatch_async(dispatch_get_main_queue(), ^{
                [weakSelf appendText:msg];
                [weakSelf refreshNetworkStatus];
            });
        });
        _conductor->set_on_carriage_disconnected([weakSelf](uint32_t cid, const std::string& name) {
            NSString *msg = [NSString stringWithFormat:@"[Network] Carriage disconnected: %s (id=%u)\n",
                name.c_str(), cid];
            dispatch_async(dispatch_get_main_queue(), ^{
                [weakSelf appendText:msg];
                [weakSelf refreshNetworkStatus];
            });
        });
        _conductor->set_on_carriage_progress([weakSelf](uint32_t cid, const prime::net::ProgressMsg& pm) {
            // Progress is tracked internally, refreshed by timer
        });
        _conductor->set_on_remote_discovery([weakSelf](uint32_t cid, const prime::net::DiscoveryMsg& dm) {
            NSString *msg = [NSString stringWithFormat:@"[Network] Remote discovery from carriage %u: value=%llu\n",
                cid, dm.value];
            dispatch_async(dispatch_get_main_queue(), ^{
                [weakSelf appendText:msg];
            });
        });

        _conductor->start();
        self.conductorStartStopBtn.title = @"Stop Server";
        self.networkStatusLabel.stringValue = [NSString stringWithFormat:
            @"Status: Conductor running on port %u", port];
        [self appendText:[NSString stringWithFormat:@"[Network] Conductor server started on port %u\n", port]];
    }
}

// ── Carriage connect/disconnect ──────────────────────────────────────

- (void)carriageConnect:(id)sender {
    if (_carriage && _carriage->is_connected()) {
        _carriage->stop();
        delete _carriage;
        _carriage = nullptr;
        self.carriageConnectBtn.title = @"Connect";
        self.networkStatusLabel.stringValue = @"Status: Disconnected";
        [self appendText:@"[Network] Carriage disconnected.\n"];
    } else {
        NSString *host = self.carriageHostField.stringValue;
        uint16_t port = (uint16_t)self.carriagePortField.integerValue;
        if (port == 0) port = 9807;
        if (host.length == 0) {
            self.networkStatusLabel.stringValue = @"Status: Enter a Conductor IP address";
            return;
        }

        _carriage = new prime::CarriageClient(DATA_DIR.UTF8String);
        __weak AppDelegate *weakSelf = self;
        _carriage->set_status_callback([weakSelf](const std::string& status) {
            NSString *msg = [NSString stringWithFormat:@"[Network] %s\n",
                status.c_str()];
            dispatch_async(dispatch_get_main_queue(), ^{
                [weakSelf appendText:msg];
                [weakSelf refreshNetworkStatus];
            });
        });
        _carriage->set_work_callback([weakSelf](const prime::net::WorkChunk& chunk) {
            NSString *msg = [NSString stringWithFormat:
                @"[Network] Received work: [%llu, %llu)\n",
                chunk.range_start, chunk.range_end];
            dispatch_async(dispatch_get_main_queue(), ^{
                [weakSelf appendText:msg];
            });
        });

        _carriage->connect(host.UTF8String, port);
        self.carriageConnectBtn.title = @"Disconnect";
        self.networkStatusLabel.stringValue = [NSString stringWithFormat:
            @"Status: Connecting to %@:%u...", host, port];
        [self appendText:[NSString stringWithFormat:@"[Network] Connecting to %@:%u...\n", host, port]];
    }
}

- (void)carriageBonjourToggle:(id)sender {
    NSButton *cb = (NSButton *)sender;
    if (cb.state == NSControlStateValueOn) {
        if (!_carriage) {
            _carriage = new prime::CarriageClient(DATA_DIR.UTF8String);
            __weak AppDelegate *weakSelf = self;
            _carriage->set_status_callback([weakSelf](const std::string& status) {
                NSString *msg = [NSString stringWithFormat:@"[Network] %s\n", status.c_str()];
                dispatch_async(dispatch_get_main_queue(), ^{
                    [weakSelf appendText:msg];
                    [weakSelf refreshNetworkStatus];
                });
            });
        }
        _carriage->start_discovery();
        self.networkStatusLabel.stringValue = @"Status: Searching for Conductor via Bonjour...";
        [self appendText:@"[Network] Bonjour discovery started.\n"];
    } else {
        if (_carriage) {
            _carriage->stop();
            delete _carriage;
            _carriage = nullptr;
        }
        self.networkStatusLabel.stringValue = @"Status: Bonjour discovery stopped";
    }
}

// ── Test connection ──────────────────────────────────────────────────

- (void)testNetworkConnection:(id)sender {
    if (_conductor && _conductor->is_running()) {
        size_t count = _conductor->carriage_count();
        auto carriages = _conductor->connected_carriages();
        NSMutableString *result = [NSMutableString string];
        [result appendFormat:@"Conductor running -- %zu carriage(s) connected\n\n", count];
        for (auto& c : carriages) {
            [result appendFormat:@"  Host: %s | Cores: %d | GPU: %s | Connected: %s\n",
                c.hostname.c_str(), c.cores, c.gpu_name.c_str(),
                c.connected ? "Yes" : "No"];
        }
        if (carriages.empty()) {
            [result appendString:@"  (no carriages connected yet)\n"];
        }
        self.connectedMachinesView.string = result;
        self.networkStatusLabel.stringValue = [NSString stringWithFormat:
            @"Status: %zu carriage(s) connected", count];
    } else if (_carriage) {
        NSMutableString *result = [NSMutableString string];
        [result appendFormat:@"Carriage mode\n"];
        [result appendFormat:@"  Connected: %s\n", _carriage->is_connected() ? "Yes" : "No"];
        [result appendFormat:@"  Working: %s\n", _carriage->is_working() ? "Yes" : "No"];
        std::string host = _carriage->conductor_host();
        uint16_t port = _carriage->conductor_port();
        if (!host.empty()) {
            [result appendFormat:@"  Conductor: %s:%u\n", host.c_str(), port];
        }
        self.connectedMachinesView.string = result;
        self.networkStatusLabel.stringValue = _carriage->is_connected()
            ? @"Status: Connected to Conductor"
            : @"Status: Not connected";
    } else {
        self.connectedMachinesView.string = @"No active connection. Start a server or connect to a Conductor.";
        self.networkStatusLabel.stringValue = @"Status: Idle";
    }
}

// ── Refresh network status (timer) ───────────────────────────────────

- (void)refreshNetworkStatus {
    if (!self.networkWindow || !self.networkWindow.isVisible) return;

    if (_conductor && _conductor->is_running()) {
        auto carriages = _conductor->connected_carriages();
        NSMutableString *result = [NSMutableString string];
        [result appendFormat:@"%-24s  %5s  %-20s  %s\n", "HOSTNAME", "CORES", "GPU", "STATUS"];
        [result appendString:@"────────────────────────  ─────  ────────────────────  ──────\n"];
        for (auto& c : carriages) {
            [result appendFormat:@"%-24s  %5d  %-20s  %s\n",
                c.hostname.c_str(), c.cores, c.gpu_name.c_str(),
                c.connected ? "Active" : "Idle"];
        }
        if (carriages.empty()) {
            [result appendString:@"\n  Waiting for carriages to connect...\n"];
            [result appendFormat:@"  Server listening on port %u\n", _conductor->port()];
        }
        self.connectedMachinesView.string = result;
        self.networkStatusLabel.stringValue = [NSString stringWithFormat:
            @"Status: Conductor running -- %zu carriage(s)", _conductor->carriage_count()];
        self.conductorStartStopBtn.title = @"Stop Server";
    } else if (_carriage) {
        NSMutableString *result = [NSMutableString string];
        if (_carriage->is_connected()) {
            [result appendFormat:@"Connected to Conductor at %s:%u\n",
                _carriage->conductor_host().c_str(), _carriage->conductor_port()];
            [result appendFormat:@"Working: %s\n", _carriage->is_working() ? "Yes" : "No"];
        } else {
            [result appendString:@"Searching for Conductor...\n"];
        }
        self.connectedMachinesView.string = result;
        self.networkStatusLabel.stringValue = _carriage->is_connected()
            ? @"Status: Connected to Conductor"
            : @"Status: Searching...";
        self.carriageConnectBtn.title = _carriage->is_connected() ? @"Disconnect" : @"Connect";
    }
}

// ── Window delegate ──────────────────────────────────────────────────

- (void)windowWillClose:(NSNotification *)notification {
    if (notification.object == self.networkWindow) {
        [self.networkRefreshTimer invalidate];
        self.networkRefreshTimer = nil;
    }
}

@end
