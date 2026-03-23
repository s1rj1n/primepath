#import "AppDelegate.h"
#import "MetalCompute.h"
#import "PrimeEngine.hpp"
#import "TaskManager.hpp"
#import "KnownPrimes.hpp"
#import "Benchmark.hpp"
#include <string>
#include <mach/mach.h>
#include <thread>
#include <atomic>
#include <algorithm>
#include <future>

static NSString *const DATA_DIR = @"/Users/sergeinester/Documents/primes/primelocations";

static NSString *formatNumber(uint64_t n) {
    NSNumberFormatter *fmt = [[NSNumberFormatter alloc] init];
    fmt.numberStyle = NSNumberFormatterDecimalStyle;
    fmt.groupingSeparator = @",";
    return [fmt stringFromNumber:@(n)];
}

// ═══════════════════════════════════════════════════════════════════════
// PrimePath v0.5 — Metal GPU + Multi-Core Prime Discovery
// ═══════════════════════════════════════════════════════════════════════

@interface AppDelegate () {
    prime::TaskManager *_taskMgr;
    prime::GPUBackend *_gpu;
    std::atomic<bool> _checkRunning;
    std::thread _checkThread;
    std::atomic<bool> _benchRunning;
    std::thread _benchThread;
    // CPU monitoring delta state
    uint64_t _prevIdleTicks;
    uint64_t _prevTotalTicks;
    // PrimeLocation predicted primes list (persists after test ends)
    std::vector<uint64_t> _predictedPrimes;
}

@property (strong) NSWindow *mainWindow;
@property (strong) NSTextView *resultView;
@property (strong) NSTextField *statusLabel;
@property (strong) NSTimer *refreshTimer;
@property (strong) NSTimer *frontierCheckTimer;
@property (strong) id eventMonitor;
@property (strong) NSMutableDictionary<NSNumber *, NSButton *> *taskButtons;
@property (strong) NSMutableDictionary<NSNumber *, NSTextField *> *taskLabels;
@property (strong) NSPopUpButton *taskSelectPopup;   // dropdown to pick which test to start
@property (strong) NSButton *taskStartBtn;            // Start/Pause for selected test
@property (strong) NSTextField *activeTaskListLabel;  // compact list of running/paused tasks

// Resource visualizer
@property (strong) NSProgressIndicator *cpuBar;
@property (strong) NSTextField *cpuLabel;
@property (strong) NSProgressIndicator *memBar;
@property (strong) NSTextField *memLabel;
@property (strong) NSTextField *gpuStatusLabel;
@property (strong) NSTextField *gpuDetailLabel;   // GPU utilization %, threads, batches
@property (strong) NSTextField *aluDetailLabel;    // NEON/SIMD stats from MatrixSieve
@property (strong) NSButton *disableVisualizerBtn; // checkbox to disable resource monitor

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
@property (strong) NSButton *primeFactorButton;      // "Run PrimeFactor" — factor composites using predicted primes
@property (strong) NSButton *checkAtDiscoveryButton;  // checkbox: run special tests on each predicted prime as found

// Benchmark
@property (strong) NSButton *benchmarkButton;
@property (strong) NSButton *benchmarkStopButton;

@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    _checkRunning = false;
    _prevIdleTicks = 0;
    _prevTotalTicks = 0;

    // Init GPU backend (abstract — auto-selects Metal, Vulkan, or CPU)
    _gpu = prime::create_best_backend();

    // Init Task Manager
    _taskMgr = new prime::TaskManager(DATA_DIR.UTF8String);
    _taskMgr->init_defaults();
    _taskMgr->set_gpu(_gpu);

    // Wire callbacks
    __weak AppDelegate *weakSelf = self;
    _taskMgr->set_log_callback([weakSelf](const std::string& msg) {
        NSString *s = [NSString stringWithFormat:@"%@\n",
            [NSString stringWithUTF8String:msg.c_str()]];
        dispatch_async(dispatch_get_main_queue(), ^{
            [weakSelf appendText:s];
        });
    });
    _taskMgr->set_discovery_callback([weakSelf](const prime::Discovery& d) {
        NSString *s = [NSString stringWithFormat:@">>> DISCOVERY: %s %llu",
            prime::task_name(d.type), d.value];
        if (d.value2 > 0) s = [s stringByAppendingFormat:@", %llu", d.value2];
        s = [s stringByAppendingString:@" <<<\n"];
        dispatch_async(dispatch_get_main_queue(), ^{
            [weakSelf appendText:s];
        });
    });

    // Load saved state (restores positions from previous session)
    _taskMgr->load_state();
    _taskMgr->save_state();

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
    [self.mainWindow setTitle:@"PrimePath v0.5 — Metal GPU Prime Discovery"];
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
        text:@"PrimePath — Metal GPU Prime Discovery" bold:YES size:14];
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

    // ── SYSTEM RESOURCES ─────────────────────────────────────────────
    CGFloat resY = y; // save for autoresizing
    CGFloat rx = M;

    NSTextField *cpuLbl = [self labelAt:NSMakeRect(rx, y - 14, 28, 14) text:@"CPU" bold:YES size:9.5];
    cpuLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:cpuLbl];
    rx += 28;
    self.cpuBar = [[NSProgressIndicator alloc] initWithFrame:NSMakeRect(rx, y - 12, 120, 11)];
    self.cpuBar.style = NSProgressIndicatorStyleBar;
    self.cpuBar.indeterminate = NO; self.cpuBar.minValue = 0; self.cpuBar.maxValue = 100;
    self.cpuBar.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.cpuBar];
    rx += 123;
    self.cpuLabel = [self labelAt:NSMakeRect(rx, y - 14, 38, 14) text:@"0%" bold:NO size:9.5];
    self.cpuLabel.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.cpuLabel];
    rx += 40;

    NSTextField *memLbl = [self labelAt:NSMakeRect(rx, y - 14, 28, 14) text:@"Mem" bold:YES size:9.5];
    memLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:memLbl];
    rx += 30;
    self.memBar = [[NSProgressIndicator alloc] initWithFrame:NSMakeRect(rx, y - 12, 100, 11)];
    self.memBar.style = NSProgressIndicatorStyleBar;
    self.memBar.indeterminate = NO; self.memBar.minValue = 0; self.memBar.maxValue = 100;
    self.memBar.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.memBar];
    rx += 103;
    self.memLabel = [self labelAt:NSMakeRect(rx, y - 14, 60, 14) text:@"0 MB" bold:NO size:9.5];
    self.memLabel.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.memLabel];
    rx += 64;

    self.gpuStatusLabel = [self labelAt:NSMakeRect(rx, y - 14, 200, 14)
        text:@"GPU: idle" bold:NO size:9.5];
    self.gpuStatusLabel.textColor = [NSColor secondaryLabelColor];
    self.gpuStatusLabel.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.gpuStatusLabel];

    self.disableVisualizerBtn = [NSButton checkboxWithTitle:@"Hide"
        target:self action:@selector(toggleVisualizer:)];
    self.disableVisualizerBtn.frame = NSMakeRect(W - M - 44, y - 14, 44, 14);
    self.disableVisualizerBtn.font = [NSFont systemFontOfSize:9];
    self.disableVisualizerBtn.state = NSControlStateValueOff;
    self.disableVisualizerBtn.autoresizingMask = NSViewMinYMargin | NSViewMinXMargin;
    [cv addSubview:self.disableVisualizerBtn];
    y -= 16;

    // Detail row
    self.gpuDetailLabel = [self labelAt:NSMakeRect(M + 28, y - 12, 400, 11)
        text:@"" bold:NO size:8.5];
    self.gpuDetailLabel.textColor = [NSColor tertiaryLabelColor];
    self.gpuDetailLabel.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:self.gpuDetailLabel];

    self.aluDetailLabel = [self labelAt:NSMakeRect(440, y - 12, 440, 11)
        text:@"" bold:NO size:8.5];
    self.aluDetailLabel.textColor = [NSColor tertiaryLabelColor];
    self.aluDetailLabel.autoresizingMask = NSViewMinYMargin | NSViewMinXMargin;
    [cv addSubview:self.aluDetailLabel];
    y -= 14;

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
    y -= 24;

    // Start point row — directly below the test selector dropdown
    NSTextField *startAtLbl = [self labelAt:NSMakeRect(M, y - 19, 52, 14) text:@"Start at:" bold:NO size:9.5];
    startAtLbl.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:startAtLbl];

    // Hidden startTaskPopup — synced to taskSelectPopup selection
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

    self.checkAtDiscoveryButton = [[NSButton alloc] initWithFrame:NSMakeRect(M + 340, y - 22, 150, 18)];
    self.checkAtDiscoveryButton.buttonType = NSButtonTypeSwitch;
    self.checkAtDiscoveryButton.title = @"CheckAtDiscovery";
    self.checkAtDiscoveryButton.font = [NSFont systemFontOfSize:9];
    self.checkAtDiscoveryButton.toolTip = @"Run Wieferich/WallSunSun/Wilson tests on each predicted prime";
    self.checkAtDiscoveryButton.state = NSControlStateValueOff;
    self.checkAtDiscoveryButton.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.checkAtDiscoveryButton];

    self.primeFactorButton = [self buttonAt:NSMakeRect(M + 490, y - 22, 110, 20)
        title:@"Run PrimeFactor" action:@selector(runPrimeFactor:)];
    self.primeFactorButton.font = [NSFont systemFontOfSize:10];
    self.primeFactorButton.hidden = YES;
    self.primeFactorButton.autoresizingMask = NSViewMinYMargin;
    [cv addSubview:self.primeFactorButton];

    self.benchmarkStopButton = nil;
    y -= 24;

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

    // ── OUTPUT LOG (fills remaining space) ───────────────────────────
    NSBox *sep3 = [[NSBox alloc] initWithFrame:NSMakeRect(M, y, CW, 1)];
    sep3.boxType = NSBoxSeparator;
    sep3.autoresizingMask = NSViewMinYMargin | NSViewWidthSizable;
    [cv addSubview:sep3];
    y -= 2;

    CGFloat logBottom = 4;
    CGFloat logHeight = y - logBottom;
    NSScrollView *scroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(M, logBottom, CW, logHeight)];
    scroll.hasVerticalScroller = YES;
    scroll.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.resultView = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, CW - 4, logHeight)];
    self.resultView.editable = NO;
    self.resultView.font = [NSFont monospacedSystemFontOfSize:10 weight:NSFontWeightRegular];
    self.resultView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.resultView.backgroundColor = [NSColor textBackgroundColor];
    scroll.documentView = self.resultView;
    [cv addSubview:scroll];

    // Welcome text
    [self appendText:@"PrimePath v0.5.0 -- Metal GPU Prime Discovery Engine\n"];
    [self appendText:[NSString stringWithFormat:@"GPU: %@ | Data: %@\n",
        [NSString stringWithUTF8String:_gpu->name().c_str()], DATA_DIR]];

    auto& kdb = prime::known_db();
    [self appendText:[NSString stringWithFormat:
        @"Known: %zu primes, %zu pseudoprimes\n",
        kdb.prime_count(), kdb.pseudoprime_count()]];

    if (_taskMgr->discoveries().size() > 0) {
        [self appendText:@"Discoveries: "];
        bool first = true;
        for (auto& d : _taskMgr->discoveries()) {
            if (!first) [self appendText:@", "];
            NSString *entry = [NSString stringWithFormat:@"%s(%llu",
                prime::task_name(d.type), d.value];
            if (d.value2 > 0) entry = [entry stringByAppendingFormat:@",%llu", d.value2];
            entry = [entry stringByAppendingString:@")"];
            [self appendText:entry];
            first = false;
        }
        [self appendText:@"\n"];
    }
    [self appendText:@"\nHelp menu or ? button for glossary. Ready.\n\n"];

    [self.mainWindow makeKeyAndOrderFront:nil];
}

// ── Check mode UI toggle ────────────────────────────────────────────

- (void)checkModeChanged:(id)sender {
    NSInteger mode = self.checkModePopup.indexOfSelectedItem;
    BOOL showTo = (mode >= 1); // show "to" for all modes except single
    self.checkToLabel.hidden = !showTo;
    self.checkToScroll.hidden = !showTo;
    // Update tooltip hints
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
    }
}

// ── Check dispatcher ────────────────────────────────────────────────

- (void)checkGo:(id)sender {
    // If a background check is running, stop it first
    if (_checkRunning.load()) {
        [self stopBackgroundCheck];
    }

    NSString *rawInput = self.checkFromField.string;
    NSInteger mode = self.checkModePopup.indexOfSelectedItem;

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
    }
}

- (void)checkStopAction:(id)sender {
    [self stopBackgroundCheck];
}

- (void)stopBackgroundCheck {
    _checkRunning.store(false);
    if (_checkThread.joinable()) {
        _checkThread.join();
    }
    dispatch_async(dispatch_get_main_queue(), ^{
        self.checkStopButton.enabled = NO;
        self.checkGoButton.enabled = YES;
        [self appendText:@"Check stopped.\n"];
    });
}

// ── Check Single ────────────────────────────────────────────────────

- (void)runCheckSingle:(uint64_t)n {
    auto& kdb = prime::known_db();

    // ── Check known database first (instant, no computation) ──
    if (kdb.is_known(n)) {
        auto entries = kdb.get_entries(n);
        if (kdb.is_known_prime(n)) {
            // Known prime — show all classifications
            NSMutableString *classes = [NSMutableString string];
            for (auto* e : entries) {
                if (classes.length > 0) [classes appendString:@", "];
                [classes appendFormat:@"%s", prime::known_class_name(e->kclass)];
            }
            NSString *desc = @"";
            if (entries.size() > 0 && entries[0]->description[0] != '\0') {
                desc = [NSString stringWithFormat:@" — %s", entries[0]->description];
            }
            [self appendText:[NSString stringWithFormat:
                @"✓ KNOWN PRIME: %@ [%@]%@ (0μs — database lookup)\n",
                formatNumber(n), classes, desc]];
        } else {
            // Known pseudoprime — show factors
            auto* e = entries[0];
            std::string factors = prime::factors_comma_string(n);
            NSString *fstr = factors.empty() ? @"" :
                [NSString stringWithFormat:@" = %s", factors.c_str()];
            [self appendText:[NSString stringWithFormat:
                @"✗ KNOWN %s: %@%@ — %s (0μs — database lookup)\n",
                prime::known_class_name(e->kclass), formatNumber(n), fstr, e->description]];
        }
        return;
    }

    // ── Not in database — compute ──
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
            @"✓ PRIME: %@ (%.1fμs)\n", formatNumber(n), dt]];
        if (!pinch_hits.empty()) {
            // Primes shouldn't have pinch hits (unless n itself appears) — flag it
            [self appendText:[NSString stringWithFormat:
                @"  ⚠ PinchFactor found %zu hit(s) on prime? (%.1fμs)\n",
                pinch_hits.size(), pinch_dt]];
        }
    } else {
        bool crt = prime::crt_reject(n);
        NSString *method = crt ? @"CRT" : @"Miller-Rabin";
        std::string factors = prime::factors_comma_string(n);
        NSString *fstr = factors.empty() ? @"" :
            [NSString stringWithFormat:@" = %s", factors.c_str()];
        [self appendText:[NSString stringWithFormat:
            @"✗ COMPOSITE: %@%@ (caught by %@, %.1fμs)\n", formatNumber(n), fstr, method, dt]];

        // Show Pinch Factor results
        if (pinch_hits.empty()) {
            [self appendText:[NSString stringWithFormat:
                @"  PinchFactor: no digit-structural divisors found (%.1fμs)\n", pinch_dt]];
        } else {
            [self appendText:[NSString stringWithFormat:
                @"  PinchFactor: %zu divisor(s) found via digit splits (%.1fμs):\n",
                pinch_hits.size(), pinch_dt]];
            for (auto& h : pinch_hits) {
                [self appendText:[NSString stringWithFormat:
                    @"    pos %d: [%@|%@] → %@ divides N  (%s)\n",
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
                @"  Lucky7s: no round-number proximity divisors found (%.1fμs)\n", l7_dt]];
        } else {
            [self appendText:[NSString stringWithFormat:
                @"  Lucky7s: %zu divisor(s) found near powers of 10 (%.1fμs):\n",
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

        // DivisorWeb — digit-by-digit factor sieve
        // Only run on numbers small enough to complete quickly (< ~10^12 for base 10)
        uint64_t web_limit = 1000000000000ULL; // 10^12
        if (n <= web_limit) {
            int bases[] = {10, 60};
            for (int base : bases) {
                auto web = prime::divisor_web(n, base);
                [self appendText:[NSString stringWithFormat:
                    @"  DivisorWeb (base %d): %zu divisors, %zu prime factors (%.1fμs)\n",
                    base, web.all_divisors.size(), web.prime_divisors.size(), web.elapsed_us]];
                for (auto& lv : web.levels) {
                    NSMutableString *divStr = [NSMutableString string];
                    for (uint64_t d : lv.divisors) {
                        if (divStr.length > 0) [divStr appendString:@" "];
                        [divStr appendFormat:@"%@", formatNumber(d)];
                    }
                    NSString *hitStr = lv.divisors.empty() ? @"—" : divStr;
                    [self appendText:[NSString stringWithFormat:
                        @"    L%d [%@..%@] tested:%llu pruned:%llu+%llu → %@\n",
                        lv.level, formatNumber(lv.range_lo), formatNumber(lv.range_hi),
                        lv.tested, lv.pruned_composite, lv.pruned_modular,
                        hitStr]];
                }
                if (!web.prime_divisors.empty()) {
                    NSMutableString *pStr = [NSMutableString string];
                    for (uint64_t p : web.prime_divisors) {
                        if (pStr.length > 0) [pStr appendString:@" × "];
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

            // Split: 30% CPU, 70% GPU — both test in parallel
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
    _checkThread.detach();
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
    _checkThread.detach();
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

        // Phase 1: Score candidates using convergence (CPU-intensive)
        struct Candidate {
            uint64_t value;
            double score;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(windowSize / 2);

        uint64_t pos = (from % 2 == 0 && from > 2) ? from + 1 : from;

        // Phase 0: Run MatrixSieve on the candidate range — almost zero overhead,
        // eliminates ~77% of candidates divisible by primes {3,5,7,11,13,17,19,23,29,31}
        // before we even compute convergence scores.
        const uint32_t sieveCount = (uint32_t)std::min((uint64_t)windowSize, (uint64_t)2000000);
        std::vector<uint8_t> sieveMask(sieveCount, 1);
        if (taskMgr) {
            // Use MatrixSieve pre-filter (NEON accelerated)
            prime::MatrixSieve quickSieve;
            quickSieve.sieve_block(pos, sieveCount, sieveMask.data());
        }

        uint64_t sieveRejected = 0;
        for (uint64_t i = 0; i < windowSize && ss->_checkRunning.load(); i++) {
            if (i % 5000 == 0) [ss throttleCheckIfNeeded];
            uint64_t n = pos + i * 2;
            if (n < 3) continue;
            // MatrixSieve pre-filter (checks index i*2 in the sieve block starting at pos)
            uint32_t sieveIdx = (uint32_t)(i * 2);
            if (sieveIdx < sieveCount && !sieveMask[sieveIdx]) {
                sieveRejected++;
                continue;
            }
            if (!prime::WHEEL.valid(n)) continue;
            if (prime::crt_reject(n)) continue;
            double score = prime::convergence(n, 12);
            if (score > -900.0) {
                candidates.push_back({n, score});
            }
        }

        {
            NSString *sieveMsg = [NSString stringWithFormat:
                @"  MatrixSieve pre-filter: rejected %@ of %@ candidates (%.1f%%)\n",
                formatNumber(sieveRejected), formatNumber(windowSize),
                windowSize > 0 ? 100.0 * sieveRejected / windowSize : 0.0];
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

            // Collect results — save confirmed primes, optionally run special tests
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
                    @"  ★ PREDICTED PRIME #%llu: %@ (score: %.2f, rank: %llu)\n",
                    found, formatNumber(val), score, tested];
                dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf appendText:msg2]; });

                // CheckAtDiscovery: test this prime against special categories
                if (checkAtDiscovery) {
                    // Wieferich test: 2^(p-1) ≡ 1 (mod p²)
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
                        @"  %@ predicted primes stored — click 'PrimeFactor' to use them for factoring\n",
                        formatNumber(foundCopy)]];
                }
            }
        });
        ss->_checkRunning.store(false);
    });
    _checkThread.detach();
}

// ── Run PrimeFactor — use predicted primes to factor numbers ─────────

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
        // No target specified — factor composites near the predicted primes
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
                // Predicted primes didn't help — use standard factoring
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
                @"Wall-Sun-Sun: PrimeGrid verified to 2^64 (Dec 2022). ZERO known — any find is historic!";
            break;
        case prime::TaskType::Wilson:
            // 2×10^13 verified. Start just past that.
            // 2^44 = 17.6T ≈ 1.76×10^13, so 2^44+1 is just below. Use 2^45+1 ≈ 3.5×10^13.
            self.startNumberField.stringValue = @"2";
            self.startPowerField.stringValue = @"45";
            self.startHintLabel.stringValue =
                @"Wilson: Verified to 2x10^13 (2012). 2^45+1 ≈ 3.5x10^13. Only 3 known: 5, 13, 563.";
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
            [self appendText:@"Invalid start point — need a number >= 3\n"];
            return;
        }
        start_pos = base_val;
        desc = [NSString stringWithFormat:@"%@", formatNumber(start_pos)];
    } else {
        if (base_val < 2) {
            [self appendText:@"Invalid start point — need base >= 2\n"];
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
        @"Set %s start → %@\n", prime::task_name(t), desc]];
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
        @"Copyright": @"Development by Sergei Nester\nsnester@viewbuild.com",
        @"ApplicationVersion": @"0.5",
        @"Version": @"0.5.0",
    };
    [[NSApplication sharedApplication] orderFrontStandardAboutPanelWithOptions:opts];
}

- (void)showInfoPanel:(id)sender {
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = @"Log Output Glossary";
    alert.informativeText =
        @"GPU flush: 281318 WSS → GPU | shadow: avg=128 [100-214] 26546 suspicious\n\n"

        @"GPU flush — CPU sieve buffer is full, candidates sent to GPU for testing.\n\n"
        @"281,318 — Number of candidate primes in this batch.\n\n"
        @"WSS — Wall-Sun-Sun prime search. Looking for primes p where p² divides "
        @"F(p-(p/5)). None have ever been found.\n\n"
        @"→ GPU — Candidates dispatched to Metal GPU for modular arithmetic.\n\n"
        @"shadow — Even-shadow pre-filter. Each candidate gets a score based on how "
        @"well the divisors of p±1 constrain the computation. Higher = better.\n\n"
        @"avg=128 — Average shadow score across all candidates in the batch.\n\n"
        @"[100-214] — Score range: lowest 100, highest 214.\n\n"
        @"suspicious — Candidates with low shadow scores (poor p±1 divisor structure). "
        @"Less likely to be interesting but still tested. Reordered to back of batch.\n\n"

        @"── Other terms ──\n\n"
        @"pos: — Current search position (the number being tested).\n\n"
        @"found: — Count of discoveries (primes/factors found so far).\n\n"
        @"/s — Candidates tested per second.\n\n"
        @"GPU util — Percentage of time the GPU is actively computing.\n\n"
        @"threads — Total GPU threads dispatched since launch.\n\n"
        @"batches — Number of GPU command buffers submitted.\n\n"
        @"ms/batch — Average GPU execution time per batch.\n\n"
        @"ALU/NEON — CPU SIMD sieve stats. 'tested' = candidates sieved, "
        @"'rejected' = composites filtered out before GPU.\n\n"
        @"Predictor — Pseudoprime predictor stats. Carm = Carmichael numbers, "
        @"SPRP2 = strong probable primes base 2, frontier = highest value tested.";
    alert.alertStyle = NSAlertStyleInformational;
    [alert addButtonWithTitle:@"OK"];
    [alert runModal];
}

- (void)showHelpPanel:(id)sender {
    NSWindow *helpWin = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(200, 200, 600, 520)
        styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable)
        backing:NSBackingStoreBuffered defer:NO];
    helpWin.title = @"PrimePath Help";
    helpWin.releasedWhenClosed = NO;

    NSScrollView *sv = [[NSScrollView alloc] initWithFrame:helpWin.contentView.bounds];
    sv.hasVerticalScroller = YES;
    sv.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    NSTextView *tv = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, 580, 800)];
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
        @"Mersenne trial factoring and Fermat factor searching.\n\n"

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

        @"LOG OUTPUT GLOSSARY\n"
        @"-------------------\n"
        @"  GPU flush      CPU sieve buffer full, batch sent to GPU.\n"
        @"  WSS/Wief       Abbreviations for Wall-Sun-Sun / Wieferich.\n"
        @"  -> GPU         Candidates dispatched to Metal GPU.\n"
        @"  shadow         Even-shadow pre-filter score. Higher = better candidate.\n"
        @"  avg/min/max    Shadow score statistics for the batch.\n"
        @"  suspicious     Low-score candidates (poor p+/-1 divisor structure).\n"
        @"  fully-factored Candidates whose p-1 is completely factored (best quality).\n"
        @"  pos:           Current search position.\n"
        @"  found:         Discoveries made so far.\n"
        @"  /s             Candidates tested per second.\n\n"

        @"RESOURCE MONITOR\n"
        @"----------------\n"
        @"  CPU bar        System-wide CPU usage.\n"
        @"  Mem bar        App memory usage (resident size).\n"
        @"  GPU util       Percentage of time GPU is actively computing.\n"
        @"  threads        Total GPU threads dispatched since launch.\n"
        @"  batches        Number of GPU command buffers submitted.\n"
        @"  ms/batch       Average GPU execution time per batch.\n"
        @"  ALU/NEON       CPU SIMD sieve statistics.\n"
        @"                 tested = candidates sieved, rejected = composites filtered.\n"
        @"  Predictor      Pseudoprime predictor. Carm = Carmichael numbers,\n"
        @"                 SPRP2 = strong probable primes base 2.\n\n"

        @"SET TEST BOUNDS\n"
        @"---------------\n"
        @"Set the starting position for any search. Enter base ^ exponent + 1.\n"
        @"Example: 2 ^ 64 + 1 starts searching from 2^64 + 1.\n\n"

        @"CHECK & TOOLS\n"
        @"-------------\n"
        @"  Check Single   Test if a single number is prime.\n"
        @"  Check From     Find primes starting from a number.\n"
        @"  Check Linear   Enumerate all primes in a range.\n"
        @"  PrimeLocation  Predict prime locations using the PrimeLocation algorithm.\n"
        @"  Benchmark      Run GPU and CPU performance benchmarks.\n"
        @"  CheckAtDiscovery  Run special tests on each predicted prime as found.\n\n"

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
    if (_benchRunning.load()) return;
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
    _benchThread.detach();
}

- (void)stopBenchmark:(id)sender {
    _benchRunning.store(false);
    self.benchmarkButton.enabled = YES;
    self.benchmarkStopButton.enabled = NO;
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
    // PrimeGrid verified Wieferich/WSS to 2^64 ≈ 1.84×10^19.
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
                    if ([html containsString:@"2*10^13"] || [html containsString:@"2×10^13"]) {
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
                    // Frontier is ~2^64 — use safe start below overflow ceiling
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
                    @"✓ %@ search at %@ — past known frontier.\n",
                    f.name, formatNumber(our_pos)];
                dispatch_async(dispatch_get_main_queue(), ^{
                    [weakSelf appendText:msg];
                });
            }
        }] resume];
    }
}

- (void)refreshStats {
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

    self.statusLabel.stringValue = [NSString stringWithFormat:
        @"Running: %d | Total discoveries: %@ | GPU: %@",
        running, formatNumber(total_found),
        [NSString stringWithUTF8String:_gpu->name().c_str()]];
}

- (void)toggleVisualizer:(id)sender {
    BOOL disabled = (self.disableVisualizerBtn.state == NSControlStateValueOn);
    self.cpuBar.hidden = disabled;
    self.cpuLabel.hidden = disabled;
    self.memBar.hidden = disabled;
    self.memLabel.hidden = disabled;
    self.gpuStatusLabel.hidden = disabled;
    self.gpuDetailLabel.hidden = disabled;
    self.aluDetailLabel.hidden = disabled;
}

- (void)updateResourceMonitor {
    // Skip updates if visualizer is disabled (saves CPU cycles for max speed)
    if (self.disableVisualizerBtn.state == NSControlStateValueOn) return;

    // ── CPU usage (system-wide) ──
    host_cpu_load_info_data_t cpuinfo;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                        (host_info_t)&cpuinfo, &count) == KERN_SUCCESS) {
        uint64_t totalTicks = 0;
        for (int i = 0; i < CPU_STATE_MAX; i++)
            totalTicks += cpuinfo.cpu_ticks[i];
        uint64_t idleTicks = cpuinfo.cpu_ticks[CPU_STATE_IDLE];

        if (_prevTotalTicks > 0) {
            uint64_t totalDelta = totalTicks - _prevTotalTicks;
            uint64_t idleDelta = idleTicks - _prevIdleTicks;
            if (totalDelta > 0) {
                double cpuPct = 100.0 * (1.0 - (double)idleDelta / (double)totalDelta);
                self.cpuBar.doubleValue = cpuPct;
                self.cpuLabel.stringValue = [NSString stringWithFormat:@"%.0f%%", cpuPct];
                if (cpuPct > 80) {
                    self.cpuLabel.textColor = [NSColor systemRedColor];
                } else if (cpuPct > 50) {
                    self.cpuLabel.textColor = [NSColor systemOrangeColor];
                } else {
                    self.cpuLabel.textColor = [NSColor colorWithSRGBRed:0.0 green:0.55 blue:0.0 alpha:1.0];
                }
            }
        }
        _prevTotalTicks = totalTicks;
        _prevIdleTicks = idleTicks;
    }

    // ── App memory usage ──
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
        uint64_t mb = info.resident_size / (1024 * 1024);
        double memPct = std::min(100.0, (double)mb / 2048.0 * 100.0);
        self.memBar.doubleValue = memPct;
        self.memLabel.stringValue = [NSString stringWithFormat:@"%@ MB", formatNumber(mb)];
        if (mb > 1024) {
            self.memLabel.textColor = [NSColor systemRedColor];
        } else if (mb > 512) {
            self.memLabel.textColor = [NSColor systemOrangeColor];
        } else {
            self.memLabel.textColor = [NSColor labelColor];
        }
    }

    // ── GPU status + detail ──
    int runningTasks = 0, gpuTasks = 0, cpuOnlyTasks = 0;
    uint64_t totalRate = 0;
    auto& tasks = _taskMgr->tasks();
    for (auto& [type, task] : tasks) {
        if (task.status == prime::TaskStatus::Running) {
            runningTasks++;
            totalRate += (uint64_t)task.rate;
            // GPU-using tasks: Wieferich, WallSunSun, Sophie, Wilson
            if (type == prime::TaskType::Wieferich ||
                type == prime::TaskType::WallSunSun ||
                type == prime::TaskType::SophieGermain ||
                type == prime::TaskType::Wilson) {
                gpuTasks++;
            } else {
                cpuOnlyTasks++;
            }
        }
    }

    // GPU utilization from Metal command buffer timing
    double gpuUtil = _gpu->gpu_utilization() * 100.0;
    uint64_t gpuThreads = _gpu->total_threads_dispatched();
    uint64_t gpuBatches = _gpu->total_batches_dispatched();
    double avgMs = _gpu->avg_gpu_time_ms();

    if (gpuBatches > 0) {
        self.gpuDetailLabel.stringValue = [NSString stringWithFormat:
            @"GPU: %.1f%% util | %@ threads | %@ batches | %.2fms/batch",
            gpuUtil, formatNumber(gpuThreads), formatNumber(gpuBatches), avgMs];
    } else if (gpuTasks > 0) {
        self.gpuDetailLabel.stringValue = [NSString stringWithFormat:
            @"GPU: accumulating (%d GPU tasks) — dispatches when buffer fills",
            gpuTasks];
    } else if (runningTasks > 0) {
        self.gpuDetailLabel.stringValue = [NSString stringWithFormat:
            @"GPU: idle — %d running tasks are CPU-only (start Wieferich/WallSunSun/Sophie for GPU)",
            cpuOnlyTasks];
    } else {
        self.gpuDetailLabel.stringValue = @"GPU: no tasks running";
    }

    if (gpuUtil > 50) {
        self.gpuDetailLabel.textColor = [NSColor colorWithSRGBRed:0.0 green:0.55 blue:0.0 alpha:1.0];
    } else if (gpuUtil > 10 || gpuTasks > 0) {
        self.gpuDetailLabel.textColor = [NSColor systemOrangeColor];
    } else {
        self.gpuDetailLabel.textColor = [NSColor tertiaryLabelColor];
    }

    // ALU/NEON stats from MatrixSieve + Pseudoprime Predictor
    auto* ms = _taskMgr->matrix_sieve();
    auto* pp = _taskMgr->predictor();
    if (ms) {
        uint64_t tested = ms->total_tested();
        uint64_t rejected = ms->total_rejected();
        double rejPct = tested > 0 ? (100.0 * rejected / tested) : 0.0;
        NSString *ppStr = @"";
        if (pp && pp->count() > 0) {
            ppStr = [NSString stringWithFormat:@" | Predictor: %zu Carm + %zu SPRP2 → %@",
                     pp->carmichael_count(), pp->sprp2_count(),
                     formatNumber(pp->frontier())];
        }
        self.aluDetailLabel.stringValue = [NSString stringWithFormat:
            @"ALU/NEON: %@ tested, %@ rejected (%.1f%%)%@",
            formatNumber(tested), formatNumber(rejected), rejPct, ppStr];
        if (tested > 0) {
            self.aluDetailLabel.textColor = [NSColor colorWithSRGBRed:0.0 green:0.55 blue:0.0 alpha:1.0];
        }
    }

    if (runningTasks > 0 && _gpu->available()) {
        self.gpuStatusLabel.stringValue = [NSString stringWithFormat:
            @"GPU: %d tasks | %@/s", runningTasks, formatNumber(totalRate)];
        self.gpuStatusLabel.textColor = [NSColor colorWithSRGBRed:0.0 green:0.55 blue:0.0 alpha:1.0];
    } else if (runningTasks > 0) {
        self.gpuStatusLabel.stringValue = [NSString stringWithFormat:
            @"CPU: %d tasks | %@/s", runningTasks, formatNumber(totalRate)];
        self.gpuStatusLabel.textColor = [NSColor systemOrangeColor];
    } else {
        self.gpuStatusLabel.stringValue = @"GPU: idle";
        self.gpuStatusLabel.textColor = [NSColor secondaryLabelColor];
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

- (void)appendText:(NSString *)text {
    NSAttributedString *attr = [[NSAttributedString alloc]
        initWithString:text
        attributes:@{
            NSFontAttributeName: [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular],
            NSForegroundColorAttributeName: [NSColor labelColor]
        }];
    [self.resultView.textStorage appendAttributedString:attr];
    [self.resultView scrollRangeToVisible:NSMakeRange(self.resultView.string.length, 0)];
    // Prevent macOS from thinking the window has an unsaved document
    [self.mainWindow setDocumentEdited:NO];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender {
    // Stop background check
    _checkRunning.store(false);
    if (_checkThread.joinable()) {
        _checkThread.join();
    }
    if (self.eventMonitor) {
        [NSEvent removeMonitor:self.eventMonitor];
        self.eventMonitor = nil;
    }
    if (_taskMgr) {
        _taskMgr->stop_all();
        delete _taskMgr;
        _taskMgr = nullptr;
    }
    if (_gpu) {
        delete _gpu;
        _gpu = nullptr;
    }
    return NSTerminateNow;
}

@end
