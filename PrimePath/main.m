#import <Cocoa/Cocoa.h>
#import "AppDelegate.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        AppDelegate *delegate = [[AppDelegate alloc] init];
        app.delegate = delegate;
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];

        // Create main menu
        NSMenu *menubar = [[NSMenu alloc] init];

        // App menu
        NSMenuItem *appMenuItem = [[NSMenuItem alloc] init];
        [menubar addItem:appMenuItem];
        NSMenu *appMenu = [[NSMenu alloc] initWithTitle:@"PrimePath"];
        [appMenu addItemWithTitle:@"About PrimePath"
                           action:@selector(showAboutPanel:)
                    keyEquivalent:@""];
        [appMenu addItem:[NSMenuItem separatorItem]];
        [appMenu addItemWithTitle:@"Quit PrimePath"
                           action:@selector(terminate:)
                    keyEquivalent:@"q"];
        appMenuItem.submenu = appMenu;

        // Hidden edit menu — keeps Cmd+C/V/X/A working in text fields
        NSMenuItem *editMenuItem = [[NSMenuItem alloc] init];
        [menubar addItem:editMenuItem];
        NSMenu *editMenu = [[NSMenu alloc] initWithTitle:@"Edit"];
        [editMenu addItemWithTitle:@"Undo" action:@selector(undo:) keyEquivalent:@"z"];
        [editMenu addItemWithTitle:@"Redo" action:@selector(redo:) keyEquivalent:@"Z"];
        [editMenu addItem:[NSMenuItem separatorItem]];
        [editMenu addItemWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"];
        [editMenu addItemWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"];
        [editMenu addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
        [editMenu addItemWithTitle:@"Select All" action:@selector(selectAll:) keyEquivalent:@"a"];
        editMenuItem.submenu = editMenu;
        editMenuItem.hidden = YES;

        // Network menu
        NSMenuItem *networkMenuItem = [[NSMenuItem alloc] init];
        [menubar addItem:networkMenuItem];
        NSMenu *networkMenu = [[NSMenu alloc] initWithTitle:@"Network"];
        NSMenuItem *distItem = [networkMenu addItemWithTitle:@"Distributed Setup..."
                            action:@selector(showDistributedSetup:)
                     keyEquivalent:@"D"];
        distItem.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagShift;
        networkMenuItem.submenu = networkMenu;

        // Help menu
        NSMenuItem *helpMenuItem = [[NSMenuItem alloc] init];
        [menubar addItem:helpMenuItem];
        NSMenu *helpMenu = [[NSMenu alloc] initWithTitle:@"Help"];
        [helpMenu addItemWithTitle:@"PrimePath Help"
                            action:@selector(showHelpPanel:)
                     keyEquivalent:@"?"];
        helpMenuItem.submenu = helpMenu;

        [app setMainMenu:menubar];

        [app activateIgnoringOtherApps:YES];
        [app run];
    }
    return 0;
}
