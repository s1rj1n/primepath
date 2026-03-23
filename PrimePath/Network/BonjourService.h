#pragma once
#import <Foundation/Foundation.h>

// ═══════════════════════════════════════════════════════════════════════
// BonjourPublisher — publishes a _primepath._tcp. service on the local
//                    network so Carriages can discover the Conductor.
// ═══════════════════════════════════════════════════════════════════════

@interface BonjourPublisher : NSObject <NSNetServiceDelegate>

@property (nonatomic, readonly) BOOL isPublishing;
@property (nonatomic, readonly) uint16_t port;

- (instancetype)initWithPort:(uint16_t)port;
- (void)start;
- (void)stop;

@end

// ═══════════════════════════════════════════════════════════════════════
// BonjourBrowser — browses for _primepath._tcp. services.
//                  Delivers found/lost callbacks with hostname + port.
// ═══════════════════════════════════════════════════════════════════════

typedef void (^BonjourServiceFoundBlock)(NSString *hostname, uint16_t port);
typedef void (^BonjourServiceLostBlock)(NSString *hostname);

@interface BonjourBrowser : NSObject <NSNetServiceBrowserDelegate, NSNetServiceDelegate>

@property (nonatomic, copy) BonjourServiceFoundBlock onServiceFound;
@property (nonatomic, copy) BonjourServiceLostBlock  onServiceLost;
@property (nonatomic, readonly) BOOL isBrowsing;

- (void)startBrowsing;
- (void)stopBrowsing;

@end
