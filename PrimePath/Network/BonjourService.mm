#import "BonjourService.h"
#include <arpa/inet.h>

static NSString *const kServiceType = @"_primepath._tcp.";
static NSString *const kServiceDomain = @"local.";

// ═══════════════════════════════════════════════════════════════════════
// BonjourPublisher
// ═══════════════════════════════════════════════════════════════════════

@implementation BonjourPublisher {
    NSNetService *_service;
}

- (instancetype)initWithPort:(uint16_t)port {
    self = [super init];
    if (self) {
        _port = port;
        _isPublishing = NO;
    }
    return self;
}

- (void)start {
    if (_isPublishing) return;

    NSString *name = [[NSHost currentHost] localizedName] ?: @"PrimePath-Conductor";
    _service = [[NSNetService alloc] initWithDomain:kServiceDomain
                                               type:kServiceType
                                               name:name
                                               port:_port];
    _service.delegate = self;
    [_service publish];
}

- (void)stop {
    if (!_isPublishing) return;
    [_service stop];
    _service = nil;
    _isPublishing = NO;
}

#pragma mark - NSNetServiceDelegate

- (void)netServiceDidPublish:(NSNetService *)sender {
    _isPublishing = YES;
    NSLog(@"[Bonjour] Published service '%@' on port %d", sender.name, (int)_port);
}

- (void)netService:(NSNetService *)sender didNotPublish:(NSDictionary<NSString *,NSNumber *> *)errorDict {
    _isPublishing = NO;
    NSLog(@"[Bonjour] Failed to publish: %@", errorDict);
}

- (void)netServiceDidStop:(NSNetService *)sender {
    _isPublishing = NO;
    NSLog(@"[Bonjour] Service stopped.");
}

@end

// ═══════════════════════════════════════════════════════════════════════
// BonjourBrowser
// ═══════════════════════════════════════════════════════════════════════

@implementation BonjourBrowser {
    NSNetServiceBrowser *_browser;
    NSMutableArray<NSNetService *> *_resolvingServices;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _isBrowsing = NO;
        _resolvingServices = [NSMutableArray new];
    }
    return self;
}

- (void)startBrowsing {
    if (_isBrowsing) return;

    _browser = [[NSNetServiceBrowser alloc] init];
    _browser.delegate = self;
    [_browser searchForServicesOfType:kServiceType inDomain:kServiceDomain];
    _isBrowsing = YES;
    NSLog(@"[Bonjour] Browsing for %@ services...", kServiceType);
}

- (void)stopBrowsing {
    if (!_isBrowsing) return;
    [_browser stop];
    _browser = nil;
    _isBrowsing = NO;

    for (NSNetService *svc in _resolvingServices) {
        [svc stop];
    }
    [_resolvingServices removeAllObjects];
}

#pragma mark - NSNetServiceBrowserDelegate

- (void)netServiceBrowser:(NSNetServiceBrowser *)browser
           didFindService:(NSNetService *)service
               moreComing:(BOOL)moreComing {
    NSLog(@"[Bonjour] Found service: %@ — resolving...", service.name);
    service.delegate = self;
    [_resolvingServices addObject:service];
    [service resolveWithTimeout:5.0];
}

- (void)netServiceBrowser:(NSNetServiceBrowser *)browser
         didRemoveService:(NSNetService *)service
               moreComing:(BOOL)moreComing {
    NSLog(@"[Bonjour] Lost service: %@", service.name);
    [_resolvingServices removeObject:service];

    if (self.onServiceLost) {
        NSString *host = service.hostName ?: service.name;
        self.onServiceLost(host);
    }
}

- (void)netServiceBrowser:(NSNetServiceBrowser *)browser
             didNotSearch:(NSDictionary<NSString *,NSNumber *> *)errorDict {
    NSLog(@"[Bonjour] Browse error: %@", errorDict);
}

#pragma mark - NSNetServiceDelegate (resolution)

- (void)netServiceDidResolveAddress:(NSNetService *)sender {
    NSString *hostname = sender.hostName;
    uint16_t port = (uint16_t)sender.port;

    // If hostName is nil, try extracting from addresses
    if (!hostname) {
        for (NSData *addrData in sender.addresses) {
            const struct sockaddr *addr = (const struct sockaddr *)[addrData bytes];
            if (addr->sa_family == AF_INET) {
                const struct sockaddr_in *addr4 = (const struct sockaddr_in *)addr;
                char buf[INET_ADDRSTRLEN];
                inet_ntop(AF_INET, &addr4->sin_addr, buf, sizeof(buf));
                hostname = [NSString stringWithUTF8String:buf];
                break;
            } else if (addr->sa_family == AF_INET6) {
                const struct sockaddr_in6 *addr6 = (const struct sockaddr_in6 *)addr;
                char buf[INET6_ADDRSTRLEN];
                inet_ntop(AF_INET6, &addr6->sin6_addr, buf, sizeof(buf));
                hostname = [NSString stringWithUTF8String:buf];
                // Prefer IPv4, keep looking
            }
        }
    }

    NSLog(@"[Bonjour] Resolved: %@ → %@:%d", sender.name, hostname, port);

    if (hostname && self.onServiceFound) {
        self.onServiceFound(hostname, port);
    }

    [_resolvingServices removeObject:sender];
}

- (void)netService:(NSNetService *)sender didNotResolve:(NSDictionary<NSString *,NSNumber *> *)errorDict {
    NSLog(@"[Bonjour] Failed to resolve %@: %@", sender.name, errorDict);
    [_resolvingServices removeObject:sender];
}

@end
