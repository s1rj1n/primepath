//
//  Version.h -- single source of truth for the PrimePath version string.
//
//  Every user-visible version string, User-Agent header, and PrimeNet
//  report reads from here, and this reads CFBundleShortVersionString from
//  Info.plist. To cut a new release, bump Info.plist and nothing else.
//

#pragma once

#import <Foundation/Foundation.h>
#include <string>

// e.g. @"1.4.1". Falls back to @"0.0.0" only when there is no bundle to
// read, such as a command-line test harness linking this code.
static inline NSString *PrimePathVersion(void) {
    static NSString *version;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        version = [[NSBundle mainBundle]
            objectForInfoDictionaryKey:@"CFBundleShortVersionString"];
        if (version.length == 0) version = @"0.0.0";
    });
    return version;
}

// Same value, for the C++ string building in PrimeNetClient.
static inline std::string PrimePathVersionUTF8(void) {
    return std::string(PrimePathVersion().UTF8String);
}
