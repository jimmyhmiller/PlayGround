import Foundation
import CoreMediaIO

// Entry point for the camera extension process. macOS launches this executable
// on demand; it registers the CMIO provider and then runs the run loop forever.
let providerSource = ExtensionProviderSource(clientQueue: nil)
CMIOExtensionProvider.startService(provider: providerSource.provider)
CFRunLoopRun()
