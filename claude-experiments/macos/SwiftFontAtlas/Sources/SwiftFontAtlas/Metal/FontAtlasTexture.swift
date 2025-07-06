import Foundation
import Metal
import MetalKit

/// Protocol for providing Metal texture from font atlas
public protocol FontAtlasTextureProvider {
    /// Create a new Metal texture for the atlas
    func createTexture(device: MTLDevice) -> MTLTexture?
    
    /// Update an existing texture with current atlas data
    func updateTexture(_ texture: MTLTexture)
    
    /// Check if texture needs updating
    var needsTextureUpdate: Bool { get }
}

/// Metal texture extensions for FontAtlas
extension FontAtlas {
    /// Create a Metal texture for this atlas
    /// - Parameter device: The Metal device to create texture on
    /// - Returns: A new Metal texture, or nil if creation fails
    public func createTexture(device: MTLDevice) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type2D
        descriptor.width = Int(size)
        descriptor.height = Int(size)
        descriptor.depth = 1
        descriptor.mipmapLevelCount = 1
        descriptor.sampleCount = 1
        descriptor.arrayLength = 1
        
        // Set pixel format based on atlas format
        switch format {
        case .grayscale:
            descriptor.pixelFormat = .r8Unorm
        case .bgr:
            // Metal doesn't have BGR format, would need conversion
            descriptor.pixelFormat = .bgra8Unorm
        case .bgra:
            descriptor.pixelFormat = .bgra8Unorm_srgb
        }
        
        // Configure for optimal CPU->GPU transfer
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .managed // Use .shared on Apple Silicon
        descriptor.cpuCacheMode = .writeCombined
        
        return device.makeTexture(descriptor: descriptor)
    }
    
    /// Update a Metal texture with current atlas data
    /// - Parameter texture: The texture to update
    public func updateTexture(_ texture: MTLTexture) {
        assert(texture.width == size)
        assert(texture.height == size)
        
        let bytesPerRow = Int(size) * format.bytesPerPixel
        
        data.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegion(
                    origin: MTLOrigin(x: 0, y: 0, z: 0),
                    size: MTLSize(width: Int(size), height: Int(size), depth: 1)
                ),
                mipmapLevel: 0,
                withBytes: bytes.baseAddress!,
                bytesPerRow: bytesPerRow
            )
        }
    }
}

/// Managed Metal texture that tracks atlas updates
public class FontAtlasTexture {
    private var texture: MTLTexture?
    private let device: MTLDevice
    private let atlas: FontAtlas
    private var lastModificationCount: UInt64 = 0
    private var lastResizeCount: UInt64 = 0
    
    /// Initialize with a font atlas
    /// - Parameters:
    ///   - atlas: The font atlas to track
    ///   - device: The Metal device to use
    public init(atlas: FontAtlas, device: MTLDevice) {
        self.atlas = atlas
        self.device = device
    }
    
    /// Get the current texture, creating or updating as needed
    public var metalTexture: MTLTexture? {
        let currentModCount = atlas.modificationCount.withLock { $0 }
        let currentResizeCount = atlas.resizeCount.withLock { $0 }
        
        // Check if we need to recreate texture (resize)
        if texture == nil || currentResizeCount > lastResizeCount {
            texture = atlas.createTexture(device: device)
            lastResizeCount = currentResizeCount
            lastModificationCount = 0 // Force update after recreate
        }
        
        // Check if we need to update texture data
        if let texture = texture, currentModCount > lastModificationCount {
            atlas.updateTexture(texture)
            lastModificationCount = currentModCount
        }
        
        return texture
    }
    
    /// Force texture recreation on next access
    public func invalidate() {
        texture = nil
        lastModificationCount = 0
        lastResizeCount = 0
    }
}

/// Extension for FontAtlasManager to support Metal
extension FontAtlasManager {
    /// Create a managed Metal texture for this atlas
    /// - Parameter device: The Metal device to use
    /// - Returns: A managed texture that auto-updates
    public func createManagedTexture(device: MTLDevice) -> FontAtlasTexture {
        var atlasTexture: FontAtlasTexture!
        withAtlas { atlas in
            atlasTexture = FontAtlasTexture(atlas: atlas, device: device)
        }
        return atlasTexture
    }
}