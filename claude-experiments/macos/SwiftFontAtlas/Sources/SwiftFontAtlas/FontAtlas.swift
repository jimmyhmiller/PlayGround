import Foundation
import CoreGraphics
import os.lock

/// Node in the rectangle bin packing algorithm
private struct Node {
    var x: UInt32
    var y: UInt32
    var width: UInt32
}

/// Implements a texture atlas using rectangle bin packing
/// Based on Jukka Jylänki's algorithm as implemented in Ghostty
public class FontAtlas {
    /// Raw texture data
    public private(set) var data: Data
    
    /// Width and height of the atlas (always square)
    public private(set) var size: UInt32
    
    /// Pixel format of the atlas
    public let format: PixelFormat
    
    /// Available space nodes for bin packing
    private var nodes: [Node]
    
    /// Thread-safe modification tracking
    public private(set) var modificationCount = OSAllocatedUnfairLock<UInt64>(initialState: 0)
    public private(set) var resizeCount = OSAllocatedUnfairLock<UInt64>(initialState: 0)
    
    /// Initialize a new font atlas
    /// - Parameters:
    ///   - size: Size of the atlas (must be power of 2)
    ///   - format: Pixel format for the atlas
    public init(size: UInt32, format: PixelFormat) throws {
        // Validate size is power of 2
        guard size > 0 && (size & (size - 1)) == 0 else {
            throw FontAtlasError.invalidAtlasSize
        }
        
        self.size = size
        self.format = format
        
        // Allocate data
        let dataSize = Int(size * size) * format.bytesPerPixel
        self.data = Data(count: dataSize)
        
        // Initialize with empty nodes array
        self.nodes = []
        self.nodes.reserveCapacity(64) // Pre-allocate for performance
        
        // Setup initial state
        clear()
    }
    
    /// Clear the atlas and reset to initial state
    public func clear() {
        // Clear data to zero
        data.resetBytes(in: 0..<data.count)
        
        // Reset nodes
        nodes.removeAll(keepingCapacity: true)
        
        // Add initial rectangle with 1px border
        // This prevents texture bleeding at edges
        nodes.append(Node(x: 1, y: 1, width: size - 2))
        
        // Increment modification counter
        modificationCount.withLock { $0 += 1 }
    }
    
    /// Reserve a region in the atlas
    /// - Parameters:
    ///   - width: Width of the region
    ///   - height: Height of the region
    /// - Returns: The reserved region
    /// - Throws: `FontAtlasError.atlasFull` if region doesn't fit
    public func reserve(width: UInt32, height: UInt32) throws -> AtlasRegion {
        // Handle empty regions
        if width == 0 && height == 0 {
            return AtlasRegion(x: 0, y: 0, width: 0, height: 0)
        }
        
        // Find best fitting node
        var bestHeight: UInt32 = .max
        var bestWidth: UInt32 = .max
        var bestIndex: Int?
        var region = AtlasRegion(x: 0, y: 0, width: width, height: height)
        
        for (index, _) in nodes.enumerated() {
            if let y = fit(index: index, width: width, height: height) {
                let node = nodes[index]
                if (y + height < bestHeight) ||
                   (y + height == bestHeight && node.width > 0 && node.width < bestWidth) {
                    bestIndex = index
                    bestWidth = node.width
                    bestHeight = y + height
                    region = AtlasRegion(x: node.x, y: y, width: width, height: height)
                }
            }
        }
        
        guard let bestIdx = bestIndex else {
            throw FontAtlasError.atlasFull
        }
        
        // Insert new node for this rectangle
        nodes.insert(
            Node(x: region.x, y: region.y + height, width: width),
            at: bestIdx
        )
        
        // Optimize rectangles
        var i = bestIdx + 1
        while i < nodes.count {
            let prev = nodes[i - 1]
            let node = nodes[i]
            
            if node.x < (prev.x + prev.width) {
                let shrink = prev.x + prev.width - node.x
                nodes[i].x += shrink
                
                if shrink >= node.width {
                    nodes[i].width = 0
                } else {
                    nodes[i].width -= shrink
                }
                
                if nodes[i].width <= 0 {
                    nodes.remove(at: i)
                    i -= 1
                }
            } else {
                break
            }
            
            i += 1
        }
        
        merge()
        
        return region
    }
    
    /// Set data for a reserved region
    /// - Parameters:
    ///   - region: The region to write to
    ///   - data: The pixel data to write
    public func set(region: AtlasRegion, data: Data) {
        assert(region.x < (size - 1))
        assert((region.x + region.width) <= (size - 1))
        assert(region.y < (size - 1))
        assert((region.y + region.height) <= (size - 1))
        
        let depth = format.bytesPerPixel
        
        // Copy row by row
        for row in 0..<region.height {
            let destOffset = Int((region.y + row) * size + region.x) * depth
            let srcOffset = Int(row * region.width) * depth
            let length = Int(region.width) * depth
            
            self.data.replaceSubrange(
                destOffset..<(destOffset + length),
                with: data[srcOffset..<(srcOffset + length)]
            )
        }
        
        // Increment modification counter
        modificationCount.withLock { $0 += 1 }
    }
    
    /// Grow the atlas to a new size
    /// - Parameter newSize: New size (must be larger than current)
    public func grow(to newSize: UInt32) throws {
        assert(newSize >= size)
        guard newSize > size else { return }
        
        // Validate new size is power of 2
        guard newSize > 0 && (newSize & (newSize - 1)) == 0 else {
            throw FontAtlasError.invalidAtlasSize
        }
        
        // Save old data
        let oldData = data
        let oldSize = size
        
        // Allocate new data
        let dataSize = Int(newSize * newSize) * format.bytesPerPixel
        data = Data(count: dataSize)
        
        // Update size
        size = newSize
        
        // Copy old data (skipping first row for border)
        let depth = format.bytesPerPixel
        for row in 1..<(oldSize - 1) {
            let srcOffset = Int(row * oldSize) * depth
            let destOffset = Int(row * newSize) * depth
            let length = Int(oldSize) * depth
            
            data.replaceSubrange(
                destOffset..<(destOffset + length),
                with: oldData[srcOffset..<(srcOffset + length)]
            )
        }
        
        // Add new rectangle for added space
        nodes.append(Node(
            x: oldSize - 1,
            y: 1,
            width: newSize - oldSize
        ))
        
        // Update counters
        modificationCount.withLock { $0 += 1 }
        resizeCount.withLock { $0 += 1 }
    }
    
    // MARK: - Private Methods
    
    /// Check if a rectangle fits at the given node index
    private func fit(index: Int, width: UInt32, height: UInt32) -> UInt32? {
        let node = nodes[index]
        
        // Check if it exceeds texture bounds
        if (node.x + width) > (size - 1) {
            return nil
        }
        
        // Find Y position that fits
        var y = node.y
        var widthLeft = width
        var i = index
        
        while widthLeft > 0 && i < nodes.count {
            let n = nodes[i]
            if n.y > y {
                y = n.y
            }
            
            // Check if height exceeds bounds
            if (y + height) > (size - 1) {
                return nil
            }
            
            if n.width >= widthLeft {
                widthLeft = 0
            } else {
                widthLeft -= n.width
            }
            
            i += 1
        }
        
        return y
    }
    
    /// Merge adjacent nodes with the same Y value
    private func merge() {
        var i = 0
        while i < nodes.count - 1 {
            let node = nodes[i]
            let next = nodes[i + 1]
            
            if node.y == next.y {
                nodes[i].width += next.width
                nodes.remove(at: i + 1)
            } else {
                i += 1
            }
        }
    }
}

// MARK: - Texture Coordinates

extension FontAtlas {
    /// Get normalized texture coordinates for a region
    /// - Parameter region: The atlas region
    /// - Returns: Normalized coordinates (0.0 to 1.0) as (u1, v1, u2, v2)
    public func normalizedCoordinates(for region: AtlasRegion) -> (Float, Float, Float, Float) {
        let size = Float(self.size)
        return (
            Float(region.x) / size,
            Float(region.y) / size,
            Float(region.x + region.width) / size,
            Float(region.y + region.height) / size
        )
    }
    
    /// Get normalized coordinates for a glyph
    public func normalizedCoordinates(for glyph: RenderedGlyph) -> (Float, Float, Float, Float) {
        let size = Float(self.size)
        return (
            Float(glyph.atlasX) / size,
            Float(glyph.atlasY) / size,
            Float(glyph.atlasX + glyph.width) / size,
            Float(glyph.atlasY + glyph.height) / size
        )
    }
}