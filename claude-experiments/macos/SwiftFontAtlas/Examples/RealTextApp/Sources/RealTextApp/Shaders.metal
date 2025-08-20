#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float2 position;
    float2 texCoord;
    float4 color;
};

struct Uniforms {
    float4x4 projectionMatrix;
    float2 screenSize;
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
    float4 color;
};

vertex VertexOut vertex_main(uint vertexID [[vertex_id]],
                             constant Vertex* vertices [[buffer(0)]],
                             constant Uniforms& uniforms [[buffer(1)]]) {
    VertexOut out;
    
    Vertex in = vertices[vertexID];
    
    out.position = uniforms.projectionMatrix * float4(in.position, 0.0, 1.0);
    out.texCoord = in.texCoord;
    out.color = in.color;
    
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]],
                              texture2d<float> atlasTexture [[texture(0)]],
                              constant Uniforms& uniforms [[buffer(1)]]) {
    constexpr sampler textureSampler(coord::normalized,
                                     address::clamp_to_edge,
                                     filter::linear);
    
    // Sample the atlas texture (grayscale)
    float alpha = atlasTexture.sample(textureSampler, in.texCoord).r;
    
    // Use the sampled alpha with the vertex color
    float4 color = in.color;
    color.a *= alpha;
    
    return color;
}