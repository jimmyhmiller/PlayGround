#include <metal_stdlib>
using namespace metal;

constant int kMaxCutouts = 8;

struct SurfaceUniform {
    float4 rect;
    float4 color;
    float4 params;
    uint cutoutCount;
    uint shadowCount;
    uint isOuterRect;
    uint isPressed;
    float4 cutoutRect0;
    float4 cutoutRect1;
    float4 cutoutRect2;
    float4 cutoutRect3;
    float4 cutoutRect4;
    float4 cutoutRect5;
    float4 cutoutRect6;
    float4 cutoutRect7;
    float4 cutoutRadiiA;
    float4 cutoutRadiiB;
    float4 shadowRect0;
    float4 shadowRect1;
    float4 shadowRect2;
    float4 shadowRect3;
    float4 shadowRect4;
    float4 shadowRect5;
    float4 shadowRect6;
    float4 shadowRect7;
    float4 shadowRadiiA;
    float4 shadowRadiiB;
};

struct ViewUniforms {
    float2 viewport;
};

struct RenderTuning {
    float4 shadingA;
    float4 shadingB;
    float4 shadingC;
    float4 shadingD;
};

struct SurfaceOut {
    float4 position [[position]];
    float2 world;
    uint instanceID [[flat]];
};

struct TextVertex {
    float2 position;
    float2 uv;
};

struct TextOut {
    float4 position [[position]];
    float2 uv;
};

struct ProceduralOut {
    float4 position [[position]];
    float2 world;
};

struct SceneSample {
    float height;
    float3 color;
    float occluderDistance;
    float surfaceEdgeDistance;
};

float roundedRectSDF(float2 point, float4 rect, float radius) {
    float2 center = rect.xy + rect.zw * 0.5;
    float2 halfSize = rect.zw * 0.5 - radius;
    float2 q = abs(point - center) - halfSize;
    return length(max(q, float2(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

float outerRectMask(float2 point, float4 rect) {
    if (point.x < rect.x || point.x > rect.x + rect.z || point.y < rect.y || point.y > rect.y + rect.w) {
        return 0.0;
    }
    return 1.0;
}

float4 insetRect(float4 rect, float inset) {
    return float4(rect.x + inset, rect.y + inset, rect.z - inset * 2.0, rect.w - inset * 2.0);
}

float4 offsetRect(float4 rect, float2 offset) {
    return float4(rect.xy + offset, rect.zw);
}

float cardSilhouetteSDF(float2 p, float4 card) {
    float d = roundedRectSDF(p, card, 34.0);
    float4 leftFoot = float4(card.x - 22.0, card.y + 16.0, 56.0, 58.0);
    float4 rightFoot = float4(card.x + card.z - 34.0, card.y + 16.0, 56.0, 58.0);
    float4 leftShoulder = float4(card.x - 20.0, card.y + card.w - 112.0, 54.0, 72.0);
    float4 rightShoulder = float4(card.x + card.z - 34.0, card.y + card.w - 112.0, 54.0, 72.0);
    d = min(d, roundedRectSDF(p, leftFoot, 26.0));
    d = min(d, roundedRectSDF(p, rightFoot, 26.0));
    d = min(d, roundedRectSDF(p, leftShoulder, 26.0));
    d = min(d, roundedRectSDF(p, rightShoulder, 26.0));
    return d;
}

float fieldWellSDF(float2 p, float4 field) {
    return roundedRectSDF(p, field, 18.0);
}

float minFieldDistance(float2 p, float4 field1, float4 field2, float4 field3) {
    float d = fieldWellSDF(p, field1);
    d = min(d, fieldWellSDF(p, field2));
    d = min(d, fieldWellSDF(p, field3));
    return d;
}

SceneSample paperScene(float2 p, float2 viewport) {
    const float pt = 18.0;
    float4 outer = float4(0.0, 0.0, viewport.x, viewport.y);
    float4 redCut = insetRect(outer, 30.0);
    float4 orangeCut = insetRect(outer, 60.0);
    float4 yellowCut = insetRect(outer, 90.0);
    float4 greenCut = insetRect(outer, 120.0);
    float4 tealCut = insetRect(outer, 150.0);

    SceneSample s;
    s.height = pt;
    s.color = float3(0.41, 0.54, 0.70);
    s.occluderDistance = abs(roundedRectSDF(p, tealCut, 42.0));
    s.surfaceEdgeDistance = 1e6;

    float redD = roundedRectSDF(p, redCut, 42.0);
    float orangeD = roundedRectSDF(p, orangeCut, 42.0);
    float yellowD = roundedRectSDF(p, yellowCut, 42.0);
    float greenD = roundedRectSDF(p, greenCut, 42.0);
    float tealD = roundedRectSDF(p, tealCut, 42.0);

    if (redD > 0.0) {
        s.height = 6.0 * pt;
        s.color = float3(0.86, 0.49, 0.45);
        s.occluderDistance = 1e6;
        s.surfaceEdgeDistance = redD;
    } else if (orangeD > 0.0) {
        s.height = 5.0 * pt;
        s.color = float3(0.90, 0.60, 0.33);
        s.occluderDistance = abs(redD);
        s.surfaceEdgeDistance = orangeD;
    } else if (yellowD > 0.0) {
        s.height = 4.0 * pt;
        s.color = float3(0.85, 0.72, 0.33);
        s.occluderDistance = abs(orangeD);
        s.surfaceEdgeDistance = yellowD;
    } else if (greenD > 0.0) {
        s.height = 3.0 * pt;
        s.color = float3(0.58, 0.70, 0.54);
        s.occluderDistance = abs(yellowD);
        s.surfaceEdgeDistance = greenD;
    } else if (tealD > 0.0) {
        s.height = 2.0 * pt;
        s.color = float3(0.39, 0.62, 0.60);
        s.occluderDistance = abs(greenD);
        s.surfaceEdgeDistance = tealD;
    }

    float blueOpeningW = viewport.x - 300.0;
    float blueOpeningH = viewport.y - 300.0;
    float formW = min(690.0, blueOpeningW - 72.0);
    float formH = min(560.0, blueOpeningH - 44.0);
    float4 card = float4((viewport.x - formW) * 0.5, (viewport.y - formH) * 0.5, formW, formH);

    float insetX = card.x + 64.0;
    float fieldW = card.z - 128.0;
    float fieldH = 60.0;
    float fieldGap = 18.0;
    float buttonY = card.y + 42.0;
    float rememberY = buttonY + 84.0;
    float passwordY = rememberY + 48.0;
    float emailY = passwordY + fieldH + fieldGap;
    float nameY = emailY + fieldH + fieldGap;
    float4 nameField = float4(insetX, nameY, fieldW, fieldH);
    float4 emailField = float4(insetX, emailY, fieldW, fieldH);
    float4 passwordField = float4(insetX, passwordY, fieldW, fieldH);
    float4 button = float4(insetX, buttonY, fieldW, 60.0);
    float4 checkbox = float4(insetX - 3.0, rememberY - 3.0, 28.0, 28.0);
    float4 avatar = float4(card.x + 54.0, card.y + card.w - 132.0, 104.0, 104.0);
    float4 avatarFloor = insetRect(avatar, 8.0);
    float4 avatarHead = float4(avatar.x + avatar.z * 0.5 - 17.0, avatar.y + avatar.w * 0.5 + 6.0, 34.0, 34.0);
    float4 avatarBody = float4(avatar.x + avatar.z * 0.5 - 31.0, avatar.y + 22.0, 62.0, 32.0);

    float cardD = cardSilhouetteSDF(p, card);
    float underD = cardSilhouetteSDF(p, offsetRect(card, float2(0.0, -7.0)));
    if (underD <= 0.0 && 2.0 * pt > s.height) {
        s.height = 2.0 * pt;
        s.color = float3(0.34, 0.19, 0.47);
        s.occluderDistance = abs(cardD);
        s.surfaceEdgeDistance = abs(underD);
    }

    float fieldD = minFieldDistance(p, insetRect(nameField, -1.0), insetRect(emailField, -1.0), insetRect(passwordField, -1.0));
    float buttonD = roundedRectSDF(p, insetRect(button, -1.0), 20.0);
    float checkboxD = roundedRectSDF(p, checkbox, 6.0);
    float avatarD = roundedRectSDF(p, avatar, 52.0);
    float topHoleD = min(min(fieldD, buttonD), min(checkboxD, avatarD));

    if (cardD <= 0.0) {
        s.height = 4.0 * pt;
        s.color = float3(0.64, 0.48, 0.80);
        s.occluderDistance = 1e6;
        s.surfaceEdgeDistance = abs(cardD);
    }

    if (cardD <= 0.0 && topHoleD <= 0.0) {
        float wallD = min(min(fieldD, buttonD), min(checkboxD, avatarD));
        s.height = 4.0 * pt - 1.0;
        s.color = float3(0.48, 0.30, 0.64);
        s.occluderDistance = 1e6;
        s.surfaceEdgeDistance = abs(wallD);
    }

    float fieldFloorD = minFieldDistance(p, insetRect(nameField, 5.0), insetRect(emailField, 5.0), insetRect(passwordField, 5.0));
    if (cardD <= 0.0 && fieldFloorD <= 0.0) {
        s.height = 3.0 * pt;
        s.color = float3(0.31, 0.17, 0.43);
        s.occluderDistance = abs(fieldD);
        s.surfaceEdgeDistance = abs(fieldFloorD);
    }

    float buttonFloorD = roundedRectSDF(p, insetRect(button, 6.0), 15.0);
    if (cardD <= 0.0 && buttonFloorD <= 0.0) {
        s.height = 3.0 * pt;
        s.color = float3(0.31, 0.17, 0.43);
        s.occluderDistance = abs(buttonD);
        s.surfaceEdgeDistance = abs(buttonFloorD);
    }

    float checkboxFloorD = roundedRectSDF(p, insetRect(checkbox, 5.0), 4.0);
    if (cardD <= 0.0 && checkboxFloorD <= 0.0) {
        s.height = 3.0 * pt;
        s.color = float3(0.34, 0.19, 0.47);
        s.occluderDistance = abs(checkboxD);
        s.surfaceEdgeDistance = abs(checkboxFloorD);
    }

    float avatarFloorD = roundedRectSDF(p, avatarFloor, 44.0);
    if (cardD <= 0.0 && avatarFloorD <= 0.0) {
        s.height = 3.0 * pt;
        s.color = float3(0.48, 0.30, 0.64);
        s.occluderDistance = abs(avatarD);
        s.surfaceEdgeDistance = abs(avatarFloorD);
    }
    float avatarGlyphD = min(roundedRectSDF(p, avatarHead, 17.0), roundedRectSDF(p, avatarBody, 16.0));
    if (cardD <= 0.0 && avatarGlyphD <= 0.0) {
        s.height = 2.7 * pt;
        s.color = float3(0.26, 0.13, 0.38);
        s.occluderDistance = abs(avatarGlyphD);
        s.surfaceEdgeDistance = abs(avatarGlyphD);
    }

    return s;
}

float heightAt(float2 p, float2 viewport) {
    return paperScene(p, viewport).height;
}

vertex ProceduralOut proceduralVertex(uint vertexID [[vertex_id]],
                                      constant ViewUniforms &view [[buffer(0)]]) {
    const float2 corners[4] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(0.0, 1.0),
        float2(1.0, 1.0)
    };
    float2 world = corners[vertexID] * view.viewport;
    ProceduralOut out;
    out.position = float4((world.x / view.viewport.x) * 2.0 - 1.0,
                          (world.y / view.viewport.y) * 2.0 - 1.0,
                          0.0,
                          1.0);
    out.world = world;
    return out;
}

fragment float4 proceduralFragment(ProceduralOut in [[stage_in]],
                                   constant ViewUniforms &view [[buffer(0)]],
                                   constant RenderTuning &tuning [[buffer(2)]]) {
    SceneSample s = paperScene(in.world, view.viewport);
    float eps = 1.4;
    float hx = heightAt(in.world + float2(eps, 0.0), view.viewport) - heightAt(in.world - float2(eps, 0.0), view.viewport);
    float hy = heightAt(in.world + float2(0.0, eps), view.viewport) - heightAt(in.world - float2(0.0, eps), view.viewport);
    float3 normal = normalize(float3(-hx * 0.045, -hy * 0.045, 1.0));
    float3 light = normalize(float3(-0.35, 0.45, 0.95));
    float diffuse = 0.78 + max(dot(normal, light), 0.0) * 0.26;

    float zNorm = clamp(s.height / 108.0, 0.0, 1.0);
    float grain = fract(sin(dot(floor(in.world * tuning.shadingA.x), float2(12.9898, 78.233))) * 43758.5453);
    float fiber = 1.0 + (grain - 0.5) * tuning.shadingA.y;
    float depthTone = mix(tuning.shadingB.x, tuning.shadingB.y, zNorm);
    float edgeDark = 1.0 - (1.0 - smoothstep(0.0, 16.0, s.surfaceEdgeDistance)) * tuning.shadingA.z;
    float topHighlight = 1.0 + (1.0 - smoothstep(0.0, 11.0, s.surfaceEdgeDistance)) * tuning.shadingA.w;
    float recessBoost = (1.0 - zNorm) * tuning.shadingD.y;
    float ao = (1.0 - smoothstep(0.0, tuning.shadingC.x, s.occluderDistance)) * (0.18 + recessBoost * 0.18) * tuning.shadingB.w;
    float shadow = (1.0 - smoothstep(0.0, tuning.shadingC.z, s.occluderDistance + in.world.x * tuning.shadingC.w + in.world.y * tuning.shadingD.x)) * tuning.shadingC.y * (0.10 + recessBoost * 0.08);

    float3 color = s.color * diffuse * fiber * depthTone * edgeDark * topHighlight;
    color -= ao + shadow;
    return float4(saturate(color), 1.0);
}

float4 cutoutRectAt(SurfaceUniform surface, uint index) {
    switch (index) {
        case 0: return surface.cutoutRect0;
        case 1: return surface.cutoutRect1;
        case 2: return surface.cutoutRect2;
        case 3: return surface.cutoutRect3;
        case 4: return surface.cutoutRect4;
        case 5: return surface.cutoutRect5;
        case 6: return surface.cutoutRect6;
        default: return surface.cutoutRect7;
    }
}

float cutoutRadiusAt(SurfaceUniform surface, uint index) {
    if (index < 4) {
        return surface.cutoutRadiiA[index];
    }
    return surface.cutoutRadiiB[index - 4];
}

float4 shadowRectAt(SurfaceUniform surface, uint index) {
    switch (index) {
        case 0: return surface.shadowRect0;
        case 1: return surface.shadowRect1;
        case 2: return surface.shadowRect2;
        case 3: return surface.shadowRect3;
        case 4: return surface.shadowRect4;
        case 5: return surface.shadowRect5;
        case 6: return surface.shadowRect6;
        default: return surface.shadowRect7;
    }
}

float shadowRadiusAt(SurfaceUniform surface, uint index) {
    if (index < 4) {
        return surface.shadowRadiiA[index];
    }
    return surface.shadowRadiiB[index - 4];
}

vertex SurfaceOut surfaceVertex(uint vertexID [[vertex_id]],
                                uint instanceID [[instance_id]],
                                constant ViewUniforms &view [[buffer(0)]],
                                constant SurfaceUniform *surfaces [[buffer(1)]]) {
    const float2 corners[4] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(0.0, 1.0),
        float2(1.0, 1.0)
    };

    SurfaceUniform surface = surfaces[instanceID];
    float2 local = corners[vertexID];
    float2 world = surface.rect.xy + local * surface.rect.zw;
    float2 ndc = float2((world.x / view.viewport.x) * 2.0 - 1.0,
                        (world.y / view.viewport.y) * 2.0 - 1.0);

    SurfaceOut out;
    out.position = float4(ndc, 0.0, 1.0);
    out.world = world;
    out.instanceID = instanceID;
    return out;
}

fragment float4 surfaceFragment(SurfaceOut in [[stage_in]],
                                constant SurfaceUniform *surfaces [[buffer(1)]],
                                constant RenderTuning &tuning [[buffer(2)]]) {
    SurfaceUniform surface = surfaces[in.instanceID];
    float mask = surface.isOuterRect == 1 ? outerRectMask(in.world, surface.rect) : (roundedRectSDF(in.world, surface.rect, surface.params.x) <= 0.0 ? 1.0 : 0.0);
    if (mask <= 0.0) {
        discard_fragment();
    }

    float nearestOuterEdge = min(min(in.world.x - surface.rect.x, surface.rect.x + surface.rect.z - in.world.x),
                                 min(in.world.y - surface.rect.y, surface.rect.y + surface.rect.w - in.world.y));
    float nearestCut = 1e6;
    for (uint i = 0; i < surface.cutoutCount; ++i) {
        float d = roundedRectSDF(in.world, cutoutRectAt(surface, i), cutoutRadiusAt(surface, i));
        if (d <= 0.0) {
            discard_fragment();
        }
        nearestCut = min(nearestCut, d);
    }

    float nearestOccluder = 1e6;
    for (uint i = 0; i < surface.shadowCount; ++i) {
        float d = abs(roundedRectSDF(in.world, shadowRectAt(surface, i), shadowRadiusAt(surface, i)));
        nearestOccluder = min(nearestOccluder, d);
    }

    float grainScale = tuning.shadingA.x;
    float grainAmount = tuning.shadingA.y;
    float edgeAmount = tuning.shadingA.z;
    float highlightAmount = tuning.shadingA.w;
    float grain = fract(sin(dot(floor(in.world * grainScale), float2(12.9898, 78.233))) * 43758.5453);
    float fiber = 1.0 + (grain - 0.5) * grainAmount;
    float outerHighlight = smoothstep(0.0, 12.0, nearestOuterEdge);
    float edgeDarkening = 1.0 - smoothstep(0.0, 10.0, nearestOuterEdge) * edgeAmount;
    float zNorm = clamp(surface.params.y / 108.0, 0.0, 1.0);
    float depthDarken = mix(tuning.shadingB.x, tuning.shadingB.y, zNorm);
    float recessBoost = (1.0 - zNorm) * tuning.shadingD.y;
    float upperEdgeMark = nearestCut < 1e5 ? (1.0 - smoothstep(0.0, 10.0, nearestCut)) * tuning.shadingB.z : 0.0;
    float occluderAO = nearestOccluder < 1e5 ? (1.0 - smoothstep(0.0, tuning.shadingC.x, nearestOccluder)) * (0.20 + surface.params.z * 0.08 + recessBoost * 0.20) * tuning.shadingB.w : 0.0;
    float directionalShadow = nearestOccluder < 1e5
        ? (1.0 - smoothstep(0.0, tuning.shadingC.z, nearestOccluder + (in.world.x * tuning.shadingC.w + in.world.y * tuning.shadingD.x))) * (0.08 + surface.params.z * 0.04 + recessBoost * 0.10) * tuning.shadingC.y
        : 0.0;
    float pressedShade = surface.isPressed == 1 ? 0.92 : 1.0;
    float topEdgeBoost = 1.0 + (1.0 - outerHighlight) * highlightAmount;

    float3 color = surface.color.rgb * fiber * edgeDarkening * pressedShade * topEdgeBoost * depthDarken;
    color -= upperEdgeMark;
    color -= occluderAO;
    color -= directionalShadow;
    return float4(saturate(color), 1.0);
}

vertex TextOut textVertex(uint vertexID [[vertex_id]],
                          constant ViewUniforms &view [[buffer(0)]],
                          constant TextVertex *vertices [[buffer(1)]]) {
    TextOut out;
    float2 world = vertices[vertexID].position;
    out.position = float4((world.x / view.viewport.x) * 2.0 - 1.0,
                          (world.y / view.viewport.y) * 2.0 - 1.0,
                          0.0,
                          1.0);
    out.uv = vertices[vertexID].uv;
    return out;
}

fragment float4 textFragment(TextOut in [[stage_in]],
                             texture2d<float> inkTexture [[texture(0)]],
                             sampler textureSampler [[sampler(0)]]) {
    float4 sample = inkTexture.sample(textureSampler, in.uv);
    return sample;
}
