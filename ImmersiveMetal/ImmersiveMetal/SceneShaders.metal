
#include <metal_stdlib>
using namespace metal;

constant bool useLayeredRendering [[function_constant(0)]];

struct VertexIn {
    float4 position  [[attribute(0)]];
    float2 texCoords [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoords;
};

struct LayeredVertexOut {
    float4 position [[position]];
    float2 texCoords;
    uint renderTargetIndex [[render_target_array_index]];
    uint viewportIndex [[viewport_array_index]];
};

struct FragmentIn {
    float4 position [[position]];
    float2 texCoords;
    uint renderTargetIndex [[render_target_array_index]];
    uint viewportIndex [[viewport_array_index]];
};

struct PoseConstants {
    float4x4 projectionMatrix;
    float4x4 viewMatrix;
};

struct InstanceConstants {
    float4x4 modelMatrix;
};

// Enhanced color restoration with edge-aware processing for better stereo quality
static half3 restoreVideoColor(half3 rgb, float colorBoost) {
    float3 c = float3(rgb);
    float boost = clamp(colorBoost, 0.80f, 1.40f);
    float t = (boost - 1.0f) / 0.40f;

    // Stage 1: Recover dynamic range from AVPlayer->BGRA conversion (often compressed by codec)
    // Use gamma correction that preserves highlights and lifts shadows
    float gamma = 1.08f + 0.08f * t;  // Adjust gamma based on boost
    c = pow(clamp(c, 0.0f, 1.0f), float3(gamma));
    
    // Stage 2: Enhance saturation while preserving luma for natural appearance
    float luma = dot(c, float3(0.2126f, 0.7152f, 0.0722f));
    float saturation = 1.08f + 0.14f * t;  // Enhanced saturation lift
    c = mix(float3(luma), c, saturation);
    
    // Stage 3: Contrast enhancement (slightly steeper curve for punchy colors)
    c = (c - 0.5f) * (1.08f + 0.12f * t) + 0.5f;
    
    // Stage 4: Final brightness adjustment with per-channel smoothing
    c = min(c * (1.00f + 0.04f * t), 1.0f);
    
    // Stage 5: VisionOS display optimization
    // Apple Vision Pro displays benefit from slightly boosted midtones
    c = c + (0.02f * t) * c * (1.0f - c);
    
    return half3(clamp(c, 0.0f, 1.0f));
}

// Cross-bilateral-style depth prefilter to suppress single-pixel instability on motion.
static float stableDepthSample(texture2d<float, access::sample> depthTex,
                               sampler s,
                               float2 uv)
{
    float2 texel = 1.0f / float2(depthTex.get_width(), depthTex.get_height());
    float center = clamp(float(depthTex.sample(s, uv).r), 0.0f, 1.0f);
    float left = clamp(float(depthTex.sample(s, uv + float2(-texel.x, 0.0f)).r), 0.0f, 1.0f);
    float right = clamp(float(depthTex.sample(s, uv + float2(texel.x, 0.0f)).r), 0.0f, 1.0f);
    float up = clamp(float(depthTex.sample(s, uv + float2(0.0f, -texel.y)).r), 0.0f, 1.0f);
    float down = clamp(float(depthTex.sample(s, uv + float2(0.0f, texel.y)).r), 0.0f, 1.0f);

    // Reduce contribution of neighbors that diverge too far from center (edge preservation).
    float wl = 1.0f - smoothstep(0.02f, 0.12f, abs(left - center));
    float wr = 1.0f - smoothstep(0.02f, 0.12f, abs(right - center));
    float wu = 1.0f - smoothstep(0.02f, 0.12f, abs(up - center));
    float wd = 1.0f - smoothstep(0.02f, 0.12f, abs(down - center));

    float sum = center * 0.46f + left * (0.135f * wl) + right * (0.135f * wr) + up * (0.135f * wu) + down * (0.135f * wd);
    float norm = 0.46f + 0.135f * (wl + wr + wu + wd);
    return sum / max(norm, 1e-4f);
}

static float localDepthInstability(texture2d<float, access::sample> depthTex,
                                   sampler s,
                                   float2 uv)
{
    float2 texel = 1.0f / float2(depthTex.get_width(), depthTex.get_height());
    float l = stableDepthSample(depthTex, s, uv + float2(-texel.x, 0.0f));
    float r = stableDepthSample(depthTex, s, uv + float2(texel.x, 0.0f));
    float u = stableDepthSample(depthTex, s, uv + float2(0.0f, -texel.y));
    float d = stableDepthSample(depthTex, s, uv + float2(0.0f, texel.y));

    // Large local gradient variance is where shimmer/tearing shows up most during pans.
    float gx = abs(r - l);
    float gy = abs(d - u);
    return clamp(gx + gy, 0.0f, 1.0f);
}

static half4 stableShiftedColorSample(texture2d<half, access::sample> colorTex,
                                      sampler s,
                                      float2 uv,
                                      float instability)
{
    float2 texel = 1.0f / float2(colorTex.get_width(), colorTex.get_height());
    float radius = mix(0.0f, 1.2f, smoothstep(0.08f, 0.35f, instability));

    float2 leftUV = clamp(uv + float2(-texel.x * radius, 0.0f), 0.0f, 1.0f);
    float2 rightUV = clamp(uv + float2(texel.x * radius, 0.0f), 0.0f, 1.0f);

    half4 c = colorTex.sample(s, uv);
    if (radius <= 0.001f) {
        return c;
    }

    // Horizontal prefilter suppresses sub-pixel shimmer from pan-time disparity shifts.
    half4 l = colorTex.sample(s, leftUV);
    half4 r = colorTex.sample(s, rightUV);
    return c * 0.60h + l * 0.20h + r * 0.20h;
}

[[vertex]]
LayeredVertexOut vertex_panel_main(VertexIn in [[stage_in]],
                             constant PoseConstants *poses [[buffer(1)]],
                             constant InstanceConstants &instance [[buffer(2)]],
                             uint amplificationID [[amplification_id]])
{
    constant auto &pose = poses[amplificationID];
    
    LayeredVertexOut out;
    out.position = pose.projectionMatrix * pose.viewMatrix * instance.modelMatrix * in.position;
    out.texCoords = in.texCoords;
    if (useLayeredRendering) {
        out.renderTargetIndex = amplificationID;
    }
    out.viewportIndex = amplificationID;
    return out;
}

[[vertex]]
VertexOut vertex_dedicated_panel_main(VertexIn in [[stage_in]],
                                constant PoseConstants *poses [[buffer(1)]],
                                constant InstanceConstants &instance [[buffer(2)]])
{
    constant auto &pose = poses[0];
    
    VertexOut out;
    out.position = pose.projectionMatrix * pose.viewMatrix * instance.modelMatrix * in.position;
    out.texCoords = in.texCoords;
    return out;
}

[[fragment]]
half4 fragment_stereo_conversion(FragmentIn in [[stage_in]],
                                 texture2d<half, access::sample> colorTex [[texture(0)]],
                                 texture2d<float, access::sample> depthTex [[texture(1)]],
                                 constant float &separation [[buffer(0)]],
                                 constant uint &eyeIndexOverride [[buffer(1)]],
                                 constant float &colorBoost [[buffer(2)]],
                                 constant float &stabilityAmount [[buffer(3)]])
{
    constexpr sampler s(coord::normalized, filter::linear, address::clamp_to_edge);
    // In layered/amplified rendering, viewport index maps directly to eye index.
    uint eyeIndex = (eyeIndexOverride > 1 ? in.viewportIndex : eyeIndexOverride);

    float2 uv = in.texCoords;
    float depth = stableDepthSample(depthTex, s, uv);

    // Improved depth mapping curve: better perceptual mapping for Vision Pro
    // Normalize the full [0,1] range to usable disparity with better precision
    depth = smoothstep(0.06f, 0.94f, depth);  // Slightly wider usable range
    // Apply smoother tone curve that emphasizes mid-depth details
    depth = mix(pow(depth, 0.85f), pow(depth, 0.75f), 0.4f);

    // Improved stereo disparity mapping with convergence point adjustment
    float centeredDepth = depth - 0.5f;
    float deadZone = 0.028f;  // Smaller dead-zone to increase visible depth separation
    if (abs(centeredDepth) < deadZone) {
        // Smooth transition near zero parallax for viewing comfort
        centeredDepth = centeredDepth * (1.0f - (deadZone - abs(centeredDepth)) / deadZone * 0.6f);
    } else {
        // Non-linear disparity mapping for better depth perception
        float depthSign = sign(centeredDepth);
        float absDepth = abs(centeredDepth);
        centeredDepth = depthSign * (pow(absDepth * 1.8f, 1.15f) / 1.8f) * 0.52f;
    }
    
    // Push foreground out more aggressively than background to enhance pop-out.
    centeredDepth *= (centeredDepth > 0.0f) ? 1.45f : 0.92f;
    float eyeSign = (eyeIndex == 0) ? -1.0f : 1.0f;
    // Enhanced disparity calculation with better hardware compatibility
    float shift = centeredDepth * separation * 1.06f;

    // Suppress disparity specifically in unstable depth neighborhoods to reduce pan flicker.
    float instability = localDepthInstability(depthTex, s, uv);
    float stability = 1.0f - smoothstep(0.06f, 0.22f, instability);
    shift *= mix(0.68f, 1.0f, stability);

    shift = clamp(shift, -0.0052f, 0.0052f);
    uv.x -= eyeSign * shift;

    float2 baseUV = in.texCoords;
    uv = clamp(uv, 0.0f, 1.0f);
    baseUV = clamp(baseUV, 0.0f, 1.0f);
    half4 shiftedSrc = stableShiftedColorSample(colorTex, s, uv, instability);
    half4 baseSrc = colorTex.sample(s, baseUV);
    
    // Edge-aware color restoration to prevent desaturation at boundaries
    float edgeFade = min(min(uv.x, 1.0f - uv.x), min(uv.y, 1.0f - uv.y)) * 15.0f;
    edgeFade = clamp(edgeFade, 0.0f, 1.0f);
    
    half3 shiftedConverted = restoreVideoColor(shiftedSrc.rgb, colorBoost);
    half3 baseConverted = restoreVideoColor(baseSrc.rgb, colorBoost);

    // During high motion/edge instability, mix back toward unshifted color to suppress shimmer.
    float shiftAmount = abs(shift);
    float blendByShift = smoothstep(0.0022f, 0.0052f, shiftAmount);
    float blendByInstability = smoothstep(0.06f, 0.26f, instability);
    float antiFlickerBlend = clamp(0.18f * blendByShift + 0.82f * blendByInstability, 0.0f, 0.90f);
    antiFlickerBlend *= mix(0.0f, 1.0f, clamp(stabilityAmount, 0.0f, 1.0f));
    half3 converted = mix(shiftedConverted, baseConverted, half3(antiFlickerBlend));

    // Fade color boost near edges to avoid visible stereo artifacts
    converted = mix(converted * 0.95f, converted, half3(edgeFade));
    
    return half4(converted, shiftedSrc.a);
}

[[fragment]]
half4 fragment_main(FragmentIn in [[stage_in]],
                    texture2d<half, access::sample> colorTex [[texture(0)]],
                    texture2d<float, access::sample> depthTex [[texture(1)]],
                    constant float &depthMultiplier [[buffer(0)]],
                    constant uint &eyeIndexOverride [[buffer(1)]],
                    constant float &colorBoost [[buffer(2)]],
                    constant float &stabilityAmount [[buffer(3)]])
{
    constexpr sampler s(coord::normalized, filter::linear, address::clamp_to_edge);
    uint eyeIndex = (eyeIndexOverride > 1 ? in.viewportIndex : eyeIndexOverride);

    float2 uv = in.texCoords;
    float depth = stableDepthSample(depthTex, s, uv);
    
    // Improved depth mapping: better perceptual range and smoother transitions
    depth = smoothstep(0.06f, 0.94f, depth);
    depth = mix(pow(depth, 0.85f), pow(depth, 0.75f), 0.4f);

    // Enhanced stereo disparity mapping
    float centeredDepth = depth - 0.5f;
    float deadZone = 0.028f;
    if (abs(centeredDepth) < deadZone) {
        centeredDepth = centeredDepth * (1.0f - (deadZone - abs(centeredDepth)) / deadZone * 0.6f);
    } else {
        float depthSign = sign(centeredDepth);
        float absDepth = abs(centeredDepth);
        centeredDepth = depthSign * (pow(absDepth * 1.8f, 1.15f) / 1.8f) * 0.52f;
    }
    centeredDepth *= (centeredDepth > 0.0f) ? 1.45f : 0.92f;
    float eyeSign = (eyeIndex == 0) ? -1.0f : 1.0f;
    float shift = centeredDepth * depthMultiplier * 1.06f;

    float instability = localDepthInstability(depthTex, s, uv);
    float stability = 1.0f - smoothstep(0.06f, 0.22f, instability);
    shift *= mix(0.68f, 1.0f, stability);

    shift = clamp(shift, -0.0052f, 0.0052f);
    uv.x -= eyeSign * shift;

    float2 baseUV = in.texCoords;
    uv = clamp(uv, 0.0f, 1.0f);
    baseUV = clamp(baseUV, 0.0f, 1.0f);
    half4 shiftedSrc = stableShiftedColorSample(colorTex, s, uv, instability);
    half4 baseSrc = colorTex.sample(s, baseUV);
    float edgeFade = min(min(uv.x, 1.0f - uv.x), min(uv.y, 1.0f - uv.y)) * 15.0f;
    edgeFade = clamp(edgeFade, 0.0f, 1.0f);

    half3 shiftedConverted = restoreVideoColor(shiftedSrc.rgb, colorBoost);
    half3 baseConverted = restoreVideoColor(baseSrc.rgb, colorBoost);

    float shiftAmount = abs(shift);
    float blendByShift = smoothstep(0.0022f, 0.0052f, shiftAmount);
    float blendByInstability = smoothstep(0.06f, 0.26f, instability);
    float antiFlickerBlend = clamp(0.18f * blendByShift + 0.82f * blendByInstability, 0.0f, 0.90f);
    antiFlickerBlend *= mix(0.0f, 1.0f, clamp(stabilityAmount, 0.0f, 1.0f));
    half3 converted = mix(shiftedConverted, baseConverted, half3(antiFlickerBlend));

    converted = mix(converted * 0.95f, converted, half3(edgeFade));
    return half4(converted, shiftedSrc.a);
}

[[kernel]]
void estimate_depth_from_luma(texture2d<half, access::sample> leftSource [[texture(0)]],
                              texture2d<float, access::write> depthMap [[texture(1)]],
                              uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= depthMap.get_width() || gid.y >= depthMap.get_height()) {
        return;
    }

    float2 uv = (float2(gid) + 0.5f) / float2(depthMap.get_width(), depthMap.get_height());
    constexpr sampler s(coord::normalized, filter::linear, address::clamp_to_edge);
    half3 rgb = leftSource.sample(s, uv).rgb;

    // Lightweight fallback when ML depth is unavailable; brighter areas are treated as closer.
    float luma = dot(float3(rgb), float3(0.299f, 0.587f, 0.114f));
    depthMap.write(float4(luma, 0.0f, 0.0f, 1.0f), gid);
}
