/*
 * rd_gpu_count.hlsl  The translucent layers a frame needs: the most fragments any pixel has (MOST), and from it each
 * layer's draws for rd_gpu.c to draw indirectly (DRAWS), none past the last layer with a fragment.
 * port/tools/gpu_shaders.sh builds it into rd_gpu_shaders.h.
 */

#ifdef __spirv__
#define BIND(n, s) [[vk::binding(n, s)]]
#define SAMPLED(n, s) [[vk::combinedImageSampler]] [[vk::binding(n, s)]]
#else
#define BIND(n, s)
#define SAMPLED(n, s)
#endif

BIND(0, 2) cbuffer Count : register(b0, space2) {
    uint width, height;     /* the counts' */
    uint layers;            /* drawn at most */
    uint entries;           /* a layer's draws: its keys', the last layer's pixels', the screen's for the keys and for
                               the last layer's pixels, then the batches' */
};

#ifdef MOST
SAMPLED(0, 0) Texture2D<float> counts : register(t0, space0);
SAMPLED(0, 0) SamplerState s0 : register(s0, space0);
BIND(0, 1) RWStructuredBuffer<uint> most : register(u0, space1);

groupshared uint top;

[numthreads(16, 16, 1)]
void cs_most(uint3 at : SV_DispatchThreadID, uint i : SV_GroupIndex) {
    if (i == 0) {
        top = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (at.x < width && at.y < height) {
        InterlockedMax(top, (uint)counts.Load(int3(at.xy, 0)));
    }
    GroupMemoryBarrierWithGroupSync();
    if (i == 0 && top != 0) {
        InterlockedMax(most[0], top);
    }
}
#endif

#ifdef DRAWS
BIND(0, 0) StructuredBuffer<uint2> draws : register(t0, space0);    /* first vertex and count: the keys', the batches' */
BIND(0, 1) RWStructuredBuffer<uint4> args : register(u0, space1);
BIND(1, 1) RWStructuredBuffer<uint> most : register(u1, space1);

groupshared uint n;

[numthreads(64, 1, 1)]
void cs_draws(uint i : SV_GroupIndex) {
    if (i == 0) {
        n = min(most[0], layers);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint k = i; k < layers * entries; k += 64) {
        uint l = k / entries, e = k % entries;
        uint2 d = e >= 4 ? draws[e - 3] : e >= 2 ? uint2(0, 6) : draws[0];
        bool on = e == 1 || e == 3 ? l >= 1 && l <= n : l < n;
        args[k] = uint4(on ? d.y : 0, 1, d.x, 0);
    }
}
#endif
