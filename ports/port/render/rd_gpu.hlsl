/*
 * rd_gpu.hlsl  rd_soft.c's pixel on the GPU: values over w divided here, textures read by hand with
 * Load. port/tools/gpu_shaders.sh builds it into rd_gpu_shaders.h.
 */

struct VIn {
    float2 pos : TEXCOORD0;
    float2 depth_q : TEXCOORD1; /* the depth buffer's value; 1/w */
    float2 uv : TEXCOORD2;      /* u/w, v/w */
    float4 base : TEXCOORD3;    /* a r g b over w, 0 to 255 */
    float4 offs : TEXCOORD4;    /* r g b over w, and the triangle's place in its list */
    float4 ab : TEXCOORD5;      /* the corners a and b, for the exact edges */
    float4 cf : TEXCOORD6;      /* c, and the edges' flags (see inside) */
    float4 uvb : TEXCOORD7;     /* a sprite's u and v, from and to */
};

struct VOut {
    float4 pos : SV_Position;
    noperspective float q : TEXCOORD0;
    noperspective float2 uv : TEXCOORD1;
    noperspective float4 base : TEXCOORD2;
    noperspective float3 offs : TEXCOORD3;
    nointerpolation float order : TEXCOORD4;
    nointerpolation float4 ab : TEXCOORD5;
    nointerpolation float4 cf : TEXCOORD6;
    nointerpolation float4 uvb : TEXCOORD7;
};

VOut vs_main(VIn v) {
    VOut o;
    o.pos = float4(v.pos.x / 320.0 - 1.0, 1.0 - v.pos.y / 240.0, v.depth_q.x, 1.0);
    o.q = v.depth_q.y;
    o.uv = v.uv;
    o.base = v.base;
    o.offs = v.offs.xyz;
    o.order = v.offs.w;
    o.ab = v.ab;
    o.cf = v.cf;
    o.uvb = v.uvb;
    return o;
}

/* the texture, the palette, and the peeling's keys (1/w, place): the last layer's and this one's */
#ifdef __spirv__
/* SDL_GPU's Vulkan wants combined image samplers, in set 2 */
#define BIND(n) [[vk::combinedImageSampler]] [[vk::binding(n, 2)]]
#else
#define BIND(n)
#endif
BIND(0) Texture2D<float4> tex : register(t0, space2);
BIND(1) Texture2D<float4> pal : register(t1, space2);
BIND(2) Texture2D<float> prev_q : register(t2, space2);
BIND(3) Texture2D<float> prev_o : register(t3, space2);
BIND(4) Texture2D<float> min_q : register(t4, space2);
BIND(5) Texture2D<float> min_o : register(t5, space2);
BIND(6) Texture2D<float4> under : register(t6, space2);     /* the frame so far */
BIND(0) SamplerState s0 : register(s0, space2);
BIND(1) SamplerState s1 : register(s1, space2);
BIND(2) SamplerState s2 : register(s2, space2);
BIND(3) SamplerState s3 : register(s3, space2);
BIND(4) SamplerState s4 : register(s4, space2);
BIND(5) SamplerState s5 : register(s5, space2);
BIND(6) SamplerState s6 : register(s6, space2);

cbuffer Draw : register(b0, space3) {
    uint flags;
    int width, height, levels;
    int palette;
    float mip_bias;
    int mode;           /* 0 shade, 1 the nearest 1/w past the last layer, 2 the first place at it,
                           3 shade that fragment and blend it over under, 4 under where the last layer's fragment
                           was, 5 one for each fragment, 6 mip_bias, 7 under */
    int blend;          /* source factor + 16 * destination factor */
};

cbuffer Frame : register(b1, space3) {
    float4 fog_table[32];
    float4 fog_color;   /* a r g b, 0 to 255 */
    float fog_density;
    uint zero;          /* 0, for hold() */
    float inv_scale;    /* the game's pixels a pixel drawn */
    int sprite_mode;    /* drawn larger, a flat triangle's texture: 0 as the game has it, 1 sharp, 2 the nearest texel,
                           3 smooth, 4 texel (sharp where its texels fall) */
};

#define F_TEX 1
#define F_PAL 2
#define F_BILINEAR 4
#define F_TEX_ALPHA 8
#define F_USE_ALPHA 16
#define F_OFFSET 32
#define F_FOG_TABLE 64
#define F_FOG_TABLE2 128
#define F_PUNCH 256
#define F_SHADING 512       /* two bits */
#define F_UV 2048           /* four bits */

/* as C's int division, x >= 0 */
int div255(int x) {
    return (int)((uint)x / 255u);
}

int3 div255(int3 x) {
    return (int3)((uint3)x / 255u);
}

int wrap(int x, int n, bool clamp, bool flip) {
    int m = x & (2 * n - 1);
    return clamp ? (x < 0 ? 0 : x >= n ? n - 1 : x) : flip ? (m < n ? m : 2 * n - 1 - m) : x & (n - 1);
}

int4 texel(int x, int y, int k) {
    float4 t;
    if (flags & F_PAL) {
        int i = (int)(tex.Load(int3(x, y, k)).r * 255.0 + 0.5);
        t = pal.Load(int3((palette + i) & 1023, 0, 0));
    } else {
        t = tex.Load(int3(x, y, k));
    }
    return int4(t.a * 255.0 + 0.5, t.r * 255.0 + 0.5, t.g * 255.0 + 0.5, t.b * 255.0 + 0.5);
}

/* smode 1: bilinear only across a texel's edges, fw the texels a pixel at level 0 (5: and none where a pixel only
   touches the edge); 2: the nearest texel; 3: as the game filters it; 4: bilinear, with the nearest texel's alpha. Each reads only the sprite's own texels, between uvb's
   corners */
int4 sample_tex(int k, float u, float v, int smode, float2 fw, float4 uvb) {
    uint uvm = flags / F_UV & 15;
    bool cv = (uvm & 1) != 0, cu = (uvm & 2) != 0, fv = (uvm & 4) != 0, fu = (uvm & 8) != 0;
    int w = max((uint)width >> k, 1), h = max((uint)height >> k, 1);
    float su = u * w, sv = v * h;
    bool bilinear = (flags & F_BILINEAR) != 0;
    int4 c;
    if (smode == 2) {
        bilinear = false;
    } else if (smode == 4) {
        bilinear = true;
    } else if (smode == 1 || smode == 5) {
        float2 f = fw * exp2(-(float)k);
        if (max(f.x, f.y) < 1.0) {
            /* the blend's width in texels: a pixel's, or for a texture the game magnifies and filters, up to a
               texel (its own bilinear) from 1.25 to 2 game pixels a texel */
            float2 wide = f;
            if (bilinear) {
                float2 m = inv_scale / max(f, 1.0e-6);
                wide = lerp(f, 1.0, saturate((m - 1.25) / 0.75));
            }
            float2 s = float2(su, sv), seam = floor(s + 0.5);
            float2 t = clamp((s - seam) / max(wide, 1.0e-4), -0.5, 0.5);
            if (smode == 5) {
                /* a pixel reaching less than a 64th past a texel's edge is the slopes' rounding: none of it */
                t.x = abs(t.x) > 0.5 - 1.0 / 64 ? sign(t.x) * 0.5 : t.x;
                t.y = abs(t.y) > 0.5 - 1.0 / 64 ? sign(t.y) * 0.5 : t.y;
            }
            s = seam + t;
            su = s.x, sv = s.y;
            bilinear = true;
        }
    }
    if (smode != 0) {
        float2 lo = uvb.xy * float2(w, h) + 0.5, hi = uvb.zw * float2(w, h) - 0.5;
        float2 s = clamp(float2(su, sv), min(lo, hi), max(lo, hi));
        su = s.x, sv = s.y;
    }
    if (!bilinear) {
        c = texel(wrap((int)floor(su), w, cu, fu), wrap((int)floor(sv), h, cv, fv), k);
    } else {
        int x = (int)floor(su - 0.5), y = (int)floor(sv - 0.5);
        precise float ax = su - 0.5 - x, ay = sv - 0.5 - y;
        int x0 = wrap(x, w, cu, fu), x1 = wrap(x + 1, w, cu, fu), y0 = wrap(y, h, cv, fv), y1 = wrap(y + 1, h, cv, fv);
        float4 c00 = texel(x0, y0, k), c10 = texel(x1, y0, k), c01 = texel(x0, y1, k), c11 = texel(x1, y1, k);
        precise float4 mix = (c00 * (1 - ax) + c10 * ax) * (1 - ay) + (c01 * (1 - ax) + c11 * ax) * ay;
        c = (int4)mix;
        if (smode != 0 && mix.x > 0) {
            /* a sprite's colour weighed by alpha, so a clear texel beside it fades it out without darkening it */
            float4 aw = float4((1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay) * float4(c00.x, c10.x, c01.x, c11.x);
            c.yzw = (int3)((c00.yzw * aw.x + c10.yzw * aw.y + c01.yzw * aw.z + c11.yzw * aw.w) / (aw.x + aw.y + aw.z + aw.w));
        }
        if (smode == 4) {
            c.x = (int)(ay < 0.5 ? (ax < 0.5 ? c00.x : c10.x) : (ax < 0.5 ? c01.x : c11.x));
        }
    }
    return c;
}

/* rd_soft.c's mip_level */
int mip_level(float2 dx, float2 dy) {
    float lx = dx.x * dx.x * width * width + dx.y * dx.y * height * height;
    float ly = dy.x * dy.x * width * width + dy.y * dy.y * height * height;
    int k = (int)floor(0.5 * log2(max(lx, ly)) + mip_bias + 0.5);
    return clamp(k, 0, levels);
}

float fog_value(float z) {
    float q = fog_density * z;
    q = !(q >= 1.0) ? 1.0 : q > 255.9999 ? 255.9999 : q;
    uint bits = asuint(q);      /* q = m 2^e, m in [0.5, 1) */
    int e = (int)(bits >> 23) - 126;
    float m = asfloat((bits & 0x007FFFFFu) | 0x3F000000u);
    float pos = (m * 2.0 - 1.0) * 16.0 + (float)((e - 1) * 16);
    int i = min((int)pos, 126);
    float a = fog_table[i >> 2][i & 3], b = fog_table[(i + 1) >> 2][(i + 1) & 3];
    return (int)pos >= 127 ? fog_table[31].w : a + (b - a) * (pos - (float)i);
}

/* rd_soft.c's factors */
int4 factors(int f, int4 other, int sa, int da) {
    int4 k = 255 - da;
    k = f == 0 ? 0 : f == 1 ? 255 : f == 2 ? other : f == 3 ? 255 - other : f == 4 ? sa : f == 5 ? 255 - sa : f == 6 ? da : k;
    return k;
}

#ifdef EXACT_EDGES
/* v, opaque to the driver, which may regroup sums despite precise */
float hold(float v) {
    return asfloat(asuint(v) ^ zero);
}

/* a + b as s + e exactly */
void two_sum(float a, float b, out float s, out float e) {
    precise float x = hold(a + b);
    precise float bv = hold(x - a);
    precise float av = hold(x - bv);
    precise float r = hold(a - av) + hold(b - bv);
    s = x;
    e = r;
}

/* a * b as p + e exactly (Dekker's split) */
void two_prod(float a, float b, out float p, out float e) {
    precise float x = hold(a * b);
    precise float ca = hold(4097.0 * a);
    precise float ah = hold(ca - hold(ca - a));
    precise float al = hold(a - ah);
    precise float cb = hold(4097.0 * b);
    precise float bh = hold(cb - hold(cb - b));
    precise float bl = hold(b - bh);
    precise float r1 = hold(x - hold(ah * bh));
    precise float r2 = hold(r1 - hold(al * bh));
    precise float r3 = hold(r2 - hold(ah * bl));
    p = x;
    e = hold(al * bl) - r3;
}

/* the sign of x * y - z * w for x = x0 + x1 and so on, or 0 within about 2^-43 of the products */
float fine_sign(float2 x, float2 y, float2 z, float2 w) {
    float p0, p1, q0, q1, s0, s1;
    two_prod(x[0], y[0], p0, p1);
    two_prod(z[0], w[0], q0, q1);
    two_sum(p0, -q0, s0, s1);
    precise float c1 = x[0] * y[1] + x[1] * y[0] + x[1] * y[1];
    precise float c2 = z[0] * w[1] + z[1] * w[0] + z[1] * w[1];
    precise float e = s0 + ((s1 + (p1 - q1)) + (c1 - c2));
    float bound = (abs(p0) + abs(q0)) * 2e-13;
    return abs(e) > bound ? sign(e) : 0;
}

/* the sign of rd_soft.c's edge function, as its doubles give it, or 0 within about 1e-9 pixels */
float edge_sign(float2 a, float2 b, float px, float py) {
    precise float p1 = (b.x - a.x) * (py - a.y), p2 = (b.y - a.y) * (px - a.x);
    precise float e = p1 - p2;
    float bound = (abs(p1) + abs(p2)) * 4.8e-7;     /* 8 ulps */
    float r;
    [branch] if (abs(e) > bound) {
        r = e > 0 ? 1.0 : -1.0;
    } else {
        /* near the edge */
        float2 x, y, z, w;
        two_sum(b.x, -a.x, x[0], x[1]);
        two_sum(py, -a.y, y[0], y[1]);
        two_sum(b.y, -a.y, z[0], z[1]);
        two_sum(px, -a.x, w[0], w[1]);
        r = fine_sign(x, y, z, w);
    }
    return r;
}

/* rd_soft.c's inside() at the game's point at; cf.z: bits 0-2 own b-c, c-a and a-b, bit 3 area > 0, bit 4 every pixel */
bool inside(float2 at, float4 ab, float4 cf) {
    uint f = (uint)cf.z;
    float px = at.x, py = at.y, s = f & 8 ? 1.0 : -1.0;
    float e0 = edge_sign(ab.zw, cf.xy, px, py) * s, e1 = edge_sign(cf.xy, ab.xy, px, py) * s;
    float e2 = edge_sign(ab.xy, ab.zw, px, py) * s;
    return (f & 16) != 0 || !(e0 < 0 || e1 < 0 || e2 < 0 || (e0 == 0 && !(f & 1)) || (e1 == 0 && !(f & 2)) ||
                              (e2 == 0 && !(f & 4)));
}
#define INSIDE(at, i) inside(at, i.ab, i.cf)
#else
#define INSIDE(at, i) true
#endif

/* the fragment's key (1/w, place) is past the last layer's */
bool past_last(float q, float order, int2 p) {
    float lq = prev_q.Load(int3(p, 0)), lo = prev_o.Load(int3(p, 0));
    return q > lq || (q == lq && order > lo);
}

float4 ps_main(VOut i) : SV_Target0 {
    int2 p = int2(i.pos.xy);
    if (mode == 6) {
        return float4(mip_bias, 0, 0, 0);
    }
    if (mode == 7) {
        return under.Load(int3(p, 0));
    }
    float2 at = i.pos.xy * inv_scale;   /* the pixel's centre in the game's pixels */
    float z = i.q;
    /* the helper pixels carry the steps past the edges */
    float2 uvz_x = ddx_fine(i.uv), uvz_y = ddy_fine(i.uv);
    float z_x = ddx_fine(i.q), z_y = ddy_fine(i.q);
    float4 base_x = ddx_fine(i.base), base_y = ddy_fine(i.base);
    float3 offs_x = ddx_fine(i.offs), offs_y = ddy_fine(i.offs);
    [branch] if (i.cf.w == 4) {
        /* the game's front in its own pixels: each pixel all of the game pixel it's in, as at 640 by 480 */
        float2 c = floor(at) + 0.5, d = (c - at) / inv_scale;
        i.q += z_x * d.x + z_y * d.y;
        i.uv += uvz_x * d.x + uvz_y * d.y;
        i.base += base_x * d.x + base_y * d.y;
        i.offs += offs_x * d.x + offs_y * d.y;
        at = c, z = i.q;
    }
    if (mode == 4) {
        if (z != prev_q.Load(int3(p, 0)) || i.order != prev_o.Load(int3(p, 0))) {
            discard;
        }
        return under.Load(int3(p, 0));
    }
    /* the edge test last, as the dearest (fxc's || doesn't skip) */
    if (!(z > 0.0)) {
        discard;
    }
    if (mode == 1) {
        if (!past_last(z, i.order, p)) {
            discard;
        }
        [branch] if (!INSIDE(at, i)) {
            discard;
        }
        return float4(z, 0, 0, 0);
    }
    if (mode == 2) {
        if (!past_last(z, i.order, p) || z != min_q.Load(int3(p, 0))) {
            discard;
        }
        [branch] if (!INSIDE(at, i)) {
            discard;
        }
        return float4(i.order, 0, 0, 0);
    }
    if (mode == 5) {
        [branch] if (!INSIDE(at, i)) {
            discard;
        }
        return float4(1, 0, 0, 0);
    }
    if (mode == 3 && (z != min_q.Load(int3(p, 0)) || i.order != min_o.Load(int3(p, 0)))) {
        discard;
    }
    [branch] if (!INSIDE(at, i)) {
        discard;
    }
    float rz = 1.0 / z;
    float u = i.uv.x * rz, v = i.uv.y * rz;
    int4 base = clamp((int4)(i.base * rz + 0.5), 0, 255);
    int3 offs = clamp((int3)(i.offs * rz + 0.5), 0, 255);
    if (!(flags & F_USE_ALPHA)) {
        base.x = 255;
    }
    int4 c = base;
    if (flags & F_TEX) {
        int k = 0;
        /* cf.w: a flat triangle, a sprite (4: the front, as the game has it) */
        bool sprite = i.cf.w != 0 && i.cf.w != 4;
        int smode = sprite ? sprite_mode : 0;
        /* texel: a sprite on the lattice read where its texels fall, any other as sharp reads it */
        bool texel_space = smode == 4 && i.cf.w == 3;
        smode = smode == 4 ? 1 : smode;
        float2 uv = float2(u, v), fw = 0;
        if (levels > 0) {
            /* as rd_soft.c takes the steps */
            k = mip_level((uvz_x - uv * z_x) * rz, (uvz_y - uv * z_y) * rz);
        }
        if (smode == 1) {
            fw = (abs((uvz_x - uv * z_x) * rz) + abs((uvz_y - uv * z_y) * rz)) * float2(width, height);
        }
        /* u and v a game pixel across and down, and the texels in a game pixel */
        float2 jx = (uvz_x - uv * z_x) * rz / inv_scale, jy = (uvz_y - uv * z_y) * rz / inv_scale;
        float2 m = (abs(jx) + abs(jy)) * float2(width, height);
        int4 t;
        bool bilinear = (flags & F_BILINEAR) != 0;
        [branch] if (texel_space) {
            /* each pixel the mean of the texels under it, where the game puts them */
            t = sample_tex(k, u, v, 5, fw, i.uvb);
        } else if (i.cf.w == 2) {
            /* an upright sprite drawn larger, its texels fitted to it: read where they fall */
            t = sample_tex(k, u, v, smode == 0 ? 3 : smode == 3 ? 4 : smode, fw, i.uvb);
        } else if (sprite && inv_scale < 1.0 && !(bilinear && max(m.x, m.y) < 0.8)) {
            /* pixel art drawn larger: its colours at the game's pixel centres, as at 640 by 480, each from the
               sprite's own texels, blended weighed by alpha: sharp over a pixel at their edges, smooth all the way
               across, and arcade all the way when the game filters it */
            float2 seam = floor(at + 0.5);
            float2 f = smode == 1 ? clamp((at - seam) / inv_scale, -0.5, 0.5) + 0.5
                     : smode == 3 || (smode == 0 && bilinear) ? at - seam + 0.5 : step(seam, at);
            int tap = smode == 0 ? 3 : 2;
            float2 g0 = seam - 0.5 - at, g1 = seam + 0.5 - at;
            float2 uv00 = uv + jx * g0.x + jy * g0.y, uv10 = uv + jx * g1.x + jy * g0.y;
            float2 uv01 = uv + jx * g0.x + jy * g1.y, uv11 = uv + jx * g1.x + jy * g1.y;
            [branch] if (all(f * (1 - f) == 0)) {
                float2 s = f.y == 0 ? (f.x == 0 ? uv00 : uv10) : (f.x == 0 ? uv01 : uv11);
                t = sample_tex(k, s.x, s.y, tap, fw, i.uvb);
            } else {
                float4 c00 = sample_tex(k, uv00.x, uv00.y, tap, fw, i.uvb), c10 = sample_tex(k, uv10.x, uv10.y, tap, fw, i.uvb);
                float4 c01 = sample_tex(k, uv01.x, uv01.y, tap, fw, i.uvb), c11 = sample_tex(k, uv11.x, uv11.y, tap, fw, i.uvb);
                float4 aw = float4((1 - f.x) * (1 - f.y), f.x * (1 - f.y), (1 - f.x) * f.y, f.x * f.y) *
                            float4(c00.x, c10.x, c01.x, c11.x);
                float a = aw.x + aw.y + aw.z + aw.w;
                /* alpha the game pixel's own, as at 640 by 480 */
                float2 own = step(seam, at);
                t.x = (int)(own.y == 0 ? (own.x == 0 ? c00.x : c10.x) : (own.x == 0 ? c01.x : c11.x));
                t.yzw = a > 0 ? (int3)((c00.yzw * aw.x + c10.yzw * aw.y + c01.yzw * aw.z + c11.yzw * aw.w) / a) : (int3)c00.yzw;
            }
        } else {
            t = sample_tex(k, u, v, smode, fw, i.uvb);
        }
        if (!(flags & F_TEX_ALPHA)) {
            t.x = 255;
        }
        uint shading = flags / F_SHADING & 3;
        if (shading == 0) {
            c = t;
        } else if (shading == 1) {
            c = int4(t.x, div255(t.yzw * base.yzw));
        } else if (shading == 2) {
            c = int4(base.x, div255(t.yzw * t.x + base.yzw * (255 - t.x)));
        } else {
            c = int4(div255(t.x * base.x), div255(t.yzw * base.yzw));
        }
    }
    if (flags & F_OFFSET) {
        c.yzw = clamp(c.yzw + offs, 0, 255);
    }
    if (flags & (F_FOG_TABLE | F_FOG_TABLE2)) {
        float f = fog_value(z);
        int4 k = (int4)fog_color;
        if (flags & F_FOG_TABLE2) {
            c = int4((int)(f * 255.0 + 0.5), k.yzw);
        } else {
            float3 d = (float3)(k.yzw - c.yzw);
            c.yzw += (int3)(d * f + ((float3)(k.yzw > c.yzw) - 0.5));
        }
    }
    if ((flags & F_PUNCH) && c.x < 128) {
        discard;
    }
    if (mode == 3) {
        /* rd_soft.c's blend */
        float4 u = under.Load(int3(p, 0));
        int4 d = int4(u.a * 255.0 + 0.5, u.r * 255.0 + 0.5, u.g * 255.0 + 0.5, u.b * 255.0 + 0.5);
        int4 fs = factors(blend & 15, d, c.x, d.x), fd = factors(blend >> 4, c, c.x, d.x);
        c = clamp((int4)((uint4)(c * fs + d * fd + 127) / 255u), 0, 255);
    }
    return float4(c.y, c.z, c.w, c.x) / 255.0;
}
