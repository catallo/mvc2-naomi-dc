/*
 * rd_gpu.c  The SDL_GPU back end. The translucent list is drawn far to near in an order found on the CPU, and
 * where no order holds for a whole triangle it is peeled a layer at a time, each fragment blended by the shader in
 * rd_soft.c's integer sums.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <SDL3/SDL.h>

#include "rd.h"
#include "rd_gpu.h"
#include "rd_gpu_shaders.h"
#include "rd_list.h"

#if defined(__GNUC__) && defined(__i386__)
#pragma GCC target("sse2", "fpmath=sse")
#endif

typedef struct {
    float x, y;
    float depth, q;         /* the depth buffer's value; 1/w (1 for the background) */
    float u, v;             /* u/w, v/w */
    float base[4];          /* a r g b, 0 to 255, over w */
    float offs[3];          /* r g b over w */
    float order;            /* in its list */
    float ab[4], cf[4];     /* corners a, b and c, and the edges' flags (rd_gpu.hlsl's inside) */
    float uvb[4];           /* a sprite's u and v from and to: its texels are read only between */
} GVTX;

/* GVTX's depth to offs */
#define PLANE_VALUES 11

/* rd_gpu.hlsl's Draw */
typedef struct {
    uint32_t flags;
    int32_t width, height, levels;
    int32_t palette;
    float mip_bias;
    int32_t mode;
    int32_t blend;
} DRAW;

typedef struct {
    float fog_table[128];
    float fog_color[4];     /* a r g b, 0 to 255 */
    float fog_density;
    uint32_t zero;          /* 0, unknown to the shader compiler, so it can't regroup the sums */
    float inv_scale;
    int32_t sprite_mode;
} FRAMEU;

enum {
    F_TEX = 1, F_PAL = 2, F_BILINEAR = 4, F_TEX_ALPHA = 8, F_USE_ALPHA = 16, F_OFFSET = 32,
    F_FOG_TABLE = 64, F_FOG_TABLE2 = 128, F_PUNCH = 256, F_SHADING = 512, F_UV = 2048
};

/* 8, though the shader reads 7: SDL's Direct3D 12 driver needs a table size that divides its heaps' */
#define BINDINGS 8

enum { P_BACKGROUND, P_OPAQUE, P_PUNCH, P_SORTED, P_KEY, P_PEEL, P_COUNT, P_FIX, P_FILL, P_COPY, P_KINDS };

/* a run of triangles of one state */
typedef struct {
    int first, count, state;
} BATCH;

typedef struct {
    const RD_TEXTURE* t;
    uint32_t serial;
    uint32_t used;          /* the last frame that drew it */
    SDL_GPUTexture* g;
    int w, h, levels, pal;  /* g's */
} GTEX;

/* a texture's texels, to be copied from the staging buffer at 'at' */
typedef struct {
    SDL_GPUTexture* g;
    const RD_TEXTURE* t;
    uint32_t at;
} PENDING;

/* frames a texture goes undrawn before it is let go */
#define EVICT_FRAMES 600

static SDL_GPUDevice* dev;
static SDL_GPUShader *vs, *ps;
static SDL_GPUSampler* point;
static SDL_GPUTexture *frame_tex[2], *depth_tex, *key_q[2], *key_o[2], *pal_tex, *dummy_rgba, *dummy_r32;
static SDL_GPUGraphicsPipeline* pipes[P_KINDS][8][8][2];
static SDL_GPUBuffer* vbuf;
static SDL_GPUTransferBuffer* vtrans;
static uint32_t vcap;
static GVTX* verts;         /* mapped, maybe write-combined: each vertex stored whole */
static BATCH* batches[RD_LISTS];
static int nbatches[RD_LISTS], maxbatches[RD_LISTS];
static GTEX* gtex;
static int ngtex, maxgtex;
static int* hash;           /* gtex's indices by texture, -1 empty */
static uint32_t frame_no;
static SDL_GPUTransferBuffer* tb;   /* rd_gpu_read's */
static uint32_t pal_serial = 0xFFFFFFFFu;
static int exact_edges;     /* rd_soft.c's own edges (rd_gpu.hlsl's EXACT_EDGES) */
static int layers_env;      /* MVC2_GPULAYERS */
static int fog_on;
static SDL_GPUTexture* state_tex[RD_MAX_STATES];

/* rd_gpu_view's: the frame drawn at scale times the game's size (frame_w by frame_h), then filtered to out_w by out_h
   (0: not) */
static float want_scale = 1;
static int want_sprites, want_w, want_h;
static float scale = 1;     /* the targets as made */
static int frame_w = RD_WIDTH, frame_h = RD_HEIGHT, out_w, out_h;
static int sprite_mode;
static int want_hud, hud_mode, shared_grid;
static SDL_GPUTexture* out_tex;
static SDL_GPUShader *full_vs, *resolve_ps;
static SDL_GPUGraphicsPipeline* resolve_pipe;
static uint32_t tb_size;
/* the layers a frame needs (rd_gpu_count.hlsl): each pixel's translucent fragments counted, the most found, and the
   peeling's draws made from it, so that the layers past it draw nothing. MVC2_GPUCOUNT=0, or a device without
   compute, draws them all */
static int count_on;
static int sort_on;         /* MVC2_GPUSORT=0: all of the translucent list peeled */
static int sorting;         /* this frame */
static SDL_GPUTexture* count_tex;
static SDL_GPUComputePipeline *most_cp, *draws_cp;
static SDL_GPUBuffer *most_buf, *draws_buf, *args_buf;
static SDL_GPUTransferBuffer* draws_trans;
static uint32_t draws_cap, args_cap;    /* their entries */
/* with the counts, each pixel's translucent fragments counted in the stencil too, so that a layer draws only where
   the pixel has one */
static int stencil_on, stencil_clear;
static SDL_GPUTextureFormat depth_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;

static uint32_t argb16_to_rgba(int format, uint32_t p) {
    uint32_t a, r, g, b;
    if (format == RD_ARGB1555) {
        a = p & 0x8000 ? 255 : 0, r = (p >> 10 & 31) * 255 / 31, g = (p >> 5 & 31) * 255 / 31, b = (p & 31) * 255 / 31;
    } else if (format == RD_RGB565) {
        a = 255, r = (p >> 11) * 255 / 31, g = (p >> 5 & 63) * 255 / 63, b = (p & 31) * 255 / 31;
    } else {
        a = (p >> 12) * 17, r = (p >> 8 & 15) * 17, g = (p >> 4 & 15) * 17, b = (p & 15) * 17;
    }
    return a << 24 | b << 16 | g << 8 | r;
}

static uint32_t argb_to_rgba(uint32_t c) {
    return (c & 0xFF00FF00u) | (c >> 16 & 255) | (c & 255) << 16;
}

static SDL_GPUTexture* make_texture(SDL_GPUTextureFormat format, int w, int h, int levels, SDL_GPUTextureUsageFlags usage) {
    SDL_GPUTextureCreateInfo ci;
    memset(&ci, 0, sizeof ci);
    ci.type = SDL_GPU_TEXTURETYPE_2D;
    ci.format = format;
    ci.usage = usage;
    ci.width = (Uint32)w;
    ci.height = (Uint32)h;
    ci.layer_count_or_depth = 1;
    ci.num_levels = (Uint32)levels;
    ci.sample_count = SDL_GPU_SAMPLECOUNT_1;
    return SDL_CreateGPUTexture(dev, &ci);
}

/* the frame's texels to upload, all in one staging buffer: each level at a multiple of 512 bytes */
#define STAGE_ALIGN 512u
static PENDING* pending;
static int npending, maxpending;
static SDL_GPUTransferBuffer* stage;
static uint32_t stage_cap, stage_used;

static int texel_bytes(const RD_TEXTURE* t) {
    return t->format == RD_PAL4 || t->format == RD_PAL8 ? 1 : 4;
}

static uint32_t level_bytes(const RD_TEXTURE* t, int k) {
    int w = t->width >> k, h = t->height >> k;
    w = w < 1 ? 1 : w, h = h < 1 ? 1 : h;
    return ((uint32_t)(w * h * texel_bytes(t)) + STAGE_ALIGN - 1) & ~(STAGE_ALIGN - 1);
}

static void add_pending(SDL_GPUTexture* g, const RD_TEXTURE* t) {
    int k;
    if (npending == maxpending) {
        int n = maxpending ? maxpending * 2 : 256;
        PENDING* p = realloc(pending, (size_t)n * sizeof *p);
        if (p == NULL) {
            return;
        }
        pending = p, maxpending = n;
    }
    pending[npending].g = g, pending[npending].t = t, pending[npending].at = stage_used;
    npending++;
    for (k = 0; k <= t->levels; k++) {
        stage_used += level_bytes(t, k);
    }
}

/* the pending texels and the palette (pal: 1024 ARGB, or NULL) written to the staging buffer, then copied by
   copy into their textures */
static void upload_pending(SDL_GPUCopyPass* copy, const uint32_t* pal) {
    uint32_t pal_at = stage_used;
    uint8_t* p;
    int i, k;
    if (npending == 0 && pal == NULL) {
        return;
    }
    if (pal != NULL) {
        stage_used += 4096;
    }
    if (stage_used > stage_cap) {
        SDL_GPUTransferBufferCreateInfo ti = { SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, 0, 0 };
        if (stage != NULL) {
            SDL_ReleaseGPUTransferBuffer(dev, stage);
        }
        ti.size = stage_used + stage_used / 2;
        stage = SDL_CreateGPUTransferBuffer(dev, &ti);
        stage_cap = stage != NULL ? ti.size : 0;
    }
    p = stage != NULL ? SDL_MapGPUTransferBuffer(dev, stage, true) : NULL;
    if (p == NULL) {
        npending = 0, stage_used = 0;
        return;
    }
    for (i = 0; i < npending; i++) {
        const RD_TEXTURE* t = pending[i].t;
        uint32_t at = pending[i].at;
        for (k = 0; k <= t->levels; k++) {
            int w = t->width >> k, h = t->height >> k, n, j;
            const void* d = k == 0 ? t->texels : t->mip[k];
            w = w < 1 ? 1 : w, h = h < 1 ? 1 : h, n = w * h;
            if (texel_bytes(t) == 1) {
                memcpy(p + at, d, (size_t)n);
            } else {
                uint32_t* px = (uint32_t*)(p + at);
                for (j = 0; j < n; j++) {
                    px[j] = argb16_to_rgba(t->format, ((const uint16_t*)d)[j]);
                }
            }
            at += level_bytes(t, k);
        }
    }
    if (pal != NULL) {
        uint32_t* px = (uint32_t*)(p + pal_at);
        for (i = 0; i < 1024; i++) {
            px[i] = argb_to_rgba(pal[i]);
        }
    }
    SDL_UnmapGPUTransferBuffer(dev, stage);
    for (i = 0; i < npending; i++) {
        const RD_TEXTURE* t = pending[i].t;
        uint32_t at = pending[i].at;
        for (k = 0; k <= t->levels; k++) {
            SDL_GPUTextureTransferInfo src;
            SDL_GPUTextureRegion dst;
            int w = t->width >> k, h = t->height >> k;
            w = w < 1 ? 1 : w, h = h < 1 ? 1 : h;
            memset(&src, 0, sizeof src);
            src.transfer_buffer = stage, src.offset = at;
            memset(&dst, 0, sizeof dst);
            dst.texture = pending[i].g;
            dst.mip_level = (Uint32)k;
            dst.w = (Uint32)w, dst.h = (Uint32)h, dst.d = 1;
            /* cycled before its first level, as the GPU may still be reading the last frame's */
            SDL_UploadToGPUTexture(copy, &src, &dst, k == 0);
            at += level_bytes(t, k);
        }
    }
    if (pal != NULL) {
        SDL_GPUTextureTransferInfo src;
        SDL_GPUTextureRegion dst;
        memset(&src, 0, sizeof src);
        src.transfer_buffer = stage, src.offset = pal_at;
        memset(&dst, 0, sizeof dst);
        dst.texture = pal_tex;
        dst.w = 1024, dst.h = 1, dst.d = 1;
        SDL_UploadToGPUTexture(copy, &src, &dst, true);
    }
    npending = 0, stage_used = 0;
}

static uint32_t hash_of(const RD_TEXTURE* t) {
    return (uint32_t)((uintptr_t)t >> 4) * 2654435761u & (uint32_t)(2 * maxgtex - 1);
}

static void rehash(void) {
    int i;
    for (i = 0; i < 2 * maxgtex; i++) {
        hash[i] = -1;
    }
    for (i = 0; i < ngtex; i++) {
        uint32_t h = hash_of(gtex[i].t);
        while (hash[h] >= 0) {
            h = (h + 1) & (uint32_t)(2 * maxgtex - 1);
        }
        hash[h] = i;
    }
}

/* t's entry in the cache, a new one if it has none */
static GTEX* entry(const RD_TEXTURE* t) {
    uint32_t h;
    if (maxgtex > 0) {
        for (h = hash_of(t); hash[h] >= 0; h = (h + 1) & (uint32_t)(2 * maxgtex - 1)) {
            if (gtex[hash[h]].t == t) {
                return &gtex[hash[h]];
            }
        }
    }
    if (ngtex == maxgtex) {
        int n = maxgtex ? maxgtex * 2 : 256;
        GTEX* p = realloc(gtex, (size_t)n * sizeof *p);
        int* q = realloc(hash, (size_t)n * 2 * sizeof *q);
        if (p != NULL) {
            gtex = p;
        }
        if (q != NULL) {
            hash = q;
        }
        if (p == NULL || q == NULL) {
            return NULL;
        }
        maxgtex = n;
        rehash();
    }
    gtex[ngtex].t = t, gtex[ngtex].g = NULL;
    h = hash_of(t);
    while (hash[h] >= 0) {
        h = (h + 1) & (uint32_t)(2 * maxgtex - 1);
    }
    hash[h] = ngtex;
    return &gtex[ngtex++];
}

/* textures let go, kept for the next of their size and format: making or freeing one mid-game stalls the frame */
#define SPARE_BYTES (64u << 20)
static GTEX* spare;
static int nspare, maxspare;
static uint32_t spare_bytes;

static uint32_t gtex_bytes(const GTEX* e) {
    return (uint32_t)e->w * (uint32_t)e->h * (e->pal ? 1u : 4u) * (e->levels > 0 ? 4u : 3u) / 3u;
}

static void keep_spare(const GTEX* e) {
    if (e->g == NULL) {
        return;
    }
    if (nspare == maxspare) {
        int n = maxspare ? maxspare * 2 : 256;
        GTEX* p = realloc(spare, (size_t)n * sizeof *p);
        if (p == NULL) {
            SDL_ReleaseGPUTexture(dev, e->g);
            return;
        }
        spare = p, maxspare = n;
    }
    spare[nspare++] = *e;
    spare_bytes += gtex_bytes(e);
    while (spare_bytes > SPARE_BYTES) {
        /* the oldest first */
        spare_bytes -= gtex_bytes(&spare[0]);
        SDL_ReleaseGPUTexture(dev, spare[0].g);
        memmove(&spare[0], &spare[1], (size_t)--nspare * sizeof *spare);
    }
}

static SDL_GPUTexture* take_spare(int w, int h, int levels, int pal) {
    int i;
    for (i = nspare - 1; i >= 0; i--) {
        if (spare[i].w == w && spare[i].h == h && spare[i].levels == levels && spare[i].pal == pal) {
            SDL_GPUTexture* g = spare[i].g;
            spare_bytes -= gtex_bytes(&spare[i]);
            memmove(&spare[i], &spare[i + 1], (size_t)(nspare - i - 1) * sizeof *spare);
            nspare--;
            return g;
        }
    }
    return NULL;
}

static void evict(void) {
    int i, n = 0;
    for (i = 0; i < ngtex; i++) {
        if (frame_no - gtex[i].used > EVICT_FRAMES) {
            keep_spare(&gtex[i]);
        } else {
            gtex[n++] = gtex[i];
        }
    }
    if (n != ngtex) {
        ngtex = n;
        rehash();
    }
}

/* RGBA8, or R8 indices for the palette formats: kept while its size and format are, its texels queued again
   when they change */
static SDL_GPUTexture* texture(const RD_TEXTURE* t) {
    int pal = t->format == RD_PAL4 || t->format == RD_PAL8;
    GTEX* e = entry(t);
    if (e == NULL) {
        return NULL;
    }
    e->used = frame_no;
    if (e->serial == t->serial && e->g != NULL) {
        return e->g;
    }
    if (e->g != NULL && (e->w != t->width || e->h != t->height || e->levels != t->levels || e->pal != pal)) {
        keep_spare(e);
        e->g = NULL;
    }
    if (e->g == NULL) {
        e->g = take_spare(t->width, t->height, t->levels, pal);
        if (e->g == NULL) {
            e->g = make_texture(pal ? SDL_GPU_TEXTUREFORMAT_R8_UNORM : SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, t->width,
                                t->height, t->levels + 1, SDL_GPU_TEXTUREUSAGE_SAMPLER);
        }
        e->w = t->width, e->h = t->height, e->levels = t->levels, e->pal = pal;
    }
    e->serial = t->serial;
    if (e->g != NULL) {
        add_pending(e->g, t);
    }
    return e->g;
}

static SDL_GPUBlendFactor factor(int f, int src) {
    switch (f) {
    case RD_ZERO: return SDL_GPU_BLENDFACTOR_ZERO;
    case RD_ONE: return SDL_GPU_BLENDFACTOR_ONE;
    case RD_OTHER: return src ? SDL_GPU_BLENDFACTOR_DST_COLOR : SDL_GPU_BLENDFACTOR_SRC_COLOR;
    case RD_INV_OTHER: return src ? SDL_GPU_BLENDFACTOR_ONE_MINUS_DST_COLOR : SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_COLOR;
    case RD_SRC_ALPHA: return SDL_GPU_BLENDFACTOR_SRC_ALPHA;
    case RD_INV_SRC_ALPHA: return SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    case RD_DST_ALPHA: return SDL_GPU_BLENDFACTOR_DST_ALPHA;
    default: return SDL_GPU_BLENDFACTOR_ONE_MINUS_DST_ALPHA;
    }
}

/* a and b: the depth compare and write (opaque) or the blend factors */
static SDL_GPUGraphicsPipeline* pipeline(int kind, int a, int b) {
    SDL_GPUGraphicsPipeline** slot = &pipes[kind][a & 7][b & 7][0];
    SDL_GPUGraphicsPipelineCreateInfo ci;
    SDL_GPUVertexBufferDescription vb;
    SDL_GPUVertexAttribute at[8];
    SDL_GPUColorTargetDescription ct;
    int i;
    static const Uint32 offsets[8] = { 0, 8, 16, 24, 40, 56, 72, 88 };
    static const SDL_GPUVertexElementFormat formats[8] = {
        SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
        SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4,
        SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4
    };
    if (*slot != NULL) {
        return *slot;
    }
    memset(&ci, 0, sizeof ci);
    memset(&vb, 0, sizeof vb);
    vb.slot = 0;
    vb.pitch = sizeof(GVTX);
    vb.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;
    for (i = 0; i < 8; i++) {
        at[i].location = (Uint32)i;
        at[i].buffer_slot = 0;
        at[i].format = formats[i];
        at[i].offset = offsets[i];
    }
    ci.vertex_shader = vs;
    ci.fragment_shader = ps;
    ci.vertex_input_state.vertex_buffer_descriptions = &vb;
    ci.vertex_input_state.num_vertex_buffers = 1;
    ci.vertex_input_state.vertex_attributes = at;
    ci.vertex_input_state.num_vertex_attributes = 8;
    ci.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    ci.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
    ci.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
    ci.rasterizer_state.front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE;
    ci.multisample_state.sample_count = SDL_GPU_SAMPLECOUNT_1;
    ci.depth_stencil_state.enable_depth_test = true;
    if (stencil_on && kind >= P_KEY) {
        /* the count adds each fragment; the layers pass where the count is over the layer (the reference) */
        SDL_GPUStencilOpState* f = &ci.depth_stencil_state.front_stencil_state;
        f->fail_op = f->depth_fail_op = SDL_GPU_STENCILOP_KEEP;
        f->pass_op = kind == P_COUNT ? SDL_GPU_STENCILOP_INCREMENT_AND_CLAMP : SDL_GPU_STENCILOP_KEEP;
        f->compare_op = kind == P_COUNT ? SDL_GPU_COMPAREOP_ALWAYS : kind == P_COPY && a ? SDL_GPU_COMPAREOP_EQUAL
                      : SDL_GPU_COMPAREOP_LESS;
        ci.depth_stencil_state.back_stencil_state = *f;
        ci.depth_stencil_state.compare_mask = 0xFF;
        ci.depth_stencil_state.write_mask = kind == P_COUNT ? 0xFF : 0;
        ci.depth_stencil_state.enable_stencil_test = true;
    }
    memset(&ct, 0, sizeof ct);
    ct.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    switch (kind) {
    case P_BACKGROUND:
        ci.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_ALWAYS;
        ci.depth_stencil_state.enable_depth_write = true;
        break;
    case P_OPAQUE:
        ci.depth_stencil_state.compare_op = (SDL_GPUCompareOp)(a + 1);
        ci.depth_stencil_state.enable_depth_write = b != 0;
        break;
    case P_KEY:
        ct.format = SDL_GPU_TEXTUREFORMAT_R32_FLOAT;
        ct.blend_state.enable_blend = true;
        ct.blend_state.src_color_blendfactor = ct.blend_state.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
        ct.blend_state.src_alpha_blendfactor = ct.blend_state.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
        ct.blend_state.color_blend_op = ct.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_MIN;
        ci.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_GREATER_OR_EQUAL;
        break;
    case P_COUNT:
        ct.format = SDL_GPU_TEXTUREFORMAT_R32_FLOAT;
        ct.blend_state.enable_blend = true;
        ct.blend_state.src_color_blendfactor = ct.blend_state.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
        ct.blend_state.src_alpha_blendfactor = ct.blend_state.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
        ct.blend_state.color_blend_op = ct.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_ADD;
        ci.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_GREATER_OR_EQUAL;
        break;
    case P_FILL:    /* a key target's pixels set where the layer draws */
        ct.format = SDL_GPU_TEXTUREFORMAT_R32_FLOAT;
        ci.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_ALWAYS;
        break;
    case P_COPY:    /* the frame copied where the count is over (a: is) the layer */
        ci.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_ALWAYS;
        break;
    case P_PEEL:    /* blended by the shader, over the opaque depth */
    case P_FIX:
        ci.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_GREATER_OR_EQUAL;
        break;
    default:        /* P_PUNCH and P_SORTED: blended, over the opaque depth; P_PUNCH writing it */
        ct.blend_state.enable_blend = true;
        ct.blend_state.src_color_blendfactor = ct.blend_state.src_alpha_blendfactor = factor(a, 1);
        ct.blend_state.dst_color_blendfactor = ct.blend_state.dst_alpha_blendfactor = factor(b, 0);
        ct.blend_state.color_blend_op = ct.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_ADD;
        ci.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_GREATER_OR_EQUAL;
        ci.depth_stencil_state.enable_depth_write = kind == P_PUNCH;
        break;
    }
    ci.target_info.color_target_descriptions = &ct;
    ci.target_info.num_color_targets = 1;
    ci.target_info.depth_stencil_format = depth_format;
    ci.target_info.has_depth_stencil_target = true;
    *slot = SDL_CreateGPUGraphicsPipeline(dev, &ci);
    if (*slot == NULL) {
        SDL_Log("rd_gpu: pipeline %d %d %d: %s", kind, a, b, SDL_GetError());
    }
    return *slot;
}

static SDL_GPUShader* shader(const unsigned char* code, size_t size, SDL_GPUShaderFormat format, const char* entry,
                             SDL_GPUShaderStage stage, int samplers, int uniforms) {
    SDL_GPUShaderCreateInfo ci;
    memset(&ci, 0, sizeof ci);
    ci.code = code;
    ci.code_size = size;
    ci.entrypoint = entry;
    ci.format = format;
    ci.stage = stage;
    ci.num_samplers = (Uint32)samplers;
    ci.num_uniform_buffers = (Uint32)uniforms;
    return SDL_CreateGPUShader(dev, &ci);
}

static SDL_GPUComputePipeline* compute(const unsigned char* code, size_t size, SDL_GPUShaderFormat format,
                                       const char* entry, int samplers, int reads, int writes, int x, int y) {
    SDL_GPUComputePipelineCreateInfo ci;
    memset(&ci, 0, sizeof ci);
    ci.code = code;
    ci.code_size = size;
    ci.entrypoint = entry;
    ci.format = format;
    ci.num_samplers = (Uint32)samplers;
    ci.num_readonly_storage_buffers = (Uint32)reads;
    ci.num_readwrite_storage_buffers = (Uint32)writes;
    ci.num_uniform_buffers = 1;
    ci.threadcount_x = (Uint32)x, ci.threadcount_y = (Uint32)y, ci.threadcount_z = 1;
    return SDL_CreateGPUComputePipeline(dev, &ci);
}

static SDL_GPUGraphicsPipeline* make_resolve_pipe(void) {
    SDL_GPUGraphicsPipelineCreateInfo ci;
    SDL_GPUColorTargetDescription ct;
    memset(&ci, 0, sizeof ci);
    memset(&ct, 0, sizeof ct);
    ci.vertex_shader = full_vs;
    ci.fragment_shader = resolve_ps;
    ci.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    ci.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
    ci.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
    ci.multisample_state.sample_count = SDL_GPU_SAMPLECOUNT_1;
    ct.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    ci.target_info.color_target_descriptions = &ct;
    ci.target_info.num_color_targets = 1;
    return SDL_CreateGPUGraphicsPipeline(dev, &ci);
}

static void release(SDL_GPUTexture** t) {
    if (*t != NULL) {
        SDL_ReleaseGPUTexture(dev, *t);
        *t = NULL;
    }
}

/* the frame's targets at n times the game's size, its height a multiple of 3 so it stays 4:3 in whole pixels; 0 when
   one could not be made */
static int make_targets(float n) {
    SDL_GPUTextureUsageFlags key = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;
    int i, h = 3 * (int)(RD_HEIGHT * n / 3 + 0.5f), w = h / 3 * 4;
    for (i = 0; i < 2; i++) {
        release(&frame_tex[i]), release(&key_q[i]), release(&key_o[i]);
    }
    release(&depth_tex), release(&count_tex);
    for (i = 0; i < 2; i++) {
        frame_tex[i] = make_texture(SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, w, h, 1,
                                    SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET);
        key_q[i] = make_texture(SDL_GPU_TEXTUREFORMAT_R32_FLOAT, w, h, 1, key);
        key_o[i] = make_texture(SDL_GPU_TEXTUREFORMAT_R32_FLOAT, w, h, 1, key);
    }
    depth_tex = make_texture(depth_format, w, h, 1, SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET);
    if (count_on) {
        count_tex = make_texture(SDL_GPU_TEXTUREFORMAT_R32_FLOAT, w, h, 1, key);
    }
    scale = (float)h / RD_HEIGHT, frame_w = w, frame_h = h;
    return frame_tex[0] != NULL && frame_tex[1] != NULL && key_q[0] != NULL && key_q[1] != NULL && key_o[0] != NULL &&
           key_o[1] != NULL && depth_tex != NULL;
}

/* auto: a whole multiple, filtered down to the window; texel: the window's own size */
static float view_scale(float resolution, int aa, int h, int sprites) {
    float n = resolution > 0 ? resolution
            : h <= 0 ? 1
            : sprites == RD_SPRITES_TEXEL ? (float)(h * (aa < 1 ? 1 : aa)) / RD_HEIGHT
            : (float)((h * (aa < 1 ? 1 : aa) + RD_HEIGHT - 1) / RD_HEIGHT);
    return n < 1 ? 1 : n > RD_GPU_MAX_SCALE ? RD_GPU_MAX_SCALE : n;
}

void rd_gpu_view(float resolution, int aa, int w, int h, int sprites) {
    want_scale = view_scale(resolution, aa, h, sprites);
    want_w = w, want_h = h;
    want_sprites = sprites;
}

void rd_gpu_hud(int mode) {
    want_hud = mode;
}

void rd_gpu_texel_grid(int shared) {
    shared_grid = shared;
}

/* the targets as rd_gpu_view wants them: a smaller frame when the larger can't be made */
static void fit_view(void) {
    float n = want_scale;
    int w = 0, h = 0;
    if (3 * (int)(RD_HEIGHT * n / 3 + 0.5f) != frame_h) {
        while (!make_targets(n) && n > 1) {
            n--;
        }
        want_scale = scale;
    }
    if (want_w > 0 && want_h > 0 && (want_w < frame_w || want_h < frame_h)) {
        w = want_w, h = want_h;
    }
    if (w != out_w || h != out_h) {
        release(&out_tex);
        if (w > 0 && resolve_pipe != NULL) {
            out_tex = make_texture(SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, w, h, 1,
                                   SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET);
        }
        out_w = out_tex != NULL ? w : 0, out_h = out_tex != NULL ? h : 0;
    }
    sprite_mode = scale > 1 ? want_sprites : 0;
    hud_mode = scale > 1 ? want_hud : 0;
}

void rd_gpu_view_size(float resolution, int aa, int h, int sprites, int* fw, int* fh) {
    *fh = 3 * (int)(RD_HEIGHT * view_scale(resolution, aa, h, sprites) / 3 + 0.5f), *fw = *fh / 3 * 4;
}

void rd_gpu_frame_size(int* w, int* h) {
    *w = out_w > 0 ? out_w : frame_w;
    *h = out_w > 0 ? out_h : frame_h;
}

bool rd_gpu_init(SDL_GPUDevice* device) {
    SDL_GPUSamplerCreateInfo si;
    const char* e = getenv("MVC2_GPULAYERS");
    const char* q = getenv("MVC2_RDQUALITY");
    const char* g = getenv("MVC2_GPUEDGES");
    const char* c = getenv("MVC2_GPUCOUNT");
    SDL_GPUShaderStage vtx = SDL_GPU_SHADERSTAGE_VERTEX, frag = SDL_GPU_SHADERSTAGE_FRAGMENT;
    float n = want_scale;
    dev = device;
    layers_env = e != NULL && atoi(e) > 0 ? atoi(e) : 0;
    if (q != NULL) {
        rd_quality(q);
    }
    exact_edges = g == NULL || atoi(g) != 0;
    count_on = c == NULL || atoi(c) != 0;
    sort_on = getenv("MVC2_GPUSORT") == NULL || atoi(getenv("MVC2_GPUSORT")) != 0;
    if (SDL_GetGPUShaderFormats(dev) & SDL_GPU_SHADERFORMAT_DXBC) {
        SDL_GPUShaderFormat f = SDL_GPU_SHADERFORMAT_DXBC;
        vs = shader(rd_gpu_vs_dxbc, sizeof rd_gpu_vs_dxbc, f, "vs_main", vtx, 0, 0);
        ps = exact_edges ? shader(rd_gpu_ps_exact_dxbc, sizeof rd_gpu_ps_exact_dxbc, f, "ps_main", frag, BINDINGS, 2)
                         : shader(rd_gpu_ps_dxbc, sizeof rd_gpu_ps_dxbc, f, "ps_main", frag, BINDINGS, 2);
        full_vs = shader(rd_gpu_full_vs_dxbc, sizeof rd_gpu_full_vs_dxbc, f, "vs_full", vtx, 0, 0);
        resolve_ps = shader(rd_gpu_resolve_ps_dxbc, sizeof rd_gpu_resolve_ps_dxbc, f, "ps_resolve", frag, 1, 1);
        if (count_on) {
            most_cp = compute(rd_gpu_most_dxbc, sizeof rd_gpu_most_dxbc, f, "cs_most", 1, 0, 1, 16, 16);
            draws_cp = compute(rd_gpu_draws_dxbc, sizeof rd_gpu_draws_dxbc, f, "cs_draws", 0, 1, 2, 64, 1);
        }
    } else if (SDL_GetGPUShaderFormats(dev) & SDL_GPU_SHADERFORMAT_SPIRV) {
        SDL_GPUShaderFormat f = SDL_GPU_SHADERFORMAT_SPIRV;
        vs = shader(rd_gpu_vs_spirv, sizeof rd_gpu_vs_spirv, f, "vs_main", vtx, 0, 0);
        ps = exact_edges ? shader(rd_gpu_ps_exact_spirv, sizeof rd_gpu_ps_exact_spirv, f, "ps_main", frag, BINDINGS, 2)
                         : shader(rd_gpu_ps_spirv, sizeof rd_gpu_ps_spirv, f, "ps_main", frag, BINDINGS, 2);
        full_vs = shader(rd_gpu_full_vs_spirv, sizeof rd_gpu_full_vs_spirv, f, "vs_full", vtx, 0, 0);
        resolve_ps = shader(rd_gpu_resolve_ps_spirv, sizeof rd_gpu_resolve_ps_spirv, f, "ps_resolve", frag, 1, 1);
        if (count_on) {
            most_cp = compute(rd_gpu_most_spirv, sizeof rd_gpu_most_spirv, f, "cs_most", 1, 0, 1, 16, 16);
            draws_cp = compute(rd_gpu_draws_spirv, sizeof rd_gpu_draws_spirv, f, "cs_draws", 0, 1, 2, 64, 1);
        }
    } else {
        SDL_Log("rd_gpu: the device takes neither DXBC nor SPIR-V");
        return false;
    }
    memset(&si, 0, sizeof si);
    si.min_filter = si.mag_filter = SDL_GPU_FILTER_NEAREST;
    si.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    si.address_mode_u = si.address_mode_v = si.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    point = SDL_CreateGPUSampler(dev, &si);
    if (count_on) {
        SDL_GPUBufferCreateInfo bi = { SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
                                       4, 0 };
        most_buf = most_cp != NULL && draws_cp != NULL ? SDL_CreateGPUBuffer(dev, &bi) : NULL;
        if (most_buf == NULL) {
            SDL_Log("rd_gpu: every translucent layer drawn: %s", SDL_GetError());
            count_on = 0;
        }
    }
    stencil_on = count_on && SDL_GPUTextureSupportsFormat(dev, SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT, SDL_GPU_TEXTURETYPE_2D,
                                                          SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET);
    depth_format = stencil_on ? SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT : SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
    if (full_vs != NULL && resolve_ps != NULL) {
        resolve_pipe = make_resolve_pipe();
    }
    while (!make_targets(n) && n > 1) {
        n--;
    }
    want_scale = scale;
    pal_tex = make_texture(SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, 1024, 1, 1, SDL_GPU_TEXTUREUSAGE_SAMPLER);
    dummy_rgba = make_texture(SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, 1, 1, 1, SDL_GPU_TEXTUREUSAGE_SAMPLER);
    dummy_r32 = make_texture(SDL_GPU_TEXTUREFORMAT_R32_FLOAT, 1, 1, 1, SDL_GPU_TEXTUREUSAGE_SAMPLER);
    if (vs == NULL || ps == NULL || point == NULL || frame_tex[1] == NULL || depth_tex == NULL || key_q[1] == NULL ||
        key_o[1] == NULL || pal_tex == NULL || dummy_r32 == NULL) {
        SDL_Log("rd_gpu: %s", SDL_GetError());
        return false;
    }
    /* every pipeline now, as Vulkan compiles each as it is made */
    {
        int a, b;
        pipeline(P_BACKGROUND, 0, 0);
        pipeline(P_KEY, 0, 0);
        pipeline(P_PEEL, 0, 0);
        if (count_on) {
            pipeline(P_COUNT, 0, 0);
            pipeline(P_FIX, 0, 0);
        }
        if (stencil_on) {
            pipeline(P_FILL, 0, 0);
            pipeline(P_COPY, 0, 0);
            pipeline(P_COPY, 1, 0);
        }
        for (a = 0; a < 8; a++) {
            for (b = 0; b < 8; b++) {
                if (b < 2) {
                    pipeline(P_OPAQUE, a, b);
                }
                pipeline(P_PUNCH, a, b);
            }
        }
    }
    return true;
}

static SDL_GPUFence* in_flight[2];
static int next_flight;

static void wait_flight(int i) {
    if (in_flight[i] != NULL) {
        SDL_WaitForGPUFences(dev, true, &in_flight[i], 1);
        SDL_ReleaseGPUFence(dev, in_flight[i]);
        in_flight[i] = NULL;
    }
}

void rd_gpu_submit(SDL_GPUCommandBuffer* cmd) {
    wait_flight(next_flight);
    in_flight[next_flight] = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    next_flight ^= 1;
}

void rd_gpu_quit(void) {
    SDL_GPUTexture** own[] = { &frame_tex[0], &frame_tex[1], &depth_tex, &key_q[0], &key_q[1], &key_o[0], &key_o[1],
                               &pal_tex, &dummy_rgba, &dummy_r32, &out_tex, &count_tex };
    SDL_GPUBuffer** bufs[] = { &most_buf, &draws_buf, &args_buf };
    SDL_GPUGraphicsPipeline** p = &pipes[0][0][0][0];
    size_t i;
    wait_flight(0), wait_flight(1);
    for (i = 0; i < (size_t)ngtex; i++) {
        SDL_ReleaseGPUTexture(dev, gtex[i].g);
    }
    for (i = 0; i < (size_t)nspare; i++) {
        SDL_ReleaseGPUTexture(dev, spare[i].g);
    }
    free(spare);
    spare = NULL, nspare = maxspare = 0, spare_bytes = 0;
    for (i = 0; i < sizeof own / sizeof own[0]; i++) {
        release(own[i]);
    }
    if (resolve_pipe != NULL) {
        SDL_ReleaseGPUGraphicsPipeline(dev, resolve_pipe);
        resolve_pipe = NULL;
    }
    SDL_ReleaseGPUShader(dev, full_vs);
    SDL_ReleaseGPUShader(dev, resolve_ps);
    full_vs = resolve_ps = NULL;
    for (i = 0; i < sizeof bufs / sizeof bufs[0]; i++) {
        if (*bufs[i] != NULL) {
            SDL_ReleaseGPUBuffer(dev, *bufs[i]);
            *bufs[i] = NULL;
        }
    }
    if (draws_trans != NULL) {
        SDL_ReleaseGPUTransferBuffer(dev, draws_trans);
        draws_trans = NULL;
    }
    if (most_cp != NULL) {
        SDL_ReleaseGPUComputePipeline(dev, most_cp);
    }
    if (draws_cp != NULL) {
        SDL_ReleaseGPUComputePipeline(dev, draws_cp);
    }
    most_cp = draws_cp = NULL, draws_cap = args_cap = 0;
    scale = 1, frame_w = RD_WIDTH, frame_h = RD_HEIGHT, out_w = out_h = 0, tb_size = 0;
    for (i = 0; i < sizeof pipes / sizeof pipes[0][0][0][0]; i++) {
        if (p[i] != NULL) {
            SDL_ReleaseGPUGraphicsPipeline(dev, p[i]);
        }
    }
    if (vbuf != NULL) {
        SDL_ReleaseGPUBuffer(dev, vbuf);
        SDL_ReleaseGPUTransferBuffer(dev, vtrans);
    }
    if (tb != NULL) {
        SDL_ReleaseGPUTransferBuffer(dev, tb);
        tb = NULL;
    }
    if (stage != NULL) {
        SDL_ReleaseGPUTransferBuffer(dev, stage);
        stage = NULL;
    }
    free(pending);
    pending = NULL;
    npending = maxpending = 0, stage_cap = stage_used = 0;
    SDL_ReleaseGPUSampler(dev, point);
    SDL_ReleaseGPUShader(dev, vs);
    SDL_ReleaseGPUShader(dev, ps);
    free(gtex);
    free(hash);
    hash = NULL;
    for (i = 0; i < RD_LISTS; i++) {
        free(batches[i]);
        batches[i] = NULL;
        nbatches[i] = maxbatches[i] = 0;
    }
    memset(pipes, 0, sizeof pipes);
    gtex = NULL, ngtex = maxgtex = 0, vbuf = NULL, vtrans = NULL, vcap = 0, pal_serial = 0xFFFFFFFFu;
}

static int cull(const RD_STATE* s, const RD_VTX* a, const RD_VTX* b, const RD_VTX* c) {
    double area = ((double)b->x - a->x) * ((double)c->y - a->y) - ((double)b->y - a->y) * ((double)c->x - a->x);
    return area == 0.0 || !(area == area) || (s->cull != 0 && fabs(area) < 0.02) || (s->cull == 2 && area < 0) ||
           (s->cull == 3 && area > 0);
}

static void put_vertex(GVTX* g, const RD_VTX* v, const RD_VTX* col, const RD_STATE* s, float k, float order) {
    float q = v->z;
    g->x = v->x, g->y = v->y;
    g->depth = q * k, g->q = q;
    g->u = v->u * q, g->v = v->v * q;
    g->base[0] = (float)(col->base >> 24) * q, g->base[1] = (float)(col->base >> 16 & 255) * q;
    g->base[2] = (float)(col->base >> 8 & 255) * q, g->base[3] = (float)(col->base & 255) * q;
    if (s->offset) {
        g->offs[0] = (float)(col->offset >> 16 & 255) * q, g->offs[1] = (float)(col->offset >> 8 & 255) * q;
        g->offs[2] = (float)(col->offset & 255) * q;
    } else {
        g->offs[0] = g->offs[1] = g->offs[2] = 0;
    }
    g->order = order;
    memset(g->ab, 0, sizeof g->ab);
    memset(g->cf, 0, sizeof g->cf);
    memset(g->uvb, 0, sizeof g->uvb);
}

static void add_batch(int list, int first, int count, int state) {
    BATCH* b;
    if (nbatches[list] > 0 && batches[list][nbatches[list] - 1].state == state &&
        batches[list][nbatches[list] - 1].first + batches[list][nbatches[list] - 1].count == first) {
        batches[list][nbatches[list] - 1].count += count;
        return;
    }
    if (nbatches[list] == maxbatches[list]) {
        int n = maxbatches[list] ? maxbatches[list] * 2 : 1024;
        BATCH* p = realloc(batches[list], (size_t)n * sizeof *p);
        if (p == NULL) {
            return;
        }
        batches[list] = p, maxbatches[list] = n;
    }
    b = &batches[list][nbatches[list]++];
    b->first = first, b->count = count, b->state = state;
}

/* rd_soft.c's edge function */
static double edge(double ax, double ay, double bx, double by, double x, double y) {
    return (bx - ax) * (y - ay) - (by - ay) * (x - ax);
}

/* rd_soft.c's edge_owns */
static int edge_owns(const RD_VTX* a, const RD_VTX* b, double sign) {
    double dx = -((double)b->y - a->y) * sign, dy = ((double)b->x - a->x) * sign;
    return dx > 0 || (dx == 0 && dy > 0);
}

/* the exact edges' margin, in pixels: well over the GPU's 1/256 snapping */
#define EDGE_MARGIN (1.0 / 32)
/* hud arcade's: the game pixels' whose centres are in it, all of each (half a diagonal) */
#define BLOCK_MARGIN 0.75
#define MAX_CORNERS 10      /* a hexagon cut by the screen's 4 sides */
#define NEAR_CORNER 8192.0  /* corners within this of the screen's are snapped as finely */

static float min3(float a, float b, float c) {
    float m = a < b ? a : b;
    return m < c ? m : c;
}

static float max3(float a, float b, float c) {
    float m = a > b ? a : b;
    return m > c ? m : c;
}

/* the polygon cut to the side k * coord < k * at; how many corners are left */
static int clip(double* x, double* y, int n, int axis, double k, double at) {
    double ox[MAX_CORNERS], oy[MAX_CORNERS];
    int i, m = 0;
    memcpy(ox, x, n * sizeof *x), memcpy(oy, y, n * sizeof *y);
    for (i = 0; i < n; i++) {
        int j = (i + 1) % n;
        double di = k * ((axis ? oy[i] : ox[i]) - at), dj = k * ((axis ? oy[j] : ox[j]) - at);
        if (di <= 0) {
            x[m] = ox[i], y[m++] = oy[i];
        }
        if ((di < 0 && dj > 0) || (di > 0 && dj < 0)) {
            double f = di / (di - dj);
            x[m] = ox[i] + (ox[j] - ox[i]) * f, y[m++] = oy[i] + (oy[j] - oy[i]) * f;
        }
    }
    return m;
}

/* triangle t's vertices into out: g, or with the exact edges the triangle grown by d for the shader to keep
   rd_soft.c's pixels; how many */
static int put_triangle(GVTX* out, const GVTX* g, const RD_TRI* t, double d) {
    const RD_VTX *a = &t->v[0], *b = &t->v[1], *c = &t->v[2];
    double area = edge(a->x, a->y, b->x, b->y, c->x, c->y), sign = area > 0 ? 1.0 : -1.0, inv = 1.0 / area;
    double px[MAX_CORNERS], py[MAX_CORNERS], ux[3], uy[3];
    float flags = (float)(edge_owns(b, c, sign) | edge_owns(c, a, sign) << 1 | edge_owns(a, b, sign) << 2 | (area > 0) << 3);
    GVTX v[MAX_CORNERS];
    int m = 0, n = 0, i, j, k, in = 1;
    if (!exact_edges) {
        out[0] = g[0], out[1] = g[1], out[2] = g[2];
        return 3;
    }
    {
        /* grown about its centre by 1 + s: edge i moves out s area / (3 |edge i|) */
        double cx = ((double)a->x + b->x + c->x) * (1.0 / 3), cy = ((double)a->y + b->y + c->y) * (1.0 / 3);
        double l0 = fabs((double)b->x - a->x) + fabs((double)b->y - a->y), l1 = fabs((double)c->x - b->x) + fabs((double)c->y - b->y);
        double l2 = fabs((double)a->x - c->x) + fabs((double)a->y - c->y), lmax = l0 > l1 ? l0 : l1;
        double s = 3 * d * (lmax > l2 ? lmax : l2) * fabs(inv);
        if (max3(a->x, b->x, c->x) <= 0 || min3(a->x, b->x, c->x) >= RD_WIDTH || max3(a->y, b->y, c->y) <= 0 ||
            min3(a->y, b->y, c->y) >= RD_HEIGHT) {
            return 0;
        }
        for (i = 0; i < 3; i++) {
            px[i] = cx + (1 + s) * (t->v[i].x - cx), py[i] = cy + (1 + s) * (t->v[i].y - cy);
            in &= fabs(px[i]) < NEAR_CORNER && fabs(py[i]) < NEAR_CORNER;
        }
        if (s <= 0.25 && in) {
            double mid[PLANE_VALUES];
            for (k = 0; k < PLANE_VALUES; k++) {
                mid[k] = ((double)(&g[0].depth)[k] + (&g[1].depth)[k] + (&g[2].depth)[k]) * (1.0 / 3);
            }
            for (i = 0; i < 3; i++) {
                float* f;
                v[0] = g[i];
                f = &v[0].depth;
                v[0].x = (float)px[i], v[0].y = (float)py[i];
                for (k = 0; k < PLANE_VALUES; k++) {
                    f[k] = (float)(mid[k] + (1 + s) * (f[k] - mid[k]));
                }
                for (j = 0; j < 2; j++) {
                    v[0].ab[2 * j] = t->v[j].x, v[0].ab[2 * j + 1] = t->v[j].y;
                }
                v[0].cf[0] = c->x, v[0].cf[1] = c->y, v[0].cf[2] = flags, v[0].cf[3] = g[0].cf[3];
                out[i] = v[0];
            }
            return 3;
        }
        in = 1;
    }
    /* a thin triangle, or one off the screen: each edge moved out a margin */
    for (i = 0; i < 3; i++) {
        /* outward normal sign * (uy, -ux) */
        const RD_VTX *p = &t->v[i], *q = &t->v[(i + 1) % 3];
        double ex = (double)q->x - p->x, ey = (double)q->y - p->y, r = 1.0 / sqrt(ex * ex + ey * ey);
        ux[i] = ex * r, uy[i] = ey * r;
    }
    for (i = 0; i < 3; i++) {
        /* where the moved edges cross, or at a sharp corner two points a margin past it */
        int e0 = (i + 2) % 3;
        double n0x = sign * uy[e0], n0y = -sign * ux[e0], n1x = sign * uy[i], n1y = -sign * ux[i];
        double cx = t->v[i].x, cy = t->v[i].y, k1 = 1 + n0x * n1x + n0y * n1y;
        if (k1 * (64.0 * 64.0 / 2) >= 1) {
            px[m] = cx + d * (n0x + n1x) / k1, py[m++] = cy + d * (n0y + n1y) / k1;
        } else {
            px[m] = cx + d * (n0x + ux[e0]), py[m++] = cy + d * (n0y + uy[e0]);
            px[m] = cx + d * (n1x - ux[i]), py[m++] = cy + d * (n1y - uy[i]);
        }
    }
    for (i = 0; i < m; i++) {
        in &= px[i] >= -1.0 && px[i] <= RD_WIDTH + 1.0 && py[i] >= -1.0 && py[i] <= RD_HEIGHT + 1.0;
    }
    if (!in) {
        m = clip(px, py, m, 0, -1, -1.0);
        m = clip(px, py, m, 0, 1, RD_WIDTH + 1.0);
        m = clip(px, py, m, 1, -1, -1.0);
        m = clip(px, py, m, 1, 1, RD_HEIGHT + 1.0);
    }
    for (i = 0; i < m; i++) {
        double w0 = edge(b->x, b->y, c->x, c->y, px[i], py[i]) * inv, w1 = edge(c->x, c->y, a->x, a->y, px[i], py[i]) * inv;
        double w2 = edge(a->x, a->y, b->x, b->y, px[i], py[i]) * inv;
        const float* f0 = &g[0].depth;
        const float* f1 = &g[1].depth;
        const float* f2 = &g[2].depth;
        float* f;
        v[i] = g[0];
        f = &v[i].depth;
        v[i].x = (float)px[i], v[i].y = (float)py[i];
        for (k = 0; k < PLANE_VALUES; k++) {
            f[k] = (float)(w0 * f0[k] + w1 * f1[k] + w2 * f2[k]);
        }
        for (j = 0; j < 2; j++) {
            v[i].ab[2 * j] = t->v[j].x, v[i].ab[2 * j + 1] = t->v[j].y;
        }
        v[i].cf[0] = c->x, v[i].cf[1] = c->y, v[i].cf[2] = flags, v[i].cf[3] = g[0].cf[3];
    }
    for (i = 0; i + 2 < m; i++) {
        out[n++] = v[0], out[n++] = v[i + 1], out[n++] = v[i + 2];
    }
    return n;
}

/* the background plane's value f at (x, y) */
static double plane(const RD_VTX* v, const double* f, double x, double y) {
    double area = ((double)v[1].x - v[0].x) * ((double)v[2].y - v[0].y) - ((double)v[1].y - v[0].y) * ((double)v[2].x - v[0].x);
    double w1, w2;
    if (area == 0.0) {
        return f[0];
    }
    w1 = ((x - v[0].x) * ((double)v[2].y - v[0].y) - (y - v[0].y) * ((double)v[2].x - v[0].x)) / area;
    w2 = (((double)v[1].x - v[0].x) * (y - v[0].y) - ((double)v[1].y - v[0].y) * (x - v[0].x)) / area;
    return f[0] + (f[1] - f[0]) * w1 + (f[2] - f[0]) * w2;
}

static int sprite(const RD_STATE* s, const RD_TRI* t) {
    return s->tex != NULL && t->v[0].z == t->v[1].z && t->v[1].z == t->v[2].z;
}

static double texel_at(double s) {
    return floor(fabs(s - floor(s + 0.5)) < 1e-4 ? floor(s + 0.5) : s);     /* on a texel's edge: the one after it */
}

/* the texels a sprite's game pixels show at 640 by 480, across or down: lo and hi from the pixel centres c0 and c1 of
   its first and last, u0 + du (c - x0) the texture there, n its texels */
static void shown(double u0, double du, double x0, double c0, double c1, int n, double* lo, double* hi) {
    double a = texel_at((u0 + du * (c0 - x0)) * n), b = texel_at((u0 + du * (c1 - x0)) * n);
    *lo = (du > 0 ? a : a + 1) / n;
    *hi = (du > 0 ? b + 1 : b) / n;
}

/* a sprite drawn larger, upright, a texel a game pixel or smaller: b the bounds of the texels it shows at 640 by
   480, and across or down at a texel a game pixel, its u or v moved so those texels fill it from edge to edge, each
   on its game pixel (shrunk, its texels stay where the game put them, in line with the sprites drawn with it); 0 for
   a magnified one, drawn as at 640 by 480 */
static int fit_texels(const RD_STATE* s, const RD_TRI* o, RD_TRI* t, float* b) {
    const RD_VTX *a = &o->v[0], *p = &o->v[1], *c = &o->v[2];
    double area = edge(a->x, a->y, p->x, p->y, c->x, c->y);
    double dux, duy, dvx, dvy, x0, x1, y0, y1, u0, u1, v0, v1;
    int w = s->tex->width, h = s->tex->height, j;
    if (area == 0) {
        return 0;
    }
    dux = ((p->u - a->u) * ((double)c->y - a->y) - (c->u - a->u) * ((double)p->y - a->y)) / area;
    duy = ((c->u - a->u) * ((double)p->x - a->x) - (p->u - a->u) * ((double)c->x - a->x)) / area;
    dvx = ((p->v - a->v) * ((double)c->y - a->y) - (c->v - a->v) * ((double)p->y - a->y)) / area;
    dvy = ((c->v - a->v) * ((double)p->x - a->x) - (p->v - a->v) * ((double)c->x - a->x)) / area;
    if (dux == 0 || dvy == 0 || fabs(duy * w) > 1e-4 || fabs(dvx * h) > 1e-4) {
        /* turned or slanted: its texels where they fall, between its own u and v */
        b[0] = fminf(a->u, fminf(p->u, c->u)), b[1] = fminf(a->v, fminf(p->v, c->v));
        b[2] = fmaxf(a->u, fmaxf(p->u, c->u)), b[3] = fmaxf(a->v, fmaxf(p->v, c->v));
        return 1;
    }
    if (fmin(fabs(dux * w), fabs(dvy * h)) < 1 - 1e-3) {
        return 0;   /* magnified: as at 640 by 480, so sprites drawn over each other stay in line */
    }
    x0 = x1 = t->v[0].x, y0 = y1 = t->v[0].y;
    for (j = 0; j < 3; j++) {
        if (t->v[j].x != floorf(t->v[j].x) || t->v[j].y != floorf(t->v[j].y)) {
            return 0;
        }
        x0 = fmin(x0, t->v[j].x), x1 = fmax(x1, t->v[j].x), y0 = fmin(y0, t->v[j].y), y1 = fmax(y1, t->v[j].y);
    }
    if (x1 <= x0 || y1 <= y0) {
        return 0;
    }
    shown(a->u, dux, a->x, x0 + 0.5, x1 - 0.5, w, &u0, &u1);
    shown(a->v, dvy, a->y, y0 + 0.5, y1 - 0.5, h, &v0, &v1);
    for (j = 0; j < 3; j++) {
        if (fabs(fabs(dux * w) - 1) < 1e-3) {
            t->v[j].u = (float)(u0 + (u1 - u0) * (t->v[j].x - x0) / (x1 - x0));
        }
        if (fabs(fabs(dvy * h) - 1) < 1e-3) {
            t->v[j].v = (float)(v0 + (v1 - v0) * (t->v[j].y - y0) / (y1 - y0));
        }
    }
    b[0] = (float)fmin(u0, u1), b[1] = (float)fmin(v0, v1), b[2] = (float)fmax(u0, u1), b[3] = (float)fmax(v0, v1);
    return 1;
}

static float snap16(float x) {
    return fabsf(x) < 65536.0f ? floorf(x * 16.0f + 0.5f) * (1.0f / 16.0f) : x;
}

/* the sprites' corners, on a sixteenth of a game pixel: an open-addressed set, ~0 empty */
static uint64_t* corners;
static uint32_t corners_size;

static uint64_t corner_key(float x, float y) {
    uint32_t a, b;
    memcpy(&a, &x, 4), memcpy(&b, &y, 4);
    return (uint64_t)a << 32 | b;
}

static uint64_t* corner_slot(uint64_t k) {
    uint32_t i = (uint32_t)((k * 0x9E3779B97F4A7C15ull) >> 32) & (corners_size - 1);
    while (corners[i] != ~0ull && corners[i] != k) {
        i = (i + 1) & (corners_size - 1);
    }
    return &corners[i];
}

/* an upright sprite drawn at the game's own size, a texel 5/3 by 15/7 game pixels (a character's): its texels can
   be put on the lattice of pixels their edges fall on */
static int on_lattice(const RD_STATE* s, const RD_TRI* o) {
    const RD_VTX *a = &o->v[0], *p = &o->v[1], *c = &o->v[2];
    double area = edge(a->x, a->y, p->x, p->y, c->x, c->y), dux, duy, dvx, dvy;
    if (area == 0) {
        return 0;
    }
    dux = ((p->u - a->u) * ((double)c->y - a->y) - (c->u - a->u) * ((double)p->y - a->y)) / area * s->tex->width;
    duy = ((c->u - a->u) * ((double)p->x - a->x) - (p->u - a->u) * ((double)c->x - a->x)) / area * s->tex->width;
    dvx = ((p->v - a->v) * ((double)c->y - a->y) - (c->v - a->v) * ((double)p->y - a->y)) / area * s->tex->height;
    dvy = ((c->v - a->v) * ((double)p->x - a->x) - (p->v - a->v) * ((double)c->x - a->x)) / area * s->tex->height;
    return fabs(duy) < 1e-4 && fabs(dvx) < 1e-4 && fabs(fabs(dux) - 0.6) < 1e-4 && fabs(fabs(dvy) - 7.0 / 15) < 1e-4;
}

/* texel: the sprites' edges, across (0) and down (1), sorted, for meet() */
typedef struct {
    float e, lo, hi;    /* the edge, and the sprite's extent the other way */
    uint8_t side;       /* 0 its least, 1 its greatest */
    uint8_t lattice;    /* the sprite is on the lattice (else its corners go to whole game pixels) */
} EDGE;
static EDGE* edges[2];
static int nedges, edges_size;

static int edge_order(const void* a, const void* b) {
    float x = ((const EDGE*)a)->e, y = ((const EDGE*)b)->e;
    return x < y ? -1 : x > y;
}

static void find_corners(void) {
    uint32_t want = 64, count = 0;
    int list, i, j;
    for (list = 0; list < RD_LISTS; list++) {
        count += 3 * (uint32_t)rd_frame->ntris[list];
    }
    while (want < 2 * count) {
        want *= 2;
    }
    if (want > corners_size) {
        free(corners);
        corners = malloc(want * sizeof *corners);
        corners_size = corners != NULL ? want : 0;
    }
    if (corners == NULL) {
        return;
    }
    memset(corners, 0xFF, corners_size * sizeof *corners);
    nedges = 0;
    if (sprite_mode == RD_SPRITES_TEXEL && (int)count > edges_size) {
        free(edges[0]), free(edges[1]);
        edges[0] = malloc(count * sizeof(EDGE)), edges[1] = malloc(count * sizeof(EDGE));
        edges_size = edges[0] != NULL && edges[1] != NULL ? (int)count : 0;
    }
    for (list = 0; list < RD_LISTS; list++) {
        for (i = 0; i < rd_frame->ntris[list]; i++) {
            const RD_TRI* t = &rd_frame->tris[list][i];
            const RD_STATE* s = &rd_frame->states[t->state];
            if (sprite(s, t) && !cull(s, &t->v[0], &t->v[1], &t->v[2])) {
                for (j = 0; j < 3; j++) {
                    uint64_t k = corner_key(snap16(t->v[j].x), snap16(t->v[j].y));
                    *corner_slot(k) = k;
                }
                if (sprite_mode == RD_SPRITES_TEXEL && nedges + 2 <= edges_size) {
                    int axis, lattice = on_lattice(s, t);
                    for (axis = 0; axis < 2; axis++) {
                        float lo = 1e30f, hi = -1e30f, olo = 1e30f, ohi = -1e30f;
                        for (j = 0; j < 3; j++) {
                            float c = axis ? t->v[j].y : t->v[j].x, d = axis ? t->v[j].x : t->v[j].y;
                            lo = fminf(lo, c), hi = fmaxf(hi, c), olo = fminf(olo, d), ohi = fmaxf(ohi, d);
                        }
                        edges[axis][nedges] = (EDGE){ lo, olo, ohi, 0, (uint8_t)lattice };
                        edges[axis][nedges + 1] = (EDGE){ hi, olo, ohi, 1, (uint8_t)lattice };
                    }
                    nedges += 2;
                }
            }
        }
    }
    if (nedges > 0) {
        qsort(edges[0], nedges, sizeof(EDGE), edge_order), qsort(edges[1], nedges, sizeof(EDGE), edge_order);
    }
}

static int is_corner(float x, float y) {
    return corners != NULL && *corner_slot(corner_key(x, y)) != ~0ull;
}

/* texel: a sprite on the lattice keeps every edge on its texels' lattice (at k pixels a game pixel), so its texels all
   stay whole and in place with the rest of its object, but for one case. Where its edge e across (0) or down (1), its
   least (side 0) or greatest, lo to hi the other way, meets no edge of another on the lattice (it ends its object
   there) and another sprite's opposite edge, less than a game pixel off, leaves a gap where at 640 by 480 none shows
   (they overlap, or end at the same game pixel's edge), it reaches over the gap to where that one is drawn. Its edge l
   on the lattice, moved or not. Shared: on the CPS2's pixels, a gap is at most one of them, and only the sprite that
   draws the game pixel holding its centre at 640 by 480 reaches over it */
static float meet(int axis, int side, float e, float l, float lo, float hi, float k, int shared) {
    const EDGE* d = edges[axis];
    float to = l, c = side ? l + 0.5f / k : l - 0.5f / k;
    int a = 0, b = nedges;
    if (shared && (side ? floorf(c) + 0.5f >= e : floorf(c) + 0.5f < e)) {
        return l;
    }
    while (a < b) {
        int m = (a + b) / 2;
        if (d[m].e < e - 1.0f) {
            a = m + 1;
        } else {
            b = m;
        }
    }
    for (; a < nedges && d[a].e < e + 1.0f; a++) {
        if (d[a].side != side && d[a].lo < hi && d[a].hi > lo) {
            float m;
            if (d[a].lattice && fabsf(d[a].e - e) <= 1.0f / 64) {
                return l;
            }
            if ((side ? d[a].e > e : d[a].e < e) && floorf(d[a].e + 0.5f) != floorf(e + 0.5f)) {
                continue;
            }
            m = !d[a].lattice ? floorf(d[a].e + 0.5f) : shared ? ceilf(d[a].e * k - 0.5f) / k
                                                          : floorf(d[a].e * k + (0.5f - 1.0f / 64)) / k;
            if (side ? m > l && (to == l || m < to) : m < l && (to == l || m > to)) {
                to = m;
            }
        }
    }
    return to;
}

static int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

/* The translucent list drawn a triangle at a time far to near, blended by the GPU, in an order that keeps each
   pixel's fragments as peeling sorts them (by 1/w, then the list's order). Triangles whose 1/w cross where they
   overlap, and those caught between them, are peeled together (batches[RD_TRANS]); the rest are drawn before or
   after them */
typedef struct {
    int first, count, state;    /* its vertices */
    float x0, y0, x1, y1;       /* its box on the screen, in the game's pixels */
    float qlo, qhi;             /* its 1/w */
    double pa, pb, pc;          /* 1/w = pa x + pb y + pc */
    double vx[3], vy[3];
    int block;                  /* the front in the game's pixels: whole game pixels, past its edges */
    int adds;                   /* added to the frame, by a factor of its own: in any order with others that add */
} TREC;

typedef struct {
    int* v;
    int n, max;
} IVEC;

#define BIN 16
#define BINS_X (RD_WIDTH / BIN)
#define BINS_Y (RD_HEIGHT / BIN)

static TREC* trec;
static GVTX* tverts;        /* the translucent list's vertices, to be put in order */
static int ntverts, maxtverts;
static int ntrec, maxtrec, sort_failed;
/* a run of the translucent list: sorted batches, or a group of batches[RD_TRANS] peeled within box */
typedef struct {
    int peel, from, to;
    SDL_Rect box;           /* in the frame's pixels */
} STEP;
static STEP* steps;
static int nsteps, maxsteps;
static BATCH* sorted;
static int nsorted, maxsorted;
static const BATCH* peel_b; /* the group being peeled */
static int npeel_b;
static SDL_Rect peel_box;
static IVEC flats, bin_at, bin_list, apart_at, apart_list, seq, eu, ev, out_at, out, mark, low, stk, call, heap, comp, size, grp_at, grp, level, unit, tmp;

static void ipush(IVEC* a, int x) {
    if (a->n == a->max) {
        int m = a->max ? a->max * 2 : 1024;
        int* p = realloc(a->v, (size_t)m * sizeof *p);
        if (p == NULL) {
            sort_failed = 1;
            return;
        }
        a->v = p, a->max = m;
    }
    a->v[a->n++] = x;
}

/* a holding n */
static int isize(IVEC* a, int n) {
    if (n > a->max) {
        int* p = realloc(a->v, (size_t)n * sizeof *p);
        if (p == NULL) {
            sort_failed = 1;
            a->n = 0;
            return 0;
        }
        a->v = p, a->max = n;
    }
    a->n = n;
    return 1;
}

/* joined to the last batch where it can be, from batch from on */
static void push_batch(BATCH** list, int* n, int* max, int from, int first, int count, int state) {
    BATCH* b;
    if (*n > from && (*list)[*n - 1].state == state && (*list)[*n - 1].first + (*list)[*n - 1].count == first) {
        (*list)[*n - 1].count += count;
        return;
    }
    if (*n == *max) {
        int m = *max ? *max * 2 : 1024;
        BATCH* p = realloc(*list, (size_t)m * sizeof *p);
        if (p == NULL) {
            sort_failed = 1;
            return;
        }
        *list = p, *max = m;
    }
    b = &(*list)[(*n)++];
    b->first = first, b->count = count, b->state = state;
}

static void add_record(const RD_TRI* t, int first, int count, int block) {
    const RD_VTX *a = &t->v[0], *b = &t->v[1], *c = &t->v[2];
    double area = edge(a->x, a->y, b->x, b->y, c->x, c->y);
    double d1 = (double)b->z - a->z, d2 = (double)c->z - a->z;
    TREC* r;
    int j;
    if (ntrec == maxtrec) {
        int m = maxtrec ? maxtrec * 2 : 4096;
        TREC* p = realloc(trec, (size_t)m * sizeof *p);
        if (p == NULL) {
            sort_failed = 1;
            return;
        }
        trec = p, maxtrec = m;
    }
    r = &trec[ntrec++];
    r->first = first, r->count = count, r->state = t->state, r->block = block;
    r->adds = rd_frame->states[t->state].dst == RD_ONE && (rd_frame->states[t->state].src == RD_ZERO ||
              rd_frame->states[t->state].src == RD_ONE || rd_frame->states[t->state].src == RD_SRC_ALPHA ||
              rd_frame->states[t->state].src == RD_INV_SRC_ALPHA);
    r->x0 = fmaxf(min3(a->x, b->x, c->x) - block, 0), r->x1 = fminf(max3(a->x, b->x, c->x) + block, RD_WIDTH);
    r->y0 = fmaxf(min3(a->y, b->y, c->y) - block, 0), r->y1 = fminf(max3(a->y, b->y, c->y) + block, RD_HEIGHT);
    r->qlo = min3(a->z, b->z, c->z), r->qhi = max3(a->z, b->z, c->z);
    r->pa = (d1 * ((double)c->y - a->y) - d2 * ((double)b->y - a->y)) / area;
    r->pb = (d2 * ((double)b->x - a->x) - d1 * ((double)c->x - a->x)) / area;
    r->pc = a->z - r->pa * a->x - r->pb * a->y;
    for (j = 0; j < 3; j++) {
        r->vx[j] = t->v[j].x, r->vy[j] = t->v[j].y;
    }
}

/* a's triangle cut to b's: its corners in x and y, or 0 where they share no area */
static int overlap(const TREC* a, const TREC* b, double* x, double* y) {
    double ox[12], oy[12], area = 0;
    double sign = edge(b->vx[0], b->vy[0], b->vx[1], b->vy[1], b->vx[2], b->vy[2]) > 0 ? 1 : -1;
    int n = 3, i, e;
    for (i = 0; i < 3; i++) {
        x[i] = a->vx[i], y[i] = a->vy[i];
    }
    for (e = 0; e < 3 && n >= 3; e++) {
        double ax = b->vx[e], ay = b->vy[e], bx = b->vx[(e + 1) % 3], by = b->vy[(e + 1) % 3];
        int m = 0;
        memcpy(ox, x, (size_t)n * sizeof *x), memcpy(oy, y, (size_t)n * sizeof *y);
        for (i = 0; i < n; i++) {
            int j = (i + 1) % n;
            double di = sign * edge(ax, ay, bx, by, ox[i], oy[i]), dj = sign * edge(ax, ay, bx, by, ox[j], oy[j]);
            if (di >= 0) {
                x[m] = ox[i], y[m++] = oy[i];
            }
            if ((di < 0 && dj > 0) || (di > 0 && dj < 0)) {
                double f = di / (di - dj);
                x[m] = ox[i] + (ox[j] - ox[i]) * f, y[m++] = oy[i] + (oy[j] - oy[i]) * f;
            }
        }
        n = m;
    }
    for (i = 0; i < n; i++) {
        area += x[i] * y[(i + 1) % n] - x[(i + 1) % n] * y[i];
    }
    return n >= 3 && fabs(area) >= 1e-6 ? n : 0;
}

/* where a and b overlap: 1 a nearer at every pixel, -1 b nearer, 2 neither, 0 they don't overlap */
static int compare(const TREC* a, const TREC* b) {
    double tol = 1e-5 * fmax(fabs(a->qhi), fabs(b->qhi)), x[12], y[12];
    int i, n, near = 0, far = 0;
    if ((a->adds && b->adds) || !(a->x0 < b->x1 && b->x0 < a->x1 && a->y0 < b->y1 && b->y0 < a->y1)) {
        return 0;
    }
    if (a->qlo > b->qhi + tol) {
        return 1;
    }
    if (b->qlo > a->qhi + tol) {
        return -1;
    }
    if (a->block || b->block) {
        return 2;
    }
    n = overlap(a, b, x, y);
    for (i = 0; i < n; i++) {
        double d = (a->pa - b->pa) * x[i] + (a->pb - b->pb) * y[i] + (a->pc - b->pc);
        near += d > tol, far += d < -tol;
    }
    return n == 0 ? 0 : near == n ? 1 : far == n ? -1 : 2;
}

static int flat_order(const void* pa, const void* pb) {
    int a = *(const int*)pa, b = *(const int*)pb;
    return trec[a].qlo < trec[b].qlo ? -1 : trec[a].qlo > trec[b].qlo ? 1 : a - b;
}

static int flat(int r) {
    return trec[r].qlo == trec[r].qhi;
}

static void bins_of(const TREC* r, int* bx0, int* by0, int* bx1, int* by1) {
    int x0 = (int)(r->x0 / BIN), x1 = (int)(r->x1 / BIN), y0 = (int)(r->y0 / BIN), y1 = (int)(r->y1 / BIN);
    *bx0 = x0 < 0 ? 0 : x0 >= BINS_X ? BINS_X - 1 : x0;
    *bx1 = x1 < *bx0 ? *bx0 : x1 >= BINS_X ? BINS_X - 1 : x1;
    *by0 = y0 < 0 ? 0 : y0 >= BINS_Y ? BINS_Y - 1 : y0;
    *by1 = y1 < *by0 ? *by0 : y1 >= BINS_Y ? BINS_Y - 1 : y1;
}

/* the edges from -> to as lists: node i's from at[i] */
static void adjacency(IVEC* at, IVEC* list, const IVEC* from, const IVEC* to, int m) {
    int i;
    if (!isize(at, m + 1) || !isize(list, from->n)) {
        return;
    }
    memset(at->v, 0, (size_t)(m + 1) * sizeof *at->v);
    for (i = 0; i < from->n; i++) {
        at->v[from->v[i] + 1]++;
    }
    for (i = 0; i < m; i++) {
        at->v[i + 1] += at->v[i];
    }
    for (i = 0; i < from->n; i++) {
        list->v[at->v[from->v[i]]++] = to->v[i];
    }
    for (i = m; i > 0; i--) {
        at->v[i] = at->v[i - 1];
    }
    at->v[0] = 0;
}

/* the records in each bin, from at.v[bin]: the flat ones far to near, then the sloped ones; with apart, only those
   that don't add to the frame */
static int bin_up(IVEC* at, IVEC* list, int apart) {
    int nbins = BINS_X * BINS_Y, i, j, b, bx0, by0, bx1, by1, pass;
    if (!isize(at, nbins + 1) || !isize(&tmp, nbins)) {
        return 0;
    }
    memset(at->v, 0, (size_t)(nbins + 1) * sizeof *at->v);
    for (pass = 0; pass < 2; pass++) {
        if (pass == 1) {
            for (b = 0; b < nbins; b++) {
                at->v[b + 1] += at->v[b];
            }
            if (!isize(list, at->v[nbins])) {
                return 0;
            }
            memcpy(tmp.v, at->v, (size_t)nbins * sizeof *tmp.v);
        }
        for (i = 0; i < flats.n + ntrec; i++) {
            int r = i < flats.n ? flats.v[i] : i - flats.n;
            if ((i >= flats.n && flat(r)) || (apart && trec[r].adds)) {
                continue;
            }
            bins_of(&trec[r], &bx0, &by0, &bx1, &by1);
            for (b = by0; b <= by1; b++) {
                for (j = bx0; j <= bx1; j++) {
                    if (pass == 0) {
                        at->v[b * BINS_X + j + 1]++;
                    } else {
                        list->v[tmp.v[b * BINS_X + j]++] = r;
                    }
                }
            }
        }
    }
    return 1;
}

/* each pair that overlaps in a bin ordered: the flat ones far to near (1/w, then the list's order), and each sloped
   one against every other it shares a bin with (that doesn't add, if it adds), both ways where neither is nearer
   throughout */
static int pair_up(void) {
    int m = ntrec, nbins = BINS_X * BINS_Y, i, j, b, k, bx0, by0, bx1, by1;
    eu.n = ev.n = flats.n = 0;
    for (i = 0; i < m; i++) {
        if (flat(i)) {
            ipush(&flats, i);
        }
    }
    if (sort_failed || !isize(&mark, m)) {
        return 0;
    }
    qsort(flats.v, (size_t)flats.n, sizeof *flats.v, flat_order);
    if (!bin_up(&bin_at, &bin_list, 0) || !bin_up(&apart_at, &apart_list, 1)) {
        return 0;
    }
    for (b = 0; b < nbins; b++) {
        for (k = bin_at.v[b]; k + 1 < bin_at.v[b + 1] && flat(bin_list.v[k + 1]); k++) {
            ipush(&eu, bin_list.v[k]), ipush(&ev, bin_list.v[k + 1]);
        }
    }
    for (i = 0; i < m; i++) {
        mark.v[i] = -1;
    }
    for (i = 0; i < m; i++) {
        const IVEC *at = trec[i].adds ? &apart_at : &bin_at, *list = trec[i].adds ? &apart_list : &bin_list;
        if (flat(i)) {
            continue;
        }
        bins_of(&trec[i], &bx0, &by0, &bx1, &by1);
        for (b = by0; b <= by1; b++) {
            for (j = bx0; j <= bx1; j++) {
                for (k = at->v[b * BINS_X + j]; k < at->v[b * BINS_X + j + 1]; k++) {
                    int o = list->v[k], c;
                    if (o == i || (!flat(o) && o < i) || mark.v[o] == i) {
                        continue;
                    }
                    mark.v[o] = i;
                    c = compare(&trec[i], &trec[o]);
                    if (c == 1 || c == 2) {
                        ipush(&eu, o), ipush(&ev, i);
                    }
                    if (c == -1 || c == 2) {
                        ipush(&eu, i), ipush(&ev, o);
                    }
                }
            }
        }
    }
    if (sort_failed) {
        return 0;
    }
    adjacency(&out_at, &out, &eu, &ev, m);
    return !sort_failed;
}

/* the strongly connected records (Tarjan's): comp.v[i], numbered as found, so one reached from another has the lower
   number; size.v its records. How many */
static int components(void) {
    int m = ntrec, i, n = 0, nc = 0;
    if (!isize(&low, m) || !isize(&comp, m) || !isize(&size, 0)) {
        return 0;
    }
    for (i = 0; i < m; i++) {
        mark.v[i] = -1;     /* its visit */
    }
    stk.n = 0;
    for (i = 0; i < m && !sort_failed; i++) {
        if (mark.v[i] >= 0) {
            continue;
        }
        call.n = 0;
        mark.v[i] = low.v[i] = n++;
        ipush(&stk, i), ipush(&call, i), ipush(&call, out_at.v[i]);
        while (call.n > 0 && !sort_failed) {
            int v = call.v[call.n - 2], e = call.v[call.n - 1];
            if (e < out_at.v[v + 1]) {
                int w = out.v[e];
                call.v[call.n - 1]++;
                if (mark.v[w] < 0) {
                    mark.v[w] = low.v[w] = n++;
                    ipush(&stk, w), ipush(&call, w), ipush(&call, out_at.v[w]);
                } else if (low.v[w] >= 0 && mark.v[w] < low.v[v]) {
                    low.v[v] = mark.v[w];   /* still on the stack */
                }
                continue;
            }
            call.n -= 2;
            if (low.v[v] == mark.v[v]) {
                int k = 0, w;
                do {
                    w = stk.v[--stk.n];
                    low.v[w] = -1;          /* done */
                    comp.v[w] = nc;
                    k++;
                } while (w != v);
                ipush(&size, k);
                nc++;
            } else if (low.v[v] < low.v[call.v[call.n - 2]]) {
                low.v[call.v[call.n - 2]] = low.v[v];
            }
        }
    }
    return sort_failed ? 0 : nc;
}

/* members.v from at.v[k]: the records whose key.v is k, in the list's order */
static void members(IVEC* at, IVEC* list, const IVEC* key, int nk) {
    int i;
    if (!isize(at, nk + 1) || !isize(list, key->n)) {
        return;
    }
    memset(at->v, 0, (size_t)(nk + 1) * sizeof *at->v);
    for (i = 0; i < key->n; i++) {
        if (key->v[i] >= 0) {
            at->v[key->v[i] + 1]++;
        }
    }
    for (i = 0; i < nk; i++) {
        at->v[i + 1] += at->v[i];
    }
    for (i = 0; i < key->n; i++) {
        if (key->v[i] >= 0) {
            list->v[at->v[key->v[i]]++] = i;
        }
    }
    for (i = nk; i > 0; i--) {
        at->v[i] = at->v[i - 1];
    }
    at->v[0] = 0;
}

/* each record's unit.v: itself, or m + its group: the tangled components that no path joins are peeled together,
   numbered by the tangled ones on the longest path to them. How many groups */
static int groups(int nc) {
    int m = ntrec, c, i, e, ng = 0;
    members(&grp_at, &grp, &comp, nc);
    if (sort_failed || !isize(&level, nc) || !isize(&unit, m)) {
        return 0;
    }
    memset(level.v, 0, (size_t)nc * sizeof *level.v);
    for (c = nc - 1; c >= 0; c--) {
        for (i = grp_at.v[c]; i < grp_at.v[c + 1]; i++) {
            int v = grp.v[i];
            for (e = out_at.v[v]; e < out_at.v[v + 1]; e++) {
                int cw = comp.v[out.v[e]], l = level.v[c] + (size.v[c] > 1);
                if (cw != c && l > level.v[cw]) {
                    level.v[cw] = l;
                }
            }
        }
    }
    for (i = 0; i < m; i++) {
        c = comp.v[i];
        unit.v[i] = size.v[c] > 1 ? m + level.v[c] : i;
        if (size.v[c] > 1 && level.v[c] + 1 > ng) {
            ng = level.v[c] + 1;
        }
    }
    return ng;
}

/* the heap's key: a record's place in the list; a group's, its first record's */
static int heap_key(int u) {
    return u < ntrec ? u : grp.v[grp_at.v[u - ntrec]];
}

static void heap_push(int u) {
    int i;
    ipush(&heap, u);
    for (i = heap.n - 1; i > 0 && heap_key(heap.v[(i - 1) / 2]) > heap_key(heap.v[i]); i = (i - 1) / 2) {
        int t = heap.v[i];
        heap.v[i] = heap.v[(i - 1) / 2], heap.v[(i - 1) / 2] = t;
    }
}

static int heap_pop(void) {
    int top = heap.v[0], i = 0;
    heap.v[0] = heap.v[--heap.n];
    for (;;) {
        int l = 2 * i + 1, r = l + 1, s = i, t;
        if (l < heap.n && heap_key(heap.v[l]) < heap_key(heap.v[s])) {
            s = l;
        }
        if (r < heap.n && heap_key(heap.v[r]) < heap_key(heap.v[s])) {
            s = r;
        }
        if (s == i) {
            return top;
        }
        t = heap.v[i], heap.v[i] = heap.v[s], heap.v[s] = t, i = s;
    }
}

static void add_step(int peel, int from, int to, const SDL_Rect* box) {
    if (nsteps > 0 && !peel && !steps[nsteps - 1].peel) {
        steps[nsteps - 1].to = to;
        return;
    }
    if (nsteps == maxsteps) {
        int m = maxsteps ? maxsteps * 2 : 64;
        STEP* p = realloc(steps, (size_t)m * sizeof *p);
        if (p == NULL) {
            sort_failed = 1;
            return;
        }
        steps = p, maxsteps = m;
    }
    steps[nsteps].peel = peel, steps[nsteps].from = from, steps[nsteps].to = to;
    steps[nsteps++].box = *box;
}

static void copy_record(const TREC* r, int* n, BATCH** list, int* nb, int* max, int from) {
    memcpy(&verts[*n], &tverts[r->first], (size_t)r->count * sizeof(GVTX));
    push_batch(list, nb, max, from, *n, r->count, r->state);
    *n += r->count;
}

/* the units in order (Kahn's, the list's own where free): seq.v, a record or m + a group. Whether all */
static int order_units(void) {
    int m = ntrec, nc, ng, i, e, u, units = 0;
    if (m == 0 || !pair_up() || (nc = components()) == 0) {
        return 0;
    }
    ng = groups(nc);
    /* the groups' records, and how many units wait on each */
    if (sort_failed || !isize(&tmp, m)) {
        return 0;
    }
    for (i = 0; i < m; i++) {
        tmp.v[i] = unit.v[i] >= m ? unit.v[i] - m : -1;
    }
    members(&grp_at, &grp, &tmp, ng);
    if (sort_failed || !isize(&low, m + ng) || !isize(&seq, 0)) {
        return 0;
    }
    memset(low.v, 0, (size_t)(m + ng) * sizeof *low.v);
    for (i = 0; i < eu.n; i++) {
        int a = unit.v[eu.v[i]], b = unit.v[ev.v[i]];
        low.v[b] += a != b;
    }
    heap.n = 0;
    for (u = 0; u < m + ng; u++) {
        int real = u < m ? unit.v[u] == u : grp_at.v[u - m + 1] > grp_at.v[u - m];
        units += real;
        if (real && low.v[u] == 0) {
            heap_push(u);
        }
    }
    while (heap.n > 0 && !sort_failed) {
        int g = heap_pop(), k, from = g < m ? 0 : grp_at.v[g - m], to = g < m ? 1 : grp_at.v[g - m + 1];
        ipush(&seq, g);
        for (k = from; k < to; k++) {
            int v = g < m ? g : grp.v[k];
            for (e = out_at.v[v]; e < out_at.v[v + 1]; e++) {
                int w = unit.v[out.v[e]];
                if (w != g && --low.v[w] == 0) {
                    heap_push(w);
                }
            }
        }
    }
    return !sort_failed && seq.n == units;
}

/* a box in the frame's pixels around game pixels x0..x1, y0..y1, a game pixel more each side for the front's whole
   game pixels */
static SDL_Rect frame_box(float x0, float y0, float x1, float y1) {
    float sx = (float)frame_w / RD_WIDTH, sy = (float)frame_h / RD_HEIGHT;
    int px0 = (int)floorf((x0 - 1) * sx), py0 = (int)floorf((y0 - 1) * sy);
    int px1 = (int)ceilf((x1 + 1) * sx), py1 = (int)ceilf((y1 + 1) * sy);
    SDL_Rect box;
    px0 = px0 < 0 ? 0 : px0, py0 = py0 < 0 ? 0 : py0;
    px1 = px1 > frame_w ? frame_w : px1, py1 = py1 > frame_h ? frame_h : py1;
    box.x = px0, box.y = py0, box.w = px1 > px0 ? px1 - px0 : 0, box.h = py1 > py0 ? py1 - py0 : 0;
    return box;
}

static int peel_layers(void) {
    return layers_env ? layers_env : rd_q.nosort ? 4 : 12;
}

/* records gathered for a peel: their box, and the most of them over a bin (the layers it likely needs) */
typedef struct {
    float x0, y0, x1, y1;
    int most;
    int n[BINS_X * BINS_Y];
} GATHER;
static GATHER run_g, unit_g;

static void gather_clear(GATHER* g) {
    g->x0 = RD_WIDTH, g->y0 = RD_HEIGHT, g->x1 = g->y1 = 0, g->most = 0;
    memset(g->n, 0, sizeof g->n);
}

static void gather(GATHER* g, int u) {
    int m = ntrec, k, x, y, bx0, by0, bx1, by1;
    int from = u < m ? 0 : grp_at.v[u - m], to = u < m ? 1 : grp_at.v[u - m + 1];
    for (k = from; k < to; k++) {
        const TREC* r = &trec[u < m ? u : grp.v[k]];
        g->x0 = fminf(g->x0, r->x0), g->y0 = fminf(g->y0, r->y0), g->x1 = fmaxf(g->x1, r->x1), g->y1 = fmaxf(g->y1, r->y1);
        bins_of(r, &bx0, &by0, &bx1, &by1);
        for (y = by0; y <= by1; y++) {
            for (x = bx0; x <= bx1; x++) {
                int c = ++g->n[y * BINS_X + x];
                g->most = c > g->most ? c : g->most;
            }
        }
    }
}

/* about what peeling them costs, a layer of the screen's pixels a unit; each peel's own passes about two */
static double gather_cost(const GATHER* g) {
    double area = g->x1 > g->x0 && g->y1 > g->y0 ? ((double)g->x1 - g->x0) * ((double)g->y1 - g->y0) : 0;
    int layers = peel_layers();
    return area / (RD_WIDTH * RD_HEIGHT) * (g->most < layers ? g->most : layers) + 2;
}

/* the translucent list's vertices from n, in order: steps of sorted batches and of groups to peel (batches[RD_TRANS]);
   all peeled if it can't be ordered. The order from one group to the next is peeled as one where that costs less */
static int sort_trans(int n) {
    int m = ntrec, i, j, k;
    nsteps = nsorted = nbatches[RD_TRANS] = 0;
    if (sort_failed || !order_units()) {
        for (i = 0; i < m; i++) {
            copy_record(&trec[i], &n, &batches[RD_TRANS], &nbatches[RD_TRANS], &maxbatches[RD_TRANS], 0);
        }
        return n;
    }
    for (i = 0; i < seq.n;) {
        int end, first;
        double cost;
        SDL_Rect box;
        if (seq.v[i] < m) {
            SDL_Rect none = { 0, 0, 0, 0 };
            first = nsorted;
            copy_record(&trec[seq.v[i]], &n, &sorted, &nsorted, &maxsorted, 0);
            add_step(0, first, nsorted, &none);
            i++;
            continue;
        }
        gather_clear(&run_g);
        gather(&run_g, seq.v[i]);
        cost = gather_cost(&run_g);
        for (end = j = i + 1; j < seq.n; j++) {
            GATHER* g = &unit_g;
            double merged;
            if (seq.v[j] < m) {
                continue;
            }
            /* the run and the units up to this group, against the two apart */
            gather_clear(g);
            gather(g, seq.v[j]);
            merged = gather_cost(g);
            memcpy(g, &run_g, sizeof *g);
            for (k = end; k <= j; k++) {
                gather(g, seq.v[k]);
            }
            if (gather_cost(g) > cost + merged) {
                break;
            }
            memcpy(&run_g, g, sizeof *g);
            cost = gather_cost(g), end = j + 1;
        }
        first = nbatches[RD_TRANS];
        for (; i < end; i++) {
            int g = seq.v[i], from = g < m ? 0 : grp_at.v[g - m], to = g < m ? 1 : grp_at.v[g - m + 1];
            for (k = from; k < to; k++) {
                copy_record(&trec[g < m ? g : grp.v[k]], &n, &batches[RD_TRANS], &nbatches[RD_TRANS],
                            &maxbatches[RD_TRANS], first);
            }
        }
        box = frame_box(run_g.x0, run_g.y0, run_g.x1, run_g.y1);
        add_step(1, first, nbatches[RD_TRANS], &box);
    }
    return n;
}

/* the background's 6 vertices, then each list's triangles; how many */
static int build(float* zscale) {
    int list, i, n = 6, lattice_x, lattice_y;
    float qmax = 0;
    int e;
    ntrec = ntverts = nsteps = nsorted = sort_failed = 0;
    for (list = 0; list < RD_LISTS; list++) {
        for (i = 0; i < rd_frame->ntris[list]; i++) {
            const RD_TRI* t = &rd_frame->tris[list][i];
            int j;
            for (j = 0; j < 3; j++) {
                if (t->v[j].z > qmax && isfinite(t->v[j].z)) {
                    qmax = t->v[j].z;
                }
            }
        }
    }
    for (i = 0; i < 3; i++) {
        if (rd_frame->background[i].z > qmax && isfinite(rd_frame->background[i].z)) {
            qmax = rd_frame->background[i].z;
        }
    }
    frexpf(qmax > 0 ? qmax : 1.0f, &e);
    *zscale = ldexpf(1.0f, -e);          /* a power of two: 1/w scaled exactly below 1 */
    for (list = 0; list < RD_LISTS; list++) {
        n += rd_frame->ntris[list] * (exact_edges ? 3 * (MAX_CORNERS - 2) : 3);
    }
    sorting = sort_on;
    if (sorting && rd_frame->ntris[RD_TRANS] * (exact_edges ? 3 * (MAX_CORNERS - 2) : 3) > maxtverts) {
        int m = rd_frame->ntris[RD_TRANS] * (exact_edges ? 3 * (MAX_CORNERS - 2) : 3) * 2;
        GVTX* p = realloc(tverts, (size_t)m * sizeof *p);
        if (p != NULL) {
            tverts = p, maxtverts = m;
        }
        sorting = p != NULL;
    }
    if ((uint32_t)n > vcap) {
        SDL_GPUBufferCreateInfo bi = { SDL_GPU_BUFFERUSAGE_VERTEX, (Uint32)(n * sizeof(GVTX)), 0 };
        SDL_GPUTransferBufferCreateInfo ti = { SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, (Uint32)(n * sizeof(GVTX)), 0 };
        if (vbuf != NULL) {
            SDL_ReleaseGPUBuffer(dev, vbuf);
            SDL_ReleaseGPUTransferBuffer(dev, vtrans);
        }
        vbuf = SDL_CreateGPUBuffer(dev, &bi);
        vtrans = SDL_CreateGPUTransferBuffer(dev, &ti);
        vcap = (uint32_t)n;
    }
    verts = SDL_MapGPUTransferBuffer(dev, vtrans, true);
    {
        /* the background: 1/w = 1, so it is not divided */
        static const float cx[6] = { 0, RD_WIDTH, 0, RD_WIDTH, RD_WIDTH, 0 }, cy[6] = { 0, 0, RD_HEIGHT, 0, RD_HEIGHT, RD_HEIGHT };
        const RD_VTX* b = rd_frame->background;
        double f[3];
        int c, j;
        for (j = 0; j < 6; j++) {
            GVTX g;
            memset(&g, 0, sizeof g);
            g.x = cx[j], g.y = cy[j], g.q = 1.0f;
            g.cf[2] = 16;       /* every pixel */
            for (i = 0; i < 3; i++) {
                f[i] = b[i].z;
            }
            g.depth = (float)plane(b, f, cx[j], cy[j]) * *zscale;
            for (c = 0; c < 4; c++) {
                for (i = 0; i < 3; i++) {
                    f[i] = (double)(b[i].base >> (24 - 8 * c) & 255);
                }
                g.base[c] = (float)plane(b, f, cx[j], cy[j]);
            }
            verts[j] = g;
        }
    }
    n = 6;
    if (scale > 1) {
        find_corners();
    }
    /* a character's texel, 5/3 by 15/7 game pixels, is frame_h/288 by frame_h/224 pixels: their fractions' parts */
    lattice_x = 288 / gcd(frame_h, 288), lattice_y = 224 / gcd(frame_h, 224);
    for (list = 0; list < RD_LISTS; list++) {
        nbatches[list] = 0;
        for (i = 0; i < rd_frame->ntris[list]; i++) {
            const RD_TRI *t = &rd_frame->tris[list][i], *o = t;
            const RD_STATE* s = &rd_frame->states[t->state];
            RD_TRI snapped;
            GVTX g[3];
            float fb[4];
            int j, k, fitted = 0, lattice = 0;
            /* hud smooth or arcade: only sprites read as sprites; arcade: the front in the game's pixels */
            int as_sprite = sprite(s, o) && (hud_mode == RD_HUD_SHARP || s->origin == RD_SPRITE);
            int blocks = hud_mode == RD_HUD_ARCADE && s->origin == RD_FRONT && exact_edges;
            float move_x = 0, move_y = 0, lx0, hx, ly0, hy;
            if (cull(s, &t->v[0], &t->v[1], &t->v[2])) {
                continue;
            }
            if (scale > 1 && !blocks) {
                /* drawn larger: corners on a sixteenth of a game pixel, so neighbours a float's rounding apart meet,
                   and sprites' corners (and the corners they share) on whole ones, so sprites side by side meet as
                   they do at 640 by 480 (texel: on whole pixels drawn) */
                snapped = *t;
                if (sprite_mode == RD_SPRITES_TEXEL && as_sprite) {
                    lattice = on_lattice(s, o);
                }
                lx0 = fminf(t->v[0].x, fminf(t->v[1].x, t->v[2].x)), hx = fmaxf(t->v[0].x, fmaxf(t->v[1].x, t->v[2].x));
                ly0 = fminf(t->v[0].y, fminf(t->v[1].y, t->v[2].y)), hy = fmaxf(t->v[0].y, fmaxf(t->v[1].y, t->v[2].y));
                for (j = 0; j < 3; j++) {
                    float x = snap16(t->v[j].x), y = snap16(t->v[j].y);
                    if (is_corner(x, y)) {
                        if (lattice && shared_grid) {
                            /* on the CPS2's pixels: a corner at the grid line whose pixel centres it takes (the
                               first after it), as the art drawn one pixel a texel would be */
                            float lx = ceilf(t->v[j].x * 0.6f - 0.5f) / 0.6f;
                            float ly = ceilf(t->v[j].y * (7.0f / 15) - 0.5f) / (7.0f / 15);
                            if (j == 0) {
                                move_x = lx - t->v[0].x, move_y = ly - t->v[0].y;
                            }
                            x = meet(0, t->v[j].x == hx, t->v[j].x, lx, ly0, hy, 0.6f, 1);
                            y = meet(1, t->v[j].y == hy, t->v[j].y, ly, lx0, hx, 7.0f / 15, 1);
                        } else if (lattice) {
                            /* its texels from where the game put them onto the pixels their corners can fall on
                               (whole ones at 2688 by 2016, sevenths down at 1920 by 1440): the sprite's texels all
                               move as far, and its edges stay on their lattice but where it reaches over a gap (meet) */
                            float kx = scale * lattice_x, ky = scale * lattice_y;
                            float lx = floorf(t->v[j].x * kx + (0.5f - 1.0f / 64)) / kx;
                            float ly = floorf(t->v[j].y * ky + (0.5f - 1.0f / 64)) / ky;
                            if (j == 0) {
                                move_x = lx - t->v[0].x, move_y = ly - t->v[0].y;
                            }
                            x = meet(0, t->v[j].x == hx, t->v[j].x, lx, ly0, hy, kx, 0);
                            y = meet(1, t->v[j].y == hy, t->v[j].y, ly, lx0, hx, ky, 0);
                        } else {
                            x = floorf(x + 0.5f), y = floorf(y + 0.5f);
                        }
                    }
                    snapped.v[j].x = x, snapped.v[j].y = y;
                }
                {
                    /* the game's planes kept: 1/w, and u and v over w, at the moved corners from the triangle as it was
                       (a sprite on the lattice: moved as far as its texels) */
                    const RD_VTX *a = &o->v[0], *b = &o->v[1], *c = &o->v[2];
                    double inv = 1.0 / edge(a->x, a->y, b->x, b->y, c->x, c->y);
                    int level = a->z == b->z && b->z == c->z;
                    for (j = 0; j < 3; j++) {
                        RD_VTX* v = &snapped.v[j];
                        double px = (double)v->x - move_x, py = (double)v->y - move_y;
                        double w0 = edge(b->x, b->y, c->x, c->y, px, py) * inv, w1 = edge(c->x, c->y, a->x, a->y, px, py) * inv;
                        double w2 = 1.0 - w0 - w1;
                        double q = level ? a->z : w0 * a->z + w1 * b->z + w2 * c->z;
                        double uq = w0 * a->u * a->z + w1 * b->u * b->z + w2 * c->u * c->z;
                        double vq = w0 * a->v * a->z + w1 * b->v * b->z + w2 * c->v * c->z;
                        if (q > 0 && (px != o->v[j].x || py != o->v[j].y)) {
                            v->z = (float)q, v->u = (float)(uq / q), v->v = (float)(vq / q);
                        }
                    }
                }
                fitted = as_sprite && fit_texels(s, o, &snapped, fb);
                t = &snapped;
                if (cull(s, &t->v[0], &t->v[1], &t->v[2])) {
                    continue;
                }
            }
            for (j = 0; j < 3; j++) {
                put_vertex(&g[j], &t->v[j], s->flat ? &t->v[2] : &t->v[j], s, *zscale, (float)i);
            }
            /* a sprite: textured, its 1/w the same at each corner. Half a quad has the quad's u and v, from and to.
               cf.w 2: its texels fitted to it; 3: on the lattice; 4: the front in the game's pixels */
            if (blocks) {
                for (j = 0; j < 3; j++) {
                    g[j].cf[3] = 4.0f;
                }
            } else if (as_sprite) {
                float b[4];
                b[0] = fminf(o->v[0].u, fminf(o->v[1].u, o->v[2].u)), b[1] = fminf(o->v[0].v, fminf(o->v[1].v, o->v[2].v));
                b[2] = fmaxf(o->v[0].u, fmaxf(o->v[1].u, o->v[2].u)), b[3] = fmaxf(o->v[0].v, fmaxf(o->v[1].v, o->v[2].v));
                for (j = 0; j < 3; j++) {
                    g[j].cf[3] = lattice ? 3.0f : fitted ? 2.0f : 1.0f;
                    memcpy(g[j].uvb, fitted ? fb : b, sizeof b);
                }
            }
            if (list == RD_TRANS && sorting) {
                k = put_triangle(&tverts[ntverts], g, t, blocks ? BLOCK_MARGIN : EDGE_MARGIN);
                if (k > 0) {
                    add_record(t, ntverts, k, blocks);
                    ntverts += k;
                }
                continue;
            }
            k = put_triangle(&verts[n], g, t, blocks ? BLOCK_MARGIN : EDGE_MARGIN);
            if (k > 0) {
                add_batch(list, n, k, t->state);
                n += k;
            }
        }
    }
    if (sorting) {
        n = sort_trans(n);
    }
    if (nsteps == 0 && nbatches[RD_TRANS] > 0) {
        SDL_Rect all = { 0, 0, frame_w, frame_h };
        add_step(1, 0, nbatches[RD_TRANS], &all);
    }
    SDL_UnmapGPUTransferBuffer(dev, vtrans);
    return n;
}

static void draw_values(DRAW* d, const RD_STATE* s, int list, int mode) {
    static const float bias[16] = { 0, -4, -2, -1 };
    memset(d, 0, sizeof *d);
    d->mode = mode;
    d->flags = (s->use_alpha ? F_USE_ALPHA : 0) | (s->offset ? F_OFFSET : 0) | (list == RD_PUNCH ? F_PUNCH : 0);
    if (s->fog == RD_FOG_TABLE && fog_on) {
        d->flags |= F_FOG_TABLE;
    } else if (s->fog == RD_FOG_TABLE2) {
        d->flags |= F_FOG_TABLE2;
    }
    if (s->tex != NULL) {
        d->flags |= F_TEX | (s->bilinear ? F_BILINEAR : 0) | (s->tex_alpha ? F_TEX_ALPHA : 0) | (uint32_t)s->shading * F_SHADING |
                    (uint32_t)(s->uv_mode & 15) * F_UV;
        d->flags |= s->tex->format == RD_PAL4 || s->tex->format == RD_PAL8 ? F_PAL : 0;
        d->width = s->tex->width, d->height = s->tex->height, d->levels = s->tex->levels;
        d->palette = s->palette;
        d->mip_bias = bias[s->mip_d & 15];
    }
    d->blend = s->src | s->dst << 4;
}

static SDL_GPUTexture* bound_tex[BINDINGS];

/* the texture, the palette and the peeling's (t2-t6), when they change */
static void bind_textures(SDL_GPURenderPass* rp, SDL_GPUTexture* tex, SDL_GPUTexture* const* keys) {
    SDL_GPUTextureSamplerBinding tb[BINDINGS];
    int j, same = 1;
    for (j = 0; j < BINDINGS; j++) {
        tb[j].sampler = point;
        tb[j].texture = j == 0 || j == 6 ? dummy_rgba : dummy_r32;
        if (j >= 2 && j < 7 && keys != NULL && keys[j - 2] != NULL) {
            tb[j].texture = keys[j - 2];
        }
    }
    if (tex != NULL) {
        tb[0].texture = tex;
    }
    tb[1].texture = pal_tex;
    for (j = 0; j < BINDINGS; j++) {
        same &= tb[j].texture == bound_tex[j];
        bound_tex[j] = tb[j].texture;
    }
    if (!same) {
        SDL_BindGPUFragmentSamplers(rp, 0, tb, BINDINGS);
    }
}

/* the translucent list in one draw: a layer's keys, the last layer's pixels or the count; row: its draw in args_buf,
   or -1 */
static void draw_whole(SDL_GPUCommandBuffer* cmd, SDL_GPURenderPass* rp, int kind, int mode,
                       SDL_GPUTexture* const* keys, int row) {
    const BATCH* first = &peel_b[0];
    const BATCH* last = &peel_b[npeel_b - 1];
    SDL_GPUGraphicsPipeline* p = pipeline(kind, 0, 0);
    DRAW d;
    if (p == NULL) {
        return;
    }
    SDL_BindGPUGraphicsPipeline(rp, p);
    bind_textures(rp, NULL, keys);
    memset(&d, 0, sizeof d);
    d.mode = mode;
    SDL_PushGPUFragmentUniformData(cmd, 0, &d, sizeof d);
    if (row >= 0) {
        SDL_DrawGPUPrimitivesIndirect(rp, args_buf, (Uint32)row * 16u, 1);
    } else {
        SDL_DrawGPUPrimitives(rp, (Uint32)(last->first + last->count - first->first), 1, (Uint32)first->first, 0);
    }
}

/* row: the first batch's draw in args_buf, or -1 */
static void draw_batches(SDL_GPUCommandBuffer* cmd, SDL_GPURenderPass* rp, const BATCH* batch, int nb, int list, int kind,
                         int mode, SDL_GPUTexture* const* keys, int row) {
    int i;
    SDL_GPUGraphicsPipeline* bound = NULL;
    for (i = 0; i < nb; i++) {
        const BATCH* b = &batch[i];
        const RD_STATE* s = &rd_frame->states[b->state];
        SDL_GPUGraphicsPipeline* p;
        DRAW d;
        if (kind == P_OPAQUE) {
            p = pipeline(P_OPAQUE, s->depth, s->zwrite);
        } else {
            int blended = kind == P_PUNCH || kind == P_SORTED;
            p = pipeline(kind, blended ? s->src : 0, blended ? s->dst : 0);
        }
        if (p == NULL) {
            continue;
        }
        if (p != bound) {
            SDL_BindGPUGraphicsPipeline(rp, p);
            bound = p;
        }
        bind_textures(rp, state_tex[b->state], keys);
        draw_values(&d, s, list, mode);
        SDL_PushGPUFragmentUniformData(cmd, 0, &d, sizeof d);
        if (row >= 0) {
            SDL_DrawGPUPrimitivesIndirect(rp, args_buf, (Uint32)(row + i) * 16u, 1);
        } else {
            SDL_DrawGPUPrimitives(rp, (Uint32)b->count, 1, (Uint32)b->first, 0);
        }
    }
}

static void draw_list(SDL_GPUCommandBuffer* cmd, SDL_GPURenderPass* rp, int list, int kind, int mode,
                      SDL_GPUTexture* const* keys, int row) {
    draw_batches(cmd, rp, batches[list], nbatches[list], list, kind, mode, keys, row);
}

/* room for a frame of nb translucent batches over up to layers layers; 0 when there is none */
static int count_room(int nb, int layers) {
    uint32_t d = (uint32_t)nb + 1, a = (uint32_t)layers * (uint32_t)(nb + 4);
    if (d > draws_cap) {
        SDL_GPUBufferCreateInfo bi = { SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ, 0, 0 };
        SDL_GPUTransferBufferCreateInfo ti = { SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, 0, 0 };
        if (draws_buf != NULL) {
            SDL_ReleaseGPUBuffer(dev, draws_buf);
            SDL_ReleaseGPUTransferBuffer(dev, draws_trans);
        }
        draws_cap = d * 2;
        bi.size = draws_cap * 8, ti.size = 8 + draws_cap * 8;
        draws_buf = SDL_CreateGPUBuffer(dev, &bi);
        draws_trans = SDL_CreateGPUTransferBuffer(dev, &ti);
        if (draws_buf == NULL || draws_trans == NULL) {
            draws_cap = 0;
            return 0;
        }
    }
    if (a > args_cap) {
        SDL_GPUBufferCreateInfo bi = { SDL_GPU_BUFFERUSAGE_INDIRECT | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE, 0, 0 };
        if (args_buf != NULL) {
            SDL_ReleaseGPUBuffer(dev, args_buf);
        }
        args_cap = a * 2;
        bi.size = args_cap * 16;
        args_buf = SDL_CreateGPUBuffer(dev, &bi);
        if (args_buf == NULL) {
            args_cap = 0;
            return 0;
        }
    }
    return 1;
}

/* the frame's translucent draws for the counts: the most fragments' slot zeroed, then the keys' draw and the
   batches' */
static void count_upload(SDL_GPUCopyPass* copy) {
    const BATCH* b = peel_b;
    int i, nb = npeel_b;
    uint32_t* d = SDL_MapGPUTransferBuffer(dev, draws_trans, true);
    d[0] = d[1] = 0;
    d[2] = (uint32_t)b[0].first, d[3] = (uint32_t)(b[nb - 1].first + b[nb - 1].count - b[0].first);
    for (i = 0; i < nb; i++) {
        d[4 + 2 * i] = (uint32_t)b[i].first, d[5 + 2 * i] = (uint32_t)b[i].count;
    }
    SDL_UnmapGPUTransferBuffer(dev, draws_trans);
    {
        SDL_GPUTransferBufferLocation src = { draws_trans, 0 };
        SDL_GPUBufferRegion dst = { most_buf, 0, 4 };
        SDL_UploadToGPUBuffer(copy, &src, &dst, true);
    }
    {
        SDL_GPUTransferBufferLocation src = { draws_trans, 8 };
        SDL_GPUBufferRegion dst = { draws_buf, 0, (Uint32)(nb + 1) * 8u };
        SDL_UploadToGPUBuffer(copy, &src, &dst, true);
    }
}

/* each pixel's fragments counted, the most found, and each layer's draws made: entries a layer */
static void count_layers(SDL_GPUCommandBuffer* cmd, SDL_GPURenderPass* rp, int layers, int entries) {
    uint32_t u[4] = { (uint32_t)frame_w, (uint32_t)frame_h, (uint32_t)layers, (uint32_t)entries };
    SDL_GPUStorageBufferReadWriteBinding rw[2];
    SDL_GPUTextureSamplerBinding tb;
    SDL_GPUComputePass* cp;
    draw_whole(cmd, rp, P_COUNT, 5, NULL, -1);
    SDL_EndGPURenderPass(rp);
    memset(rw, 0, sizeof rw);
    rw[0].buffer = most_buf;
    cp = SDL_BeginGPUComputePass(cmd, NULL, 0, rw, 1);
    tb.texture = count_tex, tb.sampler = point;
    SDL_BindGPUComputePipeline(cp, most_cp);
    SDL_BindGPUComputeSamplers(cp, 0, &tb, 1);
    SDL_PushGPUComputeUniformData(cmd, 0, u, sizeof u);
    SDL_DispatchGPUCompute(cp, (Uint32)(frame_w + 15) / 16u, (Uint32)(frame_h + 15) / 16u, 1);
    SDL_EndGPUComputePass(cp);
    rw[0].buffer = args_buf, rw[0].cycle = true;
    rw[1].buffer = most_buf;
    cp = SDL_BeginGPUComputePass(cmd, NULL, 0, rw, 2);
    SDL_BindGPUComputePipeline(cp, draws_cp);
    SDL_BindGPUComputeStorageBuffers(cp, 0, &draws_buf, 1);
    SDL_PushGPUComputeUniformData(cmd, 0, u, sizeof u);
    SDL_DispatchGPUCompute(cp, 1, 1, 1);
    SDL_EndGPUComputePass(cp);
}

static int scissor_on;      /* the passes cut to peel_box */

enum { CLEAR_DEPTH = 1, CLEAR_STENCIL = 2 };

static SDL_GPURenderPass* begin(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* color, int clear, float value, int clears) {
    SDL_GPUColorTargetInfo ct;
    SDL_GPUDepthStencilTargetInfo dt;
    SDL_GPUBufferBinding vb;
    SDL_GPURenderPass* rp;
    memset(&ct, 0, sizeof ct);
    ct.texture = color;
    ct.load_op = clear ? SDL_GPU_LOADOP_CLEAR : SDL_GPU_LOADOP_LOAD;
    ct.store_op = SDL_GPU_STOREOP_STORE;
    ct.clear_color.r = ct.clear_color.g = ct.clear_color.b = ct.clear_color.a = value;
    memset(&dt, 0, sizeof dt);
    dt.texture = depth_tex;
    dt.load_op = clears & CLEAR_DEPTH ? SDL_GPU_LOADOP_CLEAR : SDL_GPU_LOADOP_LOAD;
    dt.store_op = SDL_GPU_STOREOP_STORE;
    dt.stencil_load_op = !stencil_on ? SDL_GPU_LOADOP_DONT_CARE : clears & CLEAR_STENCIL ? SDL_GPU_LOADOP_CLEAR : SDL_GPU_LOADOP_LOAD;
    dt.stencil_store_op = stencil_on ? SDL_GPU_STOREOP_STORE : SDL_GPU_STOREOP_DONT_CARE;
    dt.clear_depth = 0.0f;
    dt.clear_stencil = (Uint8)stencil_clear;
    rp = SDL_BeginGPURenderPass(cmd, &ct, 1, &dt);
    if (scissor_on) {
        SDL_SetGPUScissor(rp, &peel_box);
    }
    memset(bound_tex, 0, sizeof bound_tex);
    vb.buffer = vbuf;
    vb.offset = 0;
    SDL_BindGPUVertexBuffers(rp, 0, &vb, 1);
    return rp;
}

/* the screen drawn where the stencil's count is over the layer (equal: is it): the key target set to value, or
   from (P_COPY) copied; row: its draw in args_buf, or -1 */
static void fill(SDL_GPUCommandBuffer* cmd, SDL_GPURenderPass* rp, int layer, int equal, float value, SDL_GPUTexture* from,
                 int row) {
    SDL_GPUGraphicsPipeline* p = from != NULL ? pipeline(P_COPY, equal, 0) : pipeline(P_FILL, 0, 0);
    SDL_GPUTexture* keys[5] = { NULL, NULL, NULL, NULL, from };
    DRAW d;
    if (p == NULL) {
        return;
    }
    SDL_BindGPUGraphicsPipeline(rp, p);
    SDL_SetGPUStencilReference(rp, (Uint8)layer);
    bind_textures(rp, NULL, keys);
    memset(&d, 0, sizeof d);
    d.mode = from != NULL ? 7 : 6;
    d.mip_bias = value;
    SDL_PushGPUFragmentUniformData(cmd, 0, &d, sizeof d);
    if (row >= 0) {
        SDL_DrawGPUPrimitivesIndirect(rp, args_buf, (Uint32)row * 16u, 1);
    } else {
        SDL_DrawGPUPrimitives(rp, 6, 1, 0, 0);
    }
}

/* the frame, filtered to out_tex when the window is smaller */
static SDL_GPUTexture* finish(SDL_GPUCommandBuffer* cmd) {
    SDL_GPUColorTargetInfo ct;
    SDL_GPUTextureSamplerBinding tb;
    SDL_GPURenderPass* rp;
    float ratio[4];
    if (out_tex == NULL) {
        return frame_tex[0];
    }
    memset(&ct, 0, sizeof ct);
    ct.texture = out_tex;
    ct.load_op = SDL_GPU_LOADOP_DONT_CARE;
    ct.store_op = SDL_GPU_STOREOP_STORE;
    rp = SDL_BeginGPURenderPass(cmd, &ct, 1, NULL);
    if (rp == NULL) {
        return frame_tex[0];
    }
    ratio[0] = (float)frame_w / (float)out_w, ratio[1] = (float)frame_h / (float)out_h;
    ratio[2] = ratio[3] = 0;
    tb.texture = frame_tex[0], tb.sampler = point;
    SDL_BindGPUGraphicsPipeline(rp, resolve_pipe);
    SDL_BindGPUFragmentSamplers(rp, 0, &tb, 1);
    SDL_PushGPUFragmentUniformData(cmd, 0, ratio, sizeof ratio);
    SDL_DrawGPUPrimitives(rp, 3, 1, 0, 0);
    SDL_EndGPURenderPass(rp);
    return out_tex;
}

/* a group of the translucent list (peel_b) peeled a layer at a time over the frame, far to near, within its box */
static void peel(SDL_GPUCommandBuffer* cmd, const STEP* st, int layers) {
    SDL_GPUCopyPass* copy;
    SDL_GPURenderPass* rp;
    int i, layer, f = 0, nb = st->to - st->from, entries = nb + 4;
    int counting = count_on && count_tex != NULL && count_room(nb, layers), fills = counting && stencil_on;
    peel_b = batches[RD_TRANS] + st->from, npeel_b = nb;
    peel_box = st->box;
    scissor_on = 1;
    /* not counted: every pixel passes */
    stencil_clear = counting ? 0 : 255;
    if (counting) {
        copy = SDL_BeginGPUCopyPass(cmd);
        count_upload(copy);
        SDL_EndGPUCopyPass(copy);
    } else if (stencil_on) {
        SDL_EndGPURenderPass(begin(cmd, frame_tex[0], 0, 0.0f, CLEAR_STENCIL));
    }
    if (!fills) {
        SDL_GPUColorTargetInfo ct[2];
        memset(ct, 0, sizeof ct);
        for (i = 0; i < 2; i++) {
            ct[i].texture = i == 0 ? key_q[1] : key_o[1];
            ct[i].load_op = SDL_GPU_LOADOP_CLEAR;
            ct[i].store_op = SDL_GPU_STOREOP_STORE;
            ct[i].clear_color.r = -3.0e38f;
        }
        SDL_EndGPURenderPass(SDL_BeginGPURenderPass(cmd, ct, 2, NULL));
    }
    if (counting) {
        SDL_GPUTextureLocation from, to;
        count_layers(cmd, begin(cmd, count_tex, 1, 0.0f, CLEAR_STENCIL), layers, entries);
        /* the frame copied once (where a layer draws): from then on each layer copies only the last layer's pixels */
        if (fills) {
            rp = begin(cmd, frame_tex[1], 0, 0.0f, 0);
            fill(cmd, rp, 0, 0, 0.0f, frame_tex[0], -1);
            SDL_EndGPURenderPass(rp);
        } else {
            copy = SDL_BeginGPUCopyPass(cmd);
            memset(&from, 0, sizeof from);
            memset(&to, 0, sizeof to);
            from.texture = frame_tex[0], to.texture = frame_tex[1];
            SDL_CopyGPUTextureToTexture(copy, &from, &to, (Uint32)frame_w, (Uint32)frame_h, 1, false);
            SDL_EndGPUCopyPass(copy);
        }
    }
    if (fills) {
        for (i = 0; i < 2; i++) {
            rp = begin(cmd, i == 0 ? key_q[1] : key_o[1], 0, 0.0f, 0);
            fill(cmd, rp, 0, 0, -3.0e38f, NULL, -1);
            SDL_EndGPURenderPass(rp);
        }
    }
    for (layer = 0; layer < layers; layer++) {
        int cur = layer & 1, last = cur ^ 1;
        SDL_GPUTexture* keys[5] = { NULL };
        /* the nearest 1/w past the last layer */
        keys[0] = key_q[last], keys[1] = key_o[last];
        rp = begin(cmd, key_q[cur], !fills, 3.0e38f, 0);
        if (fills) {
            fill(cmd, rp, layer, 0, 3.0e38f, NULL, layer * entries + 2);
        }
        draw_whole(cmd, rp, P_KEY, 1, keys, counting ? layer * entries : -1);
        SDL_EndGPURenderPass(rp);
        /* the first triangle at it */
        keys[2] = key_q[cur];
        rp = begin(cmd, key_o[cur], !fills, 3.0e38f, 0);
        if (fills) {
            fill(cmd, rp, layer, 0, 3.0e38f, NULL, layer * entries + 2);
        }
        draw_whole(cmd, rp, P_KEY, 2, keys, counting ? layer * entries : -1);
        SDL_EndGPURenderPass(rp);
        /* its fragment, blended by the shader over a copy of the frame */
        if (!counting) {
            SDL_GPUTextureLocation from, to;
            copy = SDL_BeginGPUCopyPass(cmd);
            memset(&from, 0, sizeof from);
            memset(&to, 0, sizeof to);
            from.texture = frame_tex[f], to.texture = frame_tex[f ^ 1];
            SDL_CopyGPUTextureToTexture(copy, &from, &to, (Uint32)frame_w, (Uint32)frame_h, 1, false);
            SDL_EndGPUCopyPass(copy);
        }
        keys[0] = keys[1] = NULL, keys[3] = key_o[cur], keys[4] = frame_tex[f];
        f ^= 1;
        rp = begin(cmd, frame_tex[f], 0, 0.0f, 0);
        if (fills && layer > 0) {
            /* the pixels the last layer was the last to change */
            fill(cmd, rp, layer, 1, 0.0f, frame_tex[f ^ 1], layer * entries + 3);
        } else if (counting && layer > 0) {
            /* the copy's pixels the last layer changed */
            SDL_GPUTexture* last_keys[5] = { key_q[last], key_o[last], NULL, NULL, frame_tex[f ^ 1] };
            draw_whole(cmd, rp, P_FIX, 4, last_keys, layer * entries + 1);
        }
        if (fills) {
            SDL_SetGPUStencilReference(rp, (Uint8)layer);
        }
        draw_batches(cmd, rp, peel_b, npeel_b, RD_TRANS, P_PEEL, 3, keys, counting ? layer * entries + 4 : -1);
        SDL_EndGPURenderPass(rp);
    }
    if (f != 0 && fills) {
        rp = begin(cmd, frame_tex[0], 0, 0.0f, 0);
        fill(cmd, rp, 0, 0, 0.0f, frame_tex[1], -1);
        SDL_EndGPURenderPass(rp);
    } else if (f != 0) {
        /* always the same texture, for the caller */
        SDL_GPUTextureLocation from, to;
        copy = SDL_BeginGPUCopyPass(cmd);
        memset(&from, 0, sizeof from);
        memset(&to, 0, sizeof to);
        from.texture = frame_tex[1], to.texture = frame_tex[0];
        SDL_CopyGPUTextureToTexture(copy, &from, &to, (Uint32)frame_w, (Uint32)frame_h, 1, false);
        SDL_EndGPUCopyPass(copy);
    }
    scissor_on = 0;
}

SDL_GPUTexture* rd_gpu_end_frame(SDL_GPUCommandBuffer* cmd) {
    SDL_GPUCopyPass* copy;
    SDL_GPURenderPass* rp;
    FRAMEU fu;
    float zscale;
    int n, i;
    /* the translucent layers peeled */
    int layers = peel_layers();
    frame_no++;
    if (frame_no % 60 == 0) {
        evict();
    }
    fit_view();
    n = build(&zscale);
    for (i = 0; i < rd_frame->nstates; i++) {
        const RD_TEXTURE* t = rd_frame->states[i].tex;
        state_tex[i] = t == NULL ? NULL : i > 0 && rd_frame->states[i - 1].tex == t ? state_tex[i - 1] : texture(t);
    }
    copy = SDL_BeginGPUCopyPass(cmd);
    {
        SDL_GPUTransferBufferLocation src = { vtrans, 0 };
        SDL_GPUBufferRegion dst = { vbuf, 0, (Uint32)(n * sizeof(GVTX)) };
        SDL_UploadToGPUBuffer(copy, &src, &dst, true);
    }
    fog_on = 0;
    for (i = 0; i < 128; i++) {
        fog_on |= rd_fog.table[i] != 0.0f;
    }
    upload_pending(copy, rd_palette_serial != pal_serial ? rd_palette : NULL);
    pal_serial = rd_palette_serial;
    SDL_EndGPUCopyPass(copy);

    memset(&fu, 0, sizeof fu);
    memcpy(fu.fog_table, rd_fog.table, sizeof fu.fog_table);
    fu.fog_color[0] = (float)(rd_fog.color >> 24), fu.fog_color[1] = (float)(rd_fog.color >> 16 & 255);
    fu.fog_color[2] = (float)(rd_fog.color >> 8 & 255), fu.fog_color[3] = (float)(rd_fog.color & 255);
    fu.fog_density = rd_fog.density;
    fu.inv_scale = 1.0f / scale;
    fu.sprite_mode = sprite_mode;
    SDL_PushGPUFragmentUniformData(cmd, 1, &fu, sizeof fu);

    stencil_clear = 0;
    rp = begin(cmd, frame_tex[0], 1, 0.0f, CLEAR_DEPTH | CLEAR_STENCIL);
    {
        SDL_GPUGraphicsPipeline* p = pipeline(P_BACKGROUND, 0, 0);
        DRAW d;
        memset(&d, 0, sizeof d);
        d.flags = F_USE_ALPHA;
        if (p != NULL) {
            SDL_BindGPUGraphicsPipeline(rp, p);
            bind_textures(rp, NULL, NULL);
            SDL_PushGPUFragmentUniformData(cmd, 0, &d, sizeof d);
            SDL_DrawGPUPrimitives(rp, 6, 1, 0, 0);
        }
    }
    draw_list(cmd, rp, RD_OPAQUE, P_OPAQUE, 0, NULL, -1);
    draw_list(cmd, rp, RD_PUNCH, P_PUNCH, 0, NULL, -1);
    /* the translucent list: sorted, blended as drawn, and peeled */
    for (i = 0; i < nsteps; i++) {
        if (!steps[i].peel) {
            if (rp == NULL) {
                rp = begin(cmd, frame_tex[0], 0, 0.0f, 0);
            }
            draw_batches(cmd, rp, sorted + steps[i].from, steps[i].to - steps[i].from, RD_TRANS, P_SORTED, 0, NULL, -1);
            continue;
        }
        if (rp != NULL) {
            SDL_EndGPURenderPass(rp);
            rp = NULL;
        }
        peel(cmd, &steps[i], layers);
    }
    if (rp != NULL) {
        SDL_EndGPURenderPass(rp);
    }
    return finish(cmd);
}

/* t's w by h pixels, RGBA, mapped from the transfer buffer: waits for the GPU. NULL when it fails */
static const uint8_t* download(SDL_GPUTexture* t, int w, int h) {
    SDL_GPUCommandBuffer* cmd;
    SDL_GPUCopyPass* copy;
    SDL_GPUTextureRegion src;
    SDL_GPUTextureTransferInfo dst;
    SDL_GPUFence* fence;
    uint32_t size = (uint32_t)(w * h * 4);
    if (tb != NULL && tb_size < size) {
        SDL_ReleaseGPUTransferBuffer(dev, tb);
        tb = NULL;
    }
    if (tb == NULL) {
        SDL_GPUTransferBufferCreateInfo ti = { SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, size, 0 };
        tb = SDL_CreateGPUTransferBuffer(dev, &ti);
        tb_size = tb != NULL ? size : 0;
    }
    cmd = tb != NULL ? SDL_AcquireGPUCommandBuffer(dev) : NULL;
    if (cmd == NULL) {
        return NULL;
    }
    copy = SDL_BeginGPUCopyPass(cmd);
    memset(&src, 0, sizeof src);
    src.texture = t;
    src.w = (Uint32)w, src.h = (Uint32)h, src.d = 1;
    memset(&dst, 0, sizeof dst);
    dst.transfer_buffer = tb;
    SDL_DownloadFromGPUTexture(copy, &src, &dst);
    SDL_EndGPUCopyPass(copy);
    fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    if (fence == NULL) {
        return NULL;
    }
    SDL_WaitForGPUFences(dev, true, &fence, 1);
    SDL_ReleaseGPUFence(dev, fence);
    return SDL_MapGPUTransferBuffer(dev, tb, false);
}

/* the last frame at the game's size, each pixel the average of those drawn whose centres are in it: slow, for
   screenshots */
bool rd_gpu_read(uint32_t* argb) {
    const uint8_t* p = download(frame_tex[0], frame_w, frame_h);
    int x, y, i, j, c, n, pitch = frame_w * 4;
    if (p == NULL) {
        return false;
    }
    for (y = 0; y < RD_HEIGHT; y++) {
        int y0 = (int)ceilf(y * scale - 0.5f), y1 = (int)ceilf((y + 1) * scale - 0.5f);
        for (x = 0; x < RD_WIDTH; x++) {
            int x0 = (int)ceilf(x * scale - 0.5f), x1 = (int)ceilf((x + 1) * scale - 0.5f);
            uint32_t sum[4] = { 0, 0, 0, 0 };
            n = (y1 - y0) * (x1 - x0);
            for (j = y0; j < y1; j++) {
                const uint8_t* q = p + (size_t)j * pitch + (size_t)x0 * 4;
                for (i = 0; i < (x1 - x0) * 4; i++) {
                    sum[i & 3] += q[i];
                }
            }
            for (c = 0; c < 4; c++) {
                sum[c] = (sum[c] + n / 2) / n;
            }
            argb[y * RD_WIDTH + x] = sum[3] << 24 | sum[0] << 16 | sum[1] << 8 | sum[2];
        }
    }
    SDL_UnmapGPUTransferBuffer(dev, tb);
    return true;
}

bool rd_gpu_read_view(uint32_t* argb) {
    int w, h, i;
    const uint8_t* p;
    rd_gpu_frame_size(&w, &h);
    p = download(out_tex != NULL ? out_tex : frame_tex[0], w, h);
    if (p == NULL) {
        return false;
    }
    for (i = 0; i < w * h; i++) {
        argb[i] = (uint32_t)p[4 * i + 3] << 24 | (uint32_t)p[4 * i] << 16 | (uint32_t)p[4 * i + 1] << 8 | p[4 * i + 2];
    }
    SDL_UnmapGPUTransferBuffer(dev, tb);
    return true;
}
