#ifndef MODULES_H
#define MODULES_H
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml.h"
#include "common.h"
#include "loader.h"
#include "enums.h"
#include <cstring>
// ===== bases =====
struct ggml_context_deleter_t
{
    void operator()(ggml_context *ctx) const noexcept { ggml_free(ctx); }
};

using unique_ggml_context_t = std::unique_ptr<ggml_context, ggml_context_deleter_t>;

static inline unique_ggml_context_t make_unique_ggml_context(size_t mem_size, void *mem_buffer, bool no_alloc)
{
    return unique_ggml_context_t(ggml_init({mem_size, mem_buffer, no_alloc}));
}

using StateDict = std::vector<std::pair<std::string, ggml_tensor *>>;

#ifdef GGML_USE_METAL
struct ggml_metal_context_deleter_t
{
    void operator()(ggml_metal_context *ctx) const noexcept { ggml_metal_free(ctx); }
};

using unique_ggml_metal_context_t = std::unique_ptr<ggml_metal_context, ggml_metal_context_deleter_t>;

static inline unique_ggml_metal_context_t make_unique_ggml_metal_context(int n_cb)
{
    return unique_ggml_metal_context_t(ggml_metal_init(n_cb));
}
#endif

struct ModelContext
{
    ggml_type dtype;
    unique_ggml_context_t ctx_w;  // weight
    unique_ggml_context_t ctx_kv; // kv cache
    unique_ggml_context_t ctx_b;  // buffer
    ggml_backend_buffer_t* buffer;
    struct ggml_tallocr* alloc;
    ggml_gallocr_t galloc;
    ggml_backend_t* backend = NULL;
#ifdef GGML_USE_METAL
    unique_ggml_metal_context_t ctx_metal;
#endif
    ggml_cgraph *gf;
    ggml_scratch scratch;
    std::vector<uninitialized_char> compute_buffer; // BLAS buffer
    std::vector<uninitialized_char> scratch_buffer; // intermediate tensor buffer
    std::string_view weight_buffer;                 // mapped weight
    std::vector<uninitialized_char> work_buffer;    // temporary buffer for graph computing
    struct ggml_context *ctx_temp;                 // temporary context
    public:
    void init_device_context() const;
    void malloc(ggml_tensor* tensor, void *data,bool copy=true) const;
};



class Embedding
{
public:
    Embedding() : weight(nullptr) {}
    Embedding(const ModelContext *ctx, int num_embeddings, int embedding_dim,bool is_fp32=false)
        : weight(ggml_new_tensor_2d(ctx->ctx_w.get(), 
        is_fp32? GGML_TYPE_F32:ctx->dtype,
        embedding_dim, num_embeddings)) {}

    ggml_tensor *forward(const ModelContext *ctx, ggml_tensor *input) const;

public:
    ggml_tensor *weight;
};

class Linear
{
public:
    Linear() : weight(nullptr), bias(nullptr) {}
    Linear(const ModelContext *ctx, int in_features, int out_features, bool use_bias = true)
        : weight(ggml_new_tensor_2d(ctx->ctx_w.get(), ctx->dtype, in_features, out_features)),
          bias(use_bias ? ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32, out_features) : nullptr) {}

    int in_features() const { return weight->ne[0]; }
    int out_features() const { return weight->ne[1]; }

    ggml_tensor *forward(const ModelContext *ctx, ggml_tensor *input) const;

public:
    ggml_tensor *weight; // [out_features, in_features]
    ggml_tensor *bias;   // [out_features]
};

class LayerNorm
{
public:
    LayerNorm() = default;
    LayerNorm(const ModelContext *ctx, int normalized_shape, bool inplace = true, float eps = 1e-5f)
        : weight(ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32, normalized_shape)),
          bias(ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32, normalized_shape)), inplace(inplace), eps(eps) {}

    ggml_tensor *forward(const ModelContext *ctx, ggml_tensor *input) const;

public:
    ggml_tensor *weight; // [normalized_shape]
    ggml_tensor *bias;   // [normalized_shape]
    bool inplace;
    float eps;
};

class RMSNorm
{
public:
    RMSNorm() = default;
    RMSNorm(const ModelContext *ctx, int normalized_shape, bool inplace = true, float eps = 1e-5f)
        : weight(ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32, normalized_shape)), inplace(inplace), eps(eps) {}

    ggml_tensor *forward(const ModelContext *ctx, ggml_tensor *input) const;

public:
    ggml_tensor *weight; // [normalized_shape]
    bool inplace;
    float eps;
};

class Conv2d
{
public:
    Conv2d() : weight(nullptr), bias(nullptr), stride(1) {}
    Conv2d(const ModelContext *ctx, int in_channels, int out_channels, int kernel_size, int stride, bool use_bias = true)
        : weight(ggml_new_tensor_4d(ctx->ctx_w.get(), GGML_TYPE_F16, kernel_size, kernel_size, in_channels, out_channels)),
          bias(use_bias ? ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32, out_channels) : nullptr), stride(stride) {}

    ggml_tensor *forward(const ModelContext *ctx, ggml_tensor *input) const;

public:
    ggml_tensor *weight;
    ggml_tensor *bias;
    int stride;
};

template <ActivationType ACT_TYPE>
static inline ggml_tensor *apply_activation_inplace(ggml_context *ctx, ggml_tensor *hidden_states)
{
    static_assert(ACT_TYPE == ActivationType::GELU || ACT_TYPE == ActivationType::SILU);
    if constexpr (ACT_TYPE == ActivationType::GELU)
    {
        hidden_states = tensor_assign_buffers(ggml_gelu_inplace(ctx, hidden_states));
    }
    else if constexpr (ACT_TYPE == ActivationType::SILU)
    {
        hidden_states = tensor_assign_buffers(ggml_silu_inplace(ctx, hidden_states));
    }
    else
    {
        CHATGLM_THROW << "Unknown activation type " << (int)ACT_TYPE;
    }
    return hidden_states;
}

// struct CausalContextMasker
// {
//     ggml_tensor *operator()(const ModelContext *ctx, ggml_tensor *attn_scores, int n_past) const
//     {
//         return tensor_assign_buffers(ggml_diag_mask_inf_inplace(ctx->ctx_b.get(), attn_scores, n_past));
//     }
// };

struct NoopRoper
{
    ggml_tensor *operator()(ModelContext *ctx, ggml_tensor *a, ggml_tensor *b, int n_ctx) const { return a; }
};

template <RopeType MODE, int DIM_SCALE>
struct BasicRoper
{
    ggml_tensor *operator()(const ModelContext *ctx, ggml_tensor *a, ggml_tensor *b, int n_ctx) const
    {
        // tensor a (activation) is of shape [qlen, heads, head_size]
        // tensor b (position_ids) is of shape [qlen]
        ggml_context *gctx = ctx->ctx_b.get();
#ifdef GGML_USE_CUBLAS
        if (!ggml_is_contiguous(a))
        {
            a = tensor_assign_buffers(ggml_cont(gctx, a));
        }
#endif
        const int head_size = a->ne[0];
        const int rope_dim = head_size / DIM_SCALE;
        a = tensor_assign_buffers(ggml_rope_ext(
            ctx->ctx_w.get(), a, b, NULL, 
            rope_dim, MODE, n_ctx, 
            500000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f)); // [qlen, heads, head_size]
        return a;
    }
};

using RotaryEmbedding = BasicRoper<ROPE_TYPE_DEFAULT, 1>;

#endif
