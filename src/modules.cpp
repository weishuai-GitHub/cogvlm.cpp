#include "modules.h"
#include <iostream>

void ModelContext::init_device_context() const {
#ifdef GGML_USE_METAL
    ctx_metal = make_unique_ggml_metal_context(1);

    const size_t max_size = ggml_get_max_tensor_size(ctx_w.get());

    void *weight_data = weight_buffer.empty() ? ggml_get_mem_buffer(ctx_w.get()) : (void *)weight_buffer.data();
    size_t weight_size = weight_buffer.empty() ? ggml_get_mem_size(ctx_w.get()) : weight_buffer.size();
    COGVLM_CHECK(ggml_metal_add_buffer(ctx_metal.get(), "weights", weight_data, weight_size, max_size));

    COGVLM_CHECK(ggml_metal_add_buffer(ctx_metal.get(), "kv", ggml_get_mem_buffer(ctx_kv.get()),
                                        ggml_get_mem_size(ctx_kv.get()), 0));

    void *compute_data = ctx_b ? ggml_get_mem_buffer(ctx_b.get()) : compute_buffer.data();
    size_t compute_size = ctx_b ? ggml_get_mem_size(ctx_b.get()) : compute_buffer.size();
    COGVLM_CHECK(ggml_metal_add_buffer(ctx_metal.get(), "compute", compute_data, compute_size, 0));

    COGVLM_CHECK(ggml_metal_add_buffer(ctx_metal.get(), "scratch", scratch.data, scratch.size, 0));
#endif
}


void ModelContext::malloc(ggml_tensor* tensor, void *data,bool copy) const{
    // alloc memory
    ggml_tallocr_alloc(alloc, tensor);
    // load data to buffer
    if(ggml_backend_is_cpu(*backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(*backend)
#endif
    ) {
        if(copy) memcpy(tensor->data, data, ggml_nbytes(tensor));
        else tensor->data = data;
    } else {
        // cuda requires copy the data directly to device
        ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor)); 
    }
}

// ===== modules =====
ggml_tensor *Embedding::forward(const ModelContext *ctx, ggml_tensor *input) const
{
    ggml_tensor *output = ggml_get_rows(ctx->ctx_b.get(), weight, input);
    return output;
}

ggml_tensor *Linear::forward(const ModelContext *ctx, ggml_tensor *input) const
{
    // input: [seqlen, in_features]
    ggml_context *gctx = ctx->ctx_b.get();
    ggml_tensor *output = tensor_assign_buffers(ggml_mul_mat(gctx, weight, input)); // [seqlen, out_features]
    if (bias)
    {
        output = tensor_assign_buffers(ggml_add_inplace(gctx, output, bias));
    }
    return output;
}

ggml_tensor *LayerNorm::forward(const ModelContext *ctx, ggml_tensor *input) const
{
    // input: [seqlen, normalized_shape]
    ggml_context *gctx = ctx->ctx_b.get();
    auto ggml_norm_fn = inplace ? ggml_norm_inplace : ggml_norm;
    ggml_tensor *output = tensor_assign_buffers(ggml_norm_fn(gctx, input, eps));
    output = tensor_assign_buffers(ggml_mul_inplace(gctx, output, weight));
    output = tensor_assign_buffers(ggml_add_inplace(gctx, output, bias));
    return output;
}

ggml_tensor *RMSNorm::forward(const ModelContext *ctx, ggml_tensor *input) const
{
    ggml_context *gctx = ctx->ctx_b.get();
    auto ggml_rms_norm_fn = inplace ? ggml_rms_norm_inplace : ggml_rms_norm;
    ggml_tensor *output = tensor_assign_buffers(ggml_rms_norm_fn(gctx, input, eps));
    output = tensor_assign_buffers(ggml_mul_inplace(gctx, output, weight));
    return output;
}

ggml_tensor *Conv2d::forward(const ModelContext *ctx, ggml_tensor *input) const
{
    ggml_context *gctx = ctx->ctx_b.get();
    ggml_tensor *output = ggml_conv_2d(gctx,weight, input,stride,stride, 0, 0, 1, 1);
    if (bias)
    {
        output = tensor_assign_buffers(ggml_add_inplace(gctx, output, 
        ggml_reshape_3d(gctx, bias, 1, 1, bias->ne[0])));
    }
    return output;
}
