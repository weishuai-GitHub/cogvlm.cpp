#include "visual.h"
#include <iostream>
namespace visual{

ggml_tensor *PatchEmbedding::forward(const ModelContext *ctx, ggml_tensor *images) const //images = [C, H, W] 
{
    ggml_context *gctx = ctx->ctx_b.get();
    ggml_tensor *x = proj.forward(ctx, images);
    COGVLM_CHECK(ggml_n_dims(x)==3) << "x->n_dims==3";
    
    x = tensor_assign_buffers(ggml_reshape_3d(gctx, x, x->ne[0]*x->ne[1], x->ne[2],x->ne[3]));
    x = ggml_cont(gctx,x);
    x = tensor_assign_buffers(ggml_permute(gctx, x,1,0,2,3));
    x = ggml_cont(gctx,x);
    x = ggml_concat(gctx, cls_embedding, x,1);
    COGVLM_CHECK(ggml_n_dims(x)==2) << "x->n_dims==2";
    x = tensor_assign_buffers(ggml_add(gctx,position_embedding.weight, x));
    return x;
}

ggml_tensor* Attention::forward(const ModelContext *ctx, ggml_tensor *hidden_states) const
{
    const int B = hidden_states->ne[2];
    const int L = hidden_states->ne[1];
    const int qlen = hidden_states->ne[0];
    ggml_context *gctx = ctx->ctx_b.get();
    ggml_tensor *qkv = query_key_value.forward(ctx,hidden_states);
    
    // split mixed qkv into separate query, key and value
    ggml_tensor *query_layer; // [B,L, num_heads, D]
    ggml_tensor *key_layer;   // [B,L, num_heads, D]
    ggml_tensor *value_layer; // [B,L, num_heads, D]
    query_layer = ggml_view_3d(gctx, qkv, qlen/num_heads, num_heads, L, qlen/num_heads*ggml_element_size(qkv),
                                qlen*ggml_element_size(qkv), 0);
    query_layer = ggml_cont(gctx,query_layer);
    
    key_layer = ggml_view_3d(gctx, qkv, qlen/num_heads, num_heads, L, qlen/num_heads*ggml_element_size(qkv),
                                qlen*ggml_element_size(qkv), qlen*L*B * ggml_element_size(qkv));
    key_layer = ggml_cont(gctx,key_layer);
    
    value_layer = ggml_view_3d(gctx, qkv, qlen/num_heads,num_heads, L,qlen/num_heads*ggml_element_size(qkv),
                                qlen*ggml_element_size(qkv), (2*qlen*L*B) * ggml_element_size(qkv));
    value_layer = ggml_cont(gctx,value_layer);
    // ggml_build_forward_expand(ctx->gf, key_layer);
    // ggml_build_forward_expand(ctx->gf, value_layer);
    // compute attention scores
    ggml_tensor *attn_scores =
            tensor_assign_buffers(ggml_mul_mat(gctx, key_layer, query_layer));
    attn_scores = tensor_assign_buffers(
            ggml_scale_inplace(gctx, attn_scores, scale));
    ggml_tensor *attn_probs =
            tensor_assign_buffers(ggml_soft_max_inplace(gctx, attn_scores));
    ggml_tensor *out = tensor_assign_buffers(
            ggml_mul_mat(gctx,ggml_cont(gctx,ggml_transpose(gctx,value_layer)), attn_probs));
    // ggml_tensor *out = ggml_flash_attn_ext(gctx, query_layer,key_layer, 
    // value_layer, nullptr,scale,0.0f);
    out = ggml_reshape_2d(gctx,out,qlen,L);
    out = dense.forward(ctx,out);
    return out;             
}

ggml_tensor *TransformerLayer::forward(const ModelContext *ctx, ggml_tensor *hidden_states) const
{
    ggml_context *gctx = ctx->ctx_b.get();
    ggml_tensor *attention_output = input_layernorm.forward(ctx, attention.forward(ctx, hidden_states));
    hidden_states = tensor_assign_buffers(ggml_add_inplace(gctx, hidden_states, attention_output));
    ggml_tensor* mlp_output = post_attention_layernorm.forward(ctx, mlp.forward(ctx, hidden_states));
    ggml_tensor* output = tensor_assign_buffers(ggml_add_inplace(gctx, mlp_output, hidden_states));
    return output;
}

ggml_tensor *Transformer::forward(const ModelContext *ctx, ggml_tensor *hidden_states) const
{   
    // ggml_set_scratch(ctx->ctx_b.get(), ctx->scratch);
    for (TransformerLayer layer : layers)
    {   
        // ggml_set_scratch(ctx->ctx_b.get(), ctx->scratch);
        hidden_states = layer.forward(ctx, hidden_states);
        // ggml_build_forward_expand(ctx->gf, hidden_states);
    }
    return hidden_states;
}

ggml_tensor *EVA2CLIPModel::forward(const ModelContext *ctx, ggml_tensor *images) const
{   
    ggml_tensor *x = patch_embedding.forward(ctx, images);
    // ggml_build_forward_expand(ctx->gf, x);
    x = transformer.forward(ctx, x);
    x = ggml_cont(ctx->ctx_b.get(),x);
    x = ggml_view_2d(ctx->ctx_b.get(), x, x->ne[0], x->ne[1]-1,
                    x->ne[0]*ggml_element_size(x),
                    x->ne[0]*ggml_element_size(x));
    const int b = x->ne[2];
    const int s = x->ne[1];
    const int h = x->ne[0];
    const int grid_size = (int)(sqrt(s));
    x = ggml_reshape_4d(ctx->ctx_b.get(), x, h, grid_size, grid_size, b);
    x = ggml_permute(ctx->ctx_b.get(), x, 1,0,2, 3);
    x = ggml_permute(ctx->ctx_b.get(), x, 0,2,1, 3);
    
    x = ggml_cont(ctx->ctx_b.get(),x);
    x = conv.forward(ctx,x);
    x = tensor_assign_buffers(ggml_reshape_3d(ctx->ctx_b.get(), x, x->ne[0]*x->ne[1], x->ne[2],x->ne[3]));
    x = ggml_cont(ctx->ctx_b.get(),ggml_permute(ctx->ctx_b.get(), x,1,0,2,3));
    x = linear_proj.forward(ctx,x);
    COGVLM_CHECK(ggml_n_dims(x)==2) << "x->n_dims==2";
    x = ggml_concat(ctx->ctx_b.get(), boi, x,1);
    x = ggml_concat(ctx->ctx_b.get(), x, boi,1);
    return x;
}

} // namespace visual