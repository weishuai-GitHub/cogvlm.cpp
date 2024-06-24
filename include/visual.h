#ifndef VISUAL_H
#define VISUAL_H
#include <cmath>
#include <vector>
#include "config.h"
namespace visual{
class PatchEmbedding{
    public:
        PatchEmbedding()=default;

        PatchEmbedding(const ModelContext *ctx,const VisionConfig *config):
        proj(ctx,config->in_channels, config->hidden_size, config->patch_size, config->patch_size),
        cls_embedding(ggml_new_tensor_2d(ctx->ctx_w.get(), GGML_TYPE_F32, config->hidden_size,1)),
        position_embedding(ctx,config->num_positions,config->hidden_size,true){}
        
        ggml_tensor *forward(const ModelContext *ctx, ggml_tensor *images) const;
    public:
        Conv2d proj;
        ggml_tensor *cls_embedding;
        Embedding position_embedding;
};

class Attention{
    public:
    Attention(const VisionConfig *config)
    {
        num_heads = config->num_heads;
        int head_dim = config->hidden_size / num_heads;
        scale = 1.0f / std::sqrt(head_dim);
    }

    Attention(const ModelContext *ctx,const VisionConfig *config):
    query_key_value(ctx,config->hidden_size,config->hidden_size*3),
    dense(ctx,config->hidden_size,config->hidden_size)
    {
        num_heads = config->num_heads;
        int head_dim = config->hidden_size / num_heads;
        scale = 1.0f / std::sqrt(head_dim);
    }
    ggml_tensor * forward(const ModelContext *ctx, ggml_tensor *hidden_states) const;
    public:
    int num_heads;
    float scale;
    Linear query_key_value;
    Linear dense;
};

template <ActivationType activationType>
class BaseMLP
{
public:
    
    BaseMLP()=default;

    BaseMLP(const ModelContext *ctx, const VisionConfig * config):
    fc1(ctx, config->hidden_size, config->intermediate_size),
    fc2(ctx, config->intermediate_size, config->hidden_size)
    {
    }

    ggml_tensor *forward(const ModelContext *ctx, ggml_tensor *hidden_states) const
    {
        ggml_context *gctx = ctx->ctx_w.get();
        ggml_tensor *intermediate_output = fc1.forward(ctx, hidden_states);
        intermediate_output = tensor_assign_buffers(apply_activation_inplace<activationType>(gctx, intermediate_output));
        ggml_tensor *output = fc2.forward(ctx, intermediate_output);
        return output;
    }
public:
    Linear fc1;
    Linear fc2;
};

using MLP = BaseMLP<ActivationType::GELU>;

class TransformerLayer
{
public:
    TransformerLayer(const VisionConfig *config):
    attention(config){}

    TransformerLayer(const ModelContext *ctx, const VisionConfig *config):
    input_layernorm(ctx,config->hidden_size,config->layer_norm_eps),
    attention(ctx,config),
    mlp(ctx,config),
    post_attention_layernorm(ctx,config->hidden_size,config->layer_norm_eps){}
    ggml_tensor *forward(const ModelContext *ctx, ggml_tensor *hidden_states) const;

public:
    Attention attention;
    MLP mlp;
    LayerNorm input_layernorm;
    LayerNorm post_attention_layernorm;
};

class Transformer
{
    public:
    Transformer(const VisionConfig *config):
    layers(config->num_hidden_layers,TransformerLayer(config)){}

    Transformer(const ModelContext *ctx,const VisionConfig *config)
    {
        layers.clear();
        for(int i=0;i<config->num_hidden_layers;i++)
        {
            layers.emplace_back(TransformerLayer(ctx,config));
        }
    }
    ggml_tensor *forward(const ModelContext *ctx, ggml_tensor *hidden_states) const;
    public:
    std::vector<TransformerLayer> layers;
};

class GLU
{
    public:
    GLU()=default;
    GLU(const ModelContext *ctx, const ModelConfig *config,int in_features):
    linear_proj(ctx, in_features, config->hidden_size,false),
    norm1(ctx,config->hidden_size),
    dense_h_to_4h(ctx,config->hidden_size,config->intermediate_size,false),
    gate_proj(ctx,config->hidden_size,config->intermediate_size,false),
    dense_4h_to_h(ctx,config->intermediate_size,config->hidden_size,false){}
    ggml_tensor *forward(const ModelContext *ctx, ggml_tensor *hidden_states) const
    {
        ggml_context *gctx = ctx->ctx_w.get();
        ggml_tensor *hidden_states_proj = linear_proj.forward(ctx, hidden_states);
        hidden_states_proj = tensor_assign_buffers(apply_activation_inplace<ActivationType::GELU>(gctx, norm1.forward(ctx, hidden_states_proj)));
        ggml_tensor* x = tensor_assign_buffers(apply_activation_inplace<ActivationType::SILU>(gctx,gate_proj.forward(ctx, hidden_states_proj)));
        ggml_tensor* y = dense_h_to_4h.forward(ctx, hidden_states_proj);
        hidden_states_proj = dense_4h_to_h.forward(ctx, tensor_assign_buffers(ggml_mul_inplace(gctx, x, y)));
        return hidden_states_proj;
    }
    public:
    Linear linear_proj;
    LayerNorm norm1;
    // ggml_tensor* (act1)(ModelContext *ctx, ggml_tensor *hidden_states) const;
    // ggml_tensor* (act2)(ModelContext *ctx, ggml_tensor *hidden_states) const;
    Linear dense_h_to_4h;
    Linear gate_proj;
    Linear dense_4h_to_h;
};

class EVA2CLIPModel
{
    public:
    EVA2CLIPModel(const ModelConfig *config):
    transformer(&config->model_vision_config){}

    EVA2CLIPModel(const ModelContext *ctx,const ModelConfig *config):
    patch_embedding(ctx,&config->model_vision_config),
    transformer(ctx,&config->model_vision_config),
    linear_proj(ctx,config,config->model_vision_config.hidden_size),
    conv(ctx,config->model_vision_config.hidden_size,config->model_vision_config.hidden_size,2,2)
    {
        boi = ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32,config->hidden_size);
        eoi = ggml_new_tensor_1d(ctx->ctx_w.get(), GGML_TYPE_F32,config->hidden_size);
    }
    ggml_tensor *forward(const ModelContext *ctx, ggml_tensor *images) const;
    public:
    PatchEmbedding patch_embedding;
    Transformer transformer;
    GLU linear_proj;
    Conv2d conv;
    ggml_tensor* boi;
    ggml_tensor* eoi;
};
}
#endif