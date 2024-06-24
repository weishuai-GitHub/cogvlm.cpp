#ifndef CONFIG_H
#define CONFIG_H
#include <ggml.h>
#include <vector>
#include <string>
#include <sstream>
#include "modules.h"

struct VisionConfigRecord
{
    int in_channels;
    int num_hidden_layers;
    int hidden_size;
    int patch_size;
    int num_heads;
    int intermediate_size;
    int num_positions;
    int image_size;
};

struct ConfigRecord
{
    /* data */
    ggml_type dtype;
    VisionConfigRecord vision_config;
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int max_position_embeddings;
    int bos_token_id;
    int eos_token_id[2];
    int pad_token_id;
    int vocab_size;
    int num_hidden_layers;
    int num_multi_query_heads;
};

class VisionConfig
{
public:
    float dropout_prob;
    int in_channels;
    int num_hidden_layers;
    ActivationType hidden_act;
    int hidden_size;
    int patch_size;
    int num_heads;
    int intermediate_size;
    float layer_norm_eps;
    int num_positions;
    int image_size;

public:
    VisionConfig() = default;
    VisionConfig(float dropout_prob, int in_channels, int num_hidden_layers, ActivationType hidden_act, int hidden_size,
                 int patch_size, int num_heads, int intermediate_size, float layer_norm_eps, int num_positions, int image_size)
        : dropout_prob(dropout_prob), in_channels(in_channels), num_hidden_layers(num_hidden_layers), hidden_act(ActivationType::GELU),
          hidden_size(hidden_size), patch_size(patch_size), num_heads(num_heads), intermediate_size(intermediate_size),
          layer_norm_eps(layer_norm_eps), num_positions(num_positions), image_size(image_size) {}
    VisionConfig(const VisionConfigRecord &rec) : dropout_prob(0.0), in_channels(rec.in_channels), num_hidden_layers(rec.num_hidden_layers), hidden_act(ActivationType::GELU),
                                                  hidden_size(rec.hidden_size), patch_size(rec.patch_size), num_heads(rec.num_heads), intermediate_size(rec.intermediate_size),
                                                  layer_norm_eps(1e-06), num_positions(rec.num_positions), image_size(rec.image_size) {}
};

// Should save kv record of ModelConfig in the future
class ModelConfig
{
public:
    /* data */
    ModelType model_type;
    ggml_type dtype;
    VisionConfig model_vision_config;
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int max_position_embeddings;
    float rms_norm_eps;
    int bos_token_id;
    int eos_token_id[2];
    int pad_token_id;
    int vocab_size;
    int num_hidden_layers;
    int num_multi_query_heads;
    ActivationType hidden_act;
    std::vector<int> extra_eos_token_ids;

public:
    ModelConfig() = default;

    ModelConfig(ModelType model_type, ggml_type dtype, int vocab_size, int hidden_size, int num_attention_heads,
                int num_hidden_layers, int intermediate_size, float rms_norm_eps, int max_position_embeddings,
                int bos_token_id, int eos_token_id, int pad_token_id, int sep_token_id, const VisionConfig &model_vision_config)
        : model_type(model_type), dtype(dtype), model_vision_config(model_vision_config), hidden_size(hidden_size), intermediate_size(intermediate_size),
          num_attention_heads(num_attention_heads), max_position_embeddings(max_position_embeddings), rms_norm_eps(rms_norm_eps),
          bos_token_id(bos_token_id), eos_token_id{eos_token_id}, pad_token_id(pad_token_id), vocab_size(vocab_size),
          num_hidden_layers(num_hidden_layers),num_multi_query_heads(num_multi_query_heads),hidden_act(ActivationType::GELU) {}

    ModelConfig(ModelType model_type, const ConfigRecord &rec)
        : model_type(model_type), dtype(rec.dtype), hidden_size(rec.hidden_size), intermediate_size(rec.intermediate_size),
          num_attention_heads(rec.num_attention_heads), max_position_embeddings(rec.max_position_embeddings), rms_norm_eps(1e-05),
          bos_token_id(rec.bos_token_id), eos_token_id{rec.eos_token_id[0], rec.eos_token_id[1]}, pad_token_id(rec.pad_token_id), vocab_size(rec.vocab_size),
          num_hidden_layers(rec.num_hidden_layers), num_multi_query_heads(rec.num_multi_query_heads),hidden_act(ActivationType::SILU)
    {
        model_vision_config = VisionConfig(rec.vision_config);
    }
    std::string model_type_name() const { return to_string(model_type); }
    std::string model_to_string() const
    {
        std::ostringstream oss;
        oss << "ModelConfig(model_type=" << model_type_name() << ", dtype=" << ggml_type_name(dtype) << ", vocab_size=" << vocab_size
            << ", hidden_size=" << hidden_size << ", num_attention_heads=" << num_attention_heads << ", num_hidden_layers=" << num_hidden_layers
            << ", intermediate_size=" << intermediate_size << ", rms_norm_eps=" << rms_norm_eps << ", max_position_embeddings=" << max_position_embeddings
            << ", bos_token_id=" << bos_token_id << ", eos_token_id=[" << eos_token_id[0] << "," << eos_token_id[1] << "], pad_token_id=" << pad_token_id
            << ", num_multi_query_heads=" << num_multi_query_heads<< ", model_vision_config=(" << model_vision_config.dropout_prob << ", " << model_vision_config.in_channels << ", " << model_vision_config.num_hidden_layers
            << ", " << model_vision_config.hidden_size << ", " << model_vision_config.patch_size << ", " << model_vision_config.num_heads
            << ", " << model_vision_config.intermediate_size << ", " << model_vision_config.layer_norm_eps << ", " << model_vision_config.num_positions << ", " << model_vision_config.image_size
            << ", " << to_string(model_vision_config.hidden_act) << "))";
        return oss.str();
    }
};

struct GenerationConfig {
    int max_length;
    int max_new_tokens;
    int max_context_length;
    bool do_sample;
    int top_k;
    float top_p;
    float temperature;
    float repetition_penalty;
    int num_threads;

    GenerationConfig(int max_length = 4096, int max_new_tokens = -1, int max_context_length = 512,
                     bool do_sample = true, int top_k = 0, float top_p = 0.9, float temperature = 0.6,
                     float repetition_penalty = 1.f, int num_threads = 0)
        : max_length(max_length), max_new_tokens(max_new_tokens), max_context_length(max_context_length),
          do_sample(do_sample), top_k(top_k), top_p(top_p), temperature(temperature),
          repetition_penalty(repetition_penalty), num_threads(num_threads) {}
};
#endif
