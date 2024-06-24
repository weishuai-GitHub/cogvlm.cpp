#ifndef COGVLMLIB_H
#define COGVLMLIB_H
// #include <cmath>
#include "loader.h"
#include "visual.h"
#ifdef GGML_USE_METAL
#include <ggml-metal.h>
#endif
#ifdef GGML_USE_CUBLAS
#include <ggml-cuda.h>
#endif

namespace cogvlm
{
    static constexpr size_t MB = 1024 * 1024;
    static constexpr int LANGUAGE_TOKEN_TYPE = 0;
    static constexpr int VISION_TOKEN_TYPE = 1;

    struct TokenIdScore
    {
        int id;
        float score;

        TokenIdScore() = default;
        TokenIdScore(int id, float score) : id(id), score(score) {}

        bool operator<(const TokenIdScore &other) const { return score < other.score; }
        bool operator>(const TokenIdScore &other) const { return score > other.score; }

        friend std::ostream &operator<<(std::ostream &os, const TokenIdScore &self)
        {
            return os << "TokenIdScore(id=" << self.id << ", score=" << self.score << ")";
        }
    };

    static std::vector<int> vision_token_index, language_token_index;
    // ===== CogVLM2-19B =====
    struct FunctionMessage
    {
        std::string name;
        std::string arguments;

        FunctionMessage() = default;
        FunctionMessage(std::string name, std::string arguments) : name(std::move(name)), arguments(std::move(arguments)) {}

        friend std::ostream &operator<<(std::ostream &os, const FunctionMessage &self)
        {
            return os << "FunctionMessage(name=" << std::quoted(self.name) << ", arguments=" << std::quoted(self.arguments)
                      << ")";
        }
    };

    struct CodeMessage
    {
        std::string input;

        CodeMessage() = default;
        CodeMessage(std::string input) : input(std::move(input)) {}

        friend std::ostream &operator<<(std::ostream &os, const CodeMessage &self)
        {
            return os << "CodeMessage(input=" << std::quoted(self.input) << ")";
        }
    };

    struct ToolCallMessage
    {
        std::string type;
        FunctionMessage function;
        CodeMessage code;

        static const std::string TYPE_FUNCTION;
        static const std::string TYPE_CODE;

        ToolCallMessage(FunctionMessage function) : type(TYPE_FUNCTION), function(std::move(function)) {}

        ToolCallMessage(CodeMessage code) : type(TYPE_CODE), code(std::move(code)) {}

        friend std::ostream &operator<<(std::ostream &os, const ToolCallMessage &self)
        {
            return os << "ToolCallMessage(type=" << std::quoted(self.type) << ", function=" << self.function
                      << ", code=" << self.code << ")";
        }
    };

    struct ChatMessage
    {
        std::string role;
        std::string content;
        std::vector<ToolCallMessage> tool_calls;

        static const std::string ROLE_USER;
        static const std::string ROLE_ASSISTANT;
        static const std::string ROLE_SYSTEM;
        static const std::string ROLE_OBSERVATION;

        ChatMessage() = default;
        ChatMessage(std::string role, std::string content, std::vector<ToolCallMessage> tool_calls = {})
            : role(std::move(role)), content(std::move(content)), tool_calls(std::move(tool_calls)) {}

        friend std::ostream &operator<<(std::ostream &os, const ChatMessage &self)
        {
            os << "ChatMessage(role=" << std::quoted(self.role) << ", content=" << std::quoted(self.content)
               << ", tool_calls=[";
            for (size_t i = 0; i < self.tool_calls.size(); i++)
            {
                os << (i > 0 ? ", " : "") << self.tool_calls[i];
            }
            return os << "])";
        }
    };

    // ===== CogVLM2-19B =====
    class MLP
    {
    public:
        MLP(const ModelConfig *config) : hidden_size(config->hidden_size),
                                         intermediate_size(config->hidden_size){};

        MLP(const ModelContext *ctx, const ModelConfig *config) : hidden_size(config->hidden_size),
                                                                  intermediate_size(config->intermediate_size),
                                                                  gate_proj(ctx, hidden_size, intermediate_size, false),
                                                                  up_proj(ctx, hidden_size, intermediate_size, false),
                                                                  down_proj(ctx, intermediate_size, hidden_size, false) {}
        ggml_tensor *forward(const ModelContext *ctx, ggml_tensor *x) const;

    public:
        int hidden_size;
        int intermediate_size;
        Linear gate_proj;
        Linear up_proj;
        Linear down_proj;
    };

    class VisionExpertMLP
    {
    public:
        VisionExpertMLP(const ModelConfig *config) : language_mlp(config),
                                                     vision_mlp(config){};

        VisionExpertMLP(const ModelContext *ctx, const ModelConfig *config) : language_mlp(ctx, config),
                                                                              vision_mlp(ctx, config) {}
        ggml_tensor *forward(const ModelContext *ctx, ggml_tensor *hidden_states,
                             ggml_tensor *token_type_ids) const;

    public:
        MLP language_mlp;
        MLP vision_mlp;
    };

    struct cogvlm_attention_output
    {
        /* data */
        ggml_tensor *attn_output;
        ggml_tensor **past_key_value;
    };

    class VisionExpertAttention
    {
    public:
        VisionExpertAttention(const ModelConfig *config) : hidden_size(config->hidden_size),
                                                           num_attention_heads(config->num_attention_heads),
                                                           num_multi_query_heads(config->num_multi_query_heads),
                                                           hidden_size_per_attention_head(hidden_size / num_attention_heads),
                                                           stride({num_attention_heads, num_multi_query_heads, num_multi_query_heads}),
                                                           qkv_size(hidden_size + 2 * hidden_size_per_attention_head * num_multi_query_heads),
                                                           head_dim(hidden_size / num_attention_heads),
                                                           max_position_embeddings(config->max_position_embeddings){};

        VisionExpertAttention(const ModelContext *ctx, const ModelConfig *config) : hidden_size(config->hidden_size),
                                                                                    num_attention_heads(config->num_attention_heads),
                                                                                    num_multi_query_heads(config->num_multi_query_heads),
                                                                                    hidden_size_per_attention_head(hidden_size / num_attention_heads),
                                                                                    stride({num_attention_heads, num_multi_query_heads, num_multi_query_heads}),
                                                                                    qkv_size(hidden_size + 2 * hidden_size_per_attention_head * num_multi_query_heads),
                                                                                    head_dim(hidden_size / num_attention_heads),
                                                                                    max_position_embeddings(config->max_position_embeddings),
                                                                                    vision_expert_query_key_value(ctx, hidden_size, qkv_size, true),
                                                                                    vision_expert_dense(ctx, hidden_size, hidden_size, false),
                                                                                    language_expert_query_key_value(ctx, hidden_size, qkv_size, false),
                                                                                    language_expert_dense(ctx, hidden_size, hidden_size, false){};

        struct cogvlm_attention_output forward(const ModelContext *ctx, ggml_tensor *hidden_states,
                                               ggml_tensor *token_type_ids, ggml_tensor *position_ids, ggml_tensor *attention_mask,
                                               ggml_tensor **past_key_value, int n_ctx, bool use_cache = false);

        ggml_tensor *_transpose_for_scores(const ModelContext *ctx, ggml_tensor *tensor) const;

    public:
        int hidden_size;
        int num_attention_heads;
        int num_multi_query_heads;
        int hidden_size_per_attention_head;
        int stride[3];
        int qkv_size;
        int head_dim;
        int max_position_embeddings;
        Linear vision_expert_query_key_value;
        Linear vision_expert_dense;
        Linear language_expert_query_key_value;
        Linear language_expert_dense;

    private:
        RotaryEmbedding roper_;
    };

    class CogVLMDecoderLayer
    {
    public:
        CogVLMDecoderLayer(const ModelConfig *config) : hidden_size(config->hidden_size),
                                                        self_attn(config),
                                                        mlp(config){};

        CogVLMDecoderLayer(const ModelContext *ctx, const ModelConfig *config) : hidden_size(config->hidden_size),
                                                                                 self_attn(ctx, config),
                                                                                 mlp(ctx, config),
                                                                                 input_layernorm(ctx, hidden_size, config->rms_norm_eps),
                                                                                 post_attention_layernorm(ctx, hidden_size, config->rms_norm_eps) {}

        cogvlm_attention_output forward(const ModelContext *ctx, ggml_tensor *hidden_states,
                                        ggml_tensor *token_type_ids, ggml_tensor *position_ids,
                                        ggml_tensor *attention_mask, ggml_tensor **past_key_value,
                                        int n_ctx, bool use_cache = false);

    public:
        int hidden_size;
        VisionExpertAttention self_attn;
        VisionExpertMLP mlp;
        RMSNorm input_layernorm;
        RMSNorm post_attention_layernorm;
    };

    struct BaseModelOutputWithPast
    {
        /* data */
        ggml_tensor *last_hidden_state;
        std::vector<ggml_tensor **> past_key_values;
    };

    class CogVLMModel
    {
    public:
        CogVLMModel() = default;
        CogVLMModel(const ModelConfig *config) : padding_idx(128002),
                                                 vocab_size(config->vocab_size),
                                                 layers(config->num_hidden_layers, CogVLMDecoderLayer(config)),
                                                 vision(config){};

        CogVLMModel(const ModelContext *ctx, const ModelConfig *config) : padding_idx(128002),
                                                                          vocab_size(config->vocab_size),
                                                                          embed_tokens(ctx, config->vocab_size, config->hidden_size),
                                                                          norm(ctx, config->hidden_size),
                                                                          vision(ctx, config)
        {
            layers.clear();
            for (int i = 0; i < config->num_hidden_layers; i++)
            {
                layers.emplace_back(CogVLMDecoderLayer(ctx, config));
            }
        };

        struct BaseModelOutputWithPast forward(const ModelContext *ctx, ggml_tensor *input_ids,
                                               ggml_tensor *token_type_ids, int *attention_mask, ggml_tensor *images,
                                               ggml_tensor *position_ids, std::vector<ggml_tensor **> past_key_values,
                                               int n_ctx, bool use_cache = false);

        struct BaseModelOutputWithPast llm_forward(const ModelContext *ctx,
                                                   ggml_tensor *token_type_ids, ggml_tensor *position_ids,
                                                   int *attention_mask, ggml_tensor *inputs_embeds,
                                                   std::vector<ggml_tensor **> past_key_values,
                                                   int n_ctxbool, bool use_cache = false);

        ggml_tensor *_prepare_decoder_attention_mask(const ModelContext *ctx,
                                                     int *attention_mask, int batch_size,
                                                     int seq_length, ggml_tensor *inputs_embeds,
                                                     int past_key_values_length);

    public:
        int padding_idx;
        int vocab_size;
        Embedding embed_tokens;
        std::vector<CogVLMDecoderLayer> layers;
        RMSNorm norm;
        visual::EVA2CLIPModel vision;
    };

    struct CausalLMOutputWithPast
    {
        /* data */
        ggml_tensor *hidden_states;
        ggml_tensor *logits;
        std::vector<ggml_tensor **> past_key_values;
    };

    class BaseModelForCausalLM
    {
    public:
        BaseModelForCausalLM() = default;
        BaseModelForCausalLM(const ModelConfig *config,
                             size_t mem_size, size_t scratch_size, size_t num_weights)
        {
            ctx_.dtype = config->dtype;
            constexpr size_t ctx_w_size = 11 * 1024 * MB;
            const size_t ctx_kv_size = 1280 * MB;
            size_t buffer_size = 0;
            buffer_size = 1024 * 1024 * 16 * ggml_type_size(GGML_TYPE_F32);
            buffer_size += 1024; // overhead
            ctx_.ctx_w = make_unique_ggml_context(ggml_tensor_overhead() * 1024 * 5, nullptr, true);
            ctx_.ctx_kv = make_unique_ggml_context(ctx_kv_size + 2 * MB, nullptr, false); // 1MB extra for MPS
            // ctx_.ctx_b = make_unique_ggml_context(32*1024, nullptr, true);
            ctx_.compute_buffer.resize(mem_size);
            printf("mem_size: %d\n", mem_size);
            // ctx_.scratch_buffer.resize(scratch_size);
            // ctx_.scratch = {0, ctx_.scratch_buffer.size(), ctx_.scratch_buffer.data()};
            ctx_.backend = (ggml_backend_t *)malloc(sizeof(ggml_backend_t));
            ctx_.buffer = (ggml_backend_buffer_t *)malloc(sizeof(ggml_backend_buffer_t));
            ctx_.alloc = (struct ggml_tallocr *)malloc(sizeof(struct ggml_tallocr));

            struct ggml_init_params params
            {
                    /*.mem_size   =*/ggml_tensor_overhead() * 1024*32,
                    /*.mem_buffer =*/NULL,
                    /*.no_alloc   =*/true,
            };

#ifdef GGML_USE_CUBLAS
            // ggml_cuda_set_scratch_size(scratch_size);
            printf("using GPU backend\n");
            *ctx_.backend = ggml_backend_cuda_init(1);
#endif
#ifdef GGML_USE_METAL
            if (use_gpu)
            {
                fprintf(stderr, "%s: using Metal backend\n", __func__);
                *ctx_.backend = ggml_backend_metal_init();
                if (*ctx_.backend)
                {
                    fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
                }
            }
#endif
            if (!*ctx_.backend)
            {
                // fallback to CPU backend
                printf("using CPU backend\n");
                *ctx_.backend = ggml_backend_cpu_init();
            }
            *ctx_.buffer = ggml_backend_alloc_buffer(*ctx_.backend, buffer_size + ctx_w_size);
            ctx_.ctx_temp = ggml_init(params);
            ctx_.galloc = NULL;

        };

        ~BaseModelForCausalLM() = default;

        virtual void load(ModelLoader &loader) = 0;

        virtual struct CausalLMOutputWithPast forward(
            ggml_tensor *input_ids,
            ggml_tensor *images,
            ggml_tensor *token_type_ids,
            int *attention_mask,
            ggml_tensor *position_ids,
            std::vector<ggml_tensor **> past_key_values,
            int n_ctx, bool use_cache = false) = 0;

        // logits processor
        static void sampling_repetition_penalty(float *first, float *last, const std::vector<int> &input_ids,
                                                float penalty);
        // logits warper
        static void sampling_temperature(float *first, float *last, float temp);
        static void sampling_top_k(TokenIdScore *first, TokenIdScore *kth, TokenIdScore *last);
        static TokenIdScore *sampling_top_p(TokenIdScore *first, TokenIdScore *last, float top_p);

        static void sampling_softmax_inplace(TokenIdScore *first, TokenIdScore *last);

    protected:
        ModelContext ctx_;
    };


    struct GenerateOutput
    {
        std::vector<ggml_tensor **> past_key_values;
        int  next_token_id;
    };

    class CogVLMForCausalLM : public BaseModelForCausalLM
    {
    public:
        StateDict state_dict() const;

    public:
        CogVLMForCausalLM(const ModelConfig *config);
        static int num_weights(int num_hidden_layers) { return 4 + num_hidden_layers * 12; }

        void load(ModelLoader &loader);

        ggml_tensor *vision_forward(std::vector<float> buff);

        struct CausalLMOutputWithPast forward(
            ggml_tensor *input_ids,
            ggml_tensor *images,
            ggml_tensor *token_type_ids,
            int *attention_mask,
            ggml_tensor *position_ids,
            std::vector<ggml_tensor **> past_key_values,
            int n_ctx, bool use_cache = false);

        CausalLMOutputWithPast forward_graph_compute(
            ggml_tensor *input_ids,
            ggml_tensor *images,
            ggml_tensor *token_type_ids,
            int *attention_mask,
            ggml_tensor *position_ids,
            std::vector<ggml_tensor **> past_key_values,
            int n_ctx, bool use_cache = false);

        GenerateOutput generate_next_token(
            std::vector<int> &input_ids_vector,
            ggml_tensor *images,
            std::vector<int> &token_type_ids,
            int *attention_mask,
            ggml_tensor *position_ids,
            std::vector<ggml_tensor **> past_key_values,
            int n_ctx, bool use_cache = false);

        std::vector<int> generate(std::vector<int> &text_ids,
                                  std::vector<float> &images_vector);

    public:
        int vocab_size;
        CogVLMModel model;
        Linear lm_head;
        const ModelConfig *config;
        static constexpr size_t MEM_SIZE = 32 * MB; // 2k context
        GenerationConfig gen_config;
        // static constexpr size_t SCRATCH_SIZE = 15*4096 * MB; // 2k context
    protected:
        StateDict state_dict_;
        void to_cpu()
        {
            for (auto &item : state_dict_)
            {
                tensor_to_cpu(item.second);
            }

            // for (auto &layer : model.layers)
            // {
            //     tensor_to_cpu(layer.attention.k_cache);
            //     tensor_to_cpu(layer.attention.v_cache);
            // }
        }

        void to_device()
        {
            for (auto &item : state_dict_)
            {
                ggml_tensor *tensor = item.second;
                // should not place embedding onto device
                if (tensor != model.embed_tokens.weight)
                {
                    tensor_to_device(tensor);
                }
            }
        }
    };

    struct CogVLM2Input
    {
        std::vector<int> input_ids;
        std::vector<int> token_type_ids;
        std::vector<int> attention_mask;
        std::vector<std::vector<float>> images;
    };

    // ===== pipeline =====
    class Pipeline
    {
    public:
        Pipeline(const std::string &path);
        std::vector<int> generate(std::vector<int> &text_ids,
                                  std::vector<float> &images_vector)
        {
            printf("generate\n");
            return model->generate(text_ids, images_vector);
        }

    public:
        std::unique_ptr<MappedFile> mapped_file;
        std::unique_ptr<CogVLMForCausalLM> model;
        ModelConfig *config;
    };
} // namespace cogvlm
#endif