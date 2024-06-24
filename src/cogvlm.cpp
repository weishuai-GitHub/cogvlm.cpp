#include <cogvlm.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "visual.h"
#include <functional>
#include <random>
// #ifdef GGML_USE_CUBLAS
// #include <ggml-cuda.h>
// #endif
namespace cogvlm
{
    void *get_expert_mask(const ModelContext *ctx,
                          ggml_tensor *token_type_ids, ggml_tensor *output[2])
    {
        int size = ggml_nelements(token_type_ids);
        COGVLM_CHECK(size == token_type_ids->ne[0]) << "size mismatch";
        int *mask = new int[size];
        ggml_backend_tensor_get(token_type_ids, mask, 0, ggml_nbytes(token_type_ids));

        vision_token_index.clear();
        language_token_index.clear();
        language_token_index.push_back(0);
        for (int i = 1; i < size; i++)
        {
            if (mask[i] == VISION_TOKEN_TYPE)
            {
                vision_token_index.push_back(i);
            }
            else
            {
                language_token_index.push_back(i);
            }
        }
        output[0] = ggml_new_tensor_1d(ctx->ctx_temp, GGML_TYPE_I32, vision_token_index.size());
        output[1] = ggml_new_tensor_1d(ctx->ctx_temp, GGML_TYPE_I32, language_token_index.size());
        // language_token_mask = ~vision_token_mask
        ctx->malloc(output[0], vision_token_index.data());
        ctx->malloc(output[1], language_token_index.data());
    }

    ggml_tensor *attention_fn(const ModelContext *ctx, ggml_tensor *query_layer,
                              ggml_tensor *key_layer, ggml_tensor *value_layer,
                              ggml_tensor *attention_mask,
                              bool scaling_attention_score = true)
    {

        ggml_tensor *attention_scores = ggml_mul_mat(ctx->ctx_b.get(), key_layer, query_layer);
        
        if (scaling_attention_score)
        {
            attention_scores = tensor_assign_buffers(ggml_scale_inplace(ctx->ctx_b.get(),
                                                                        attention_scores, 1.0f / sqrtf(query_layer->ne[0])));
        }

        attention_scores = ggml_add_inplace(ctx->ctx_b.get(), attention_scores, attention_mask);

        ggml_tensor *attn_probs =
            tensor_assign_buffers(ggml_soft_max_inplace(ctx->ctx_b.get(), attention_scores));

        ggml_tensor *context_layer = tensor_assign_buffers(
            ggml_mul_mat(ctx->ctx_b.get(),
                         ggml_cont(ctx->ctx_b.get(), ggml_transpose(ctx->ctx_b.get(), value_layer)),
                         attn_probs));
        // ggml_tensor *context_layer = ggml_flash_attn_ext(
        //     ctx->ctx_b.get(), query_layer, key_layer, value_layer,
        //     attention_mask, 1.0f / sqrtf(query_layer->ne[0]), 0.0f);
        return context_layer;
    }

    ggml_tensor *build_position_ids(const ModelContext *ctx, std::vector<int> x, std::vector<int> attention_mask)
    {
        std::vector<int> tmp;

        tmp.insert(tmp.end(), attention_mask.begin(), attention_mask.end());

        std::vector<int> is_boi_eoi = std::vector<int>(x.size(), 0);
        for (int i = 1; i < is_boi_eoi.size(); i++)
        {
            is_boi_eoi[i] = tmp[i] == VISION_TOKEN_TYPE && tmp[i - 1] == LANGUAGE_TOKEN_TYPE;
        }
        is_boi_eoi[0] = tmp[0] == VISION_TOKEN_TYPE;
        for (int i = 0; i < is_boi_eoi.size() - 1; i++)
        {
            is_boi_eoi[i] |= tmp[i] == VISION_TOKEN_TYPE && tmp[i + 1] == LANGUAGE_TOKEN_TYPE;
        }
        is_boi_eoi[is_boi_eoi.size() - 1] |= tmp[is_boi_eoi.size() - 1] == VISION_TOKEN_TYPE;
        for (int i = 0; i < is_boi_eoi.size(); i++)
        {
            if (is_boi_eoi[i])
                tmp[i] = LANGUAGE_TOKEN_TYPE;
        }
        std::vector<int> y = std::vector<int>(x.size(), 0);
        for (int i = 1; i < x.size(); i++)
        {
            y[i] = tmp[i] == LANGUAGE_TOKEN_TYPE | (tmp[i] == VISION_TOKEN_TYPE && tmp[i - 1] == LANGUAGE_TOKEN_TYPE);
            y[i] += y[i - 1];
        }

        ggml_tensor *position_ids = ggml_new_tensor_1d(ctx->ctx_temp, GGML_TYPE_I32, x.size());

        ctx->malloc(position_ids, x.data());
        return position_ids;
    }

    ggml_tensor *_make_causal_mask(const ModelContext *ctx, int bsz, int tgt_len, int past_key_values_length)
    {
        float *mask_data = new float[tgt_len * tgt_len];
        for (int i = 0; i < tgt_len; i++)
        {
            for (int j = 0; j < tgt_len; j++)
            {
                if (j <= i)
                    mask_data[i * tgt_len + j] = 0.0f;
                else
                    mask_data[i * tgt_len + j] = -65504.0f;
            }
        }
        ggml_tensor *mask = nullptr;
        if (past_key_values_length > 0)
        {
            mask = ggml_new_tensor_2d(ctx->ctx_temp, GGML_TYPE_F32, tgt_len+past_key_values_length, tgt_len);
            float *mask_data_ = new float[tgt_len * (past_key_values_length+tgt_len)];
            std::fill(mask_data_, mask_data_ + tgt_len * past_key_values_length, 0.0f);
            std::copy(mask_data, mask_data + tgt_len * tgt_len, mask_data_+tgt_len * past_key_values_length);
            ctx->malloc(mask, mask_data_);
        }
        else
        {
            mask = ggml_new_tensor_2d(ctx->ctx_temp, GGML_TYPE_F32, tgt_len, tgt_len);
            ctx->malloc(mask, mask_data);
        }
        return mask;
    }

    ggml_tensor *_expand_mask(const ModelContext *ctx, int *mask, int tgt_len, int src_len)
    {
        float *expanded_mask = new float[tgt_len * src_len];
        for (int i = 0; i < tgt_len; i++)
        {
            for (int j = 0; j < src_len; j++)
            {
                if (mask[j] != 1)
                    expanded_mask[i * src_len + j] = -65504.0f;
                else
                    expanded_mask[i * src_len + j] = 1 - mask[j];
            }
        }
        ggml_tensor *inverted_mask = ggml_new_tensor_2d(ctx->ctx_temp, GGML_TYPE_F32, src_len, tgt_len);
        ctx->malloc(inverted_mask, expanded_mask);
        // if(past_key_values_length>0)
        // {
        //     ggml_tensor* mask_ = ggml_new_tensor_2d(ctx->ctx_b.get(),GGML_TYPE_F32,past_key_values_length,tgt_len);
        //     ggml_tensor* mask__ = ggml_new_tensor_2d(ctx->ctx_b.get(),GGML_TYPE_F32,past_key_values_length,tgt_len);
        //     mask = ggml_concat(ctx->ctx_b.get(),mask_,mask,0);
        //     mask = ggml_concat(ctx->ctx_b.get(),mask,mask__,0);
        //     mask = ggml_cont(ctx->ctx_b.get(),mask);
        // }
        return inverted_mask;
    }

    // build the compute graph to perform a matrix multiplication
    struct ggml_cgraph *build_graph(ggml_tensor *result)
    {
        static size_t buf_size = ggml_tensor_overhead() * 4608 + ggml_graph_overhead();
        static std::vector<uint8_t> buf(buf_size);

        struct ggml_init_params params0 = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/buf.data(),
            /*.no_alloc   =*/true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
        };

        // create a temporally context to build the graph
        struct ggml_context *ctx0 = ggml_init(params0);

        struct ggml_cgraph *gf = ggml_new_graph_custom(ctx0, 4608, false);

        // build operations nodes
        ggml_build_forward_expand(gf, result);

        // delete the temporally context used to build the graph
        ggml_free(ctx0);
        return gf;
    }

    ggml_tensor *compute(ggml_tensor *result,
                         ggml_gallocr_t galloc, ggml_backend_t backend, int n_threads)
    {
        printf("computer!!\n");
        ggml_cgraph *gf = build_graph(result);

        ggml_gallocr_alloc_graph(galloc, gf);

        printf("gf->n_nodes=%d\n", gf->n_nodes);
        PRINT_SHAPE("result", result);
        if (ggml_backend_is_cpu(backend))
        {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }

#ifdef GGML_USE_METAL
        if (ggml_backend_is_metal(model.backend))
        {
            ggml_backend_metal_set_n_cb(model.backend, n_threads);
        }
#endif
        ggml_backend_graph_compute(backend, gf);
#ifdef GGML_PERF
        ggml_graph_print(ctx_.gf);
#endif
        return gf->nodes[gf->n_nodes - 1];
    }

    ggml_tensor *MLP::forward(const ModelContext *ctx, ggml_tensor *x) const
    {
        ggml_context *gctx = ctx->ctx_b.get();
        ggml_tensor *gate = tensor_assign_buffers(
            apply_activation_inplace<ActivationType::SILU>(gctx, gate_proj.forward(ctx, x)));

        ggml_tensor *down = down_proj.forward(ctx,
                                              ggml_mul_inplace(gctx, gate, up_proj.forward(ctx, x)));
        return down;
    }

    ggml_tensor *VisionExpertMLP::forward(const ModelContext *ctx,
                                          ggml_tensor *hidden_states, ggml_tensor *token_type_ids) const
    {
        ggml_tensor *mask[2];
        get_expert_mask(ctx, token_type_ids, mask);
        ggml_tensor *vision_input = ggml_get_rows(ctx->ctx_b.get(), hidden_states, mask[0]);
        ggml_tensor *language_input = ggml_get_rows(ctx->ctx_b.get(), hidden_states, mask[1]);
        ggml_tensor *vision_output = vision_mlp.forward(ctx, vision_input);
        ggml_tensor *language_output = language_mlp.forward(ctx, language_input);
        size_t len = language_output->ne[0] * ggml_element_size(language_output);
        ggml_tensor *language_output_eop = ggml_view_2d(ctx->ctx_b.get(), language_output,
                                                        language_output->ne[0], 1, len, 0);
        ggml_tensor *language_output_left = ggml_view_2d(ctx->ctx_b.get(), language_output,
                                                         language_output->ne[0], language_output->ne[1] - 1, len, len);
        ggml_tensor *output = ggml_concat(ctx->ctx_b.get(), language_output_eop, vision_output, 1);
        output = ggml_concat(ctx->ctx_b.get(), output, language_output_left, 1);
        return ggml_cont(ctx->ctx_b.get(), output);
    }

    ggml_tensor *VisionExpertAttention::_transpose_for_scores(
        const ModelContext *ctx, ggml_tensor *tensor) const
    {
        size_t H = tensor->ne[0] / hidden_size_per_attention_head;
        int size = ggml_element_size(tensor);
        ggml_tensor *output = ggml_view_3d(ctx->ctx_b.get(), tensor, hidden_size_per_attention_head, H,
                                           tensor->ne[1], hidden_size_per_attention_head * size,
                                           tensor->ne[0] * size, 0);
        return output;
    }

    struct cogvlm_attention_output VisionExpertAttention::forward(
        const ModelContext *ctx, ggml_tensor *hidden_states,
        ggml_tensor *token_type_ids, ggml_tensor *position_ids,
        ggml_tensor *attention_mask, ggml_tensor **past_key_value,
        int n_ctx, bool use_cache)
    {
        ggml_context *gctx = ctx->ctx_b.get();
        int q_len = hidden_states->ne[1];
        int bsz = hidden_states->ne[2];
        ggml_tensor *mask[2];
        get_expert_mask(ctx, token_type_ids, mask);

        ggml_tensor *vision_input = ggml_get_rows(ctx->ctx_b.get(), hidden_states, mask[0]);
        ggml_tensor *language_input = ggml_get_rows(ctx->ctx_b.get(), hidden_states, mask[1]);
        ggml_tensor *vision_exp = vision_expert_query_key_value.forward(ctx, vision_input);
        ggml_tensor *language_exp = language_expert_query_key_value.forward(ctx, language_input);

        size_t len = language_exp->ne[0] * ggml_element_size(language_exp);
        ggml_tensor *language_exp_eop = ggml_view_2d(ctx->ctx_b.get(), language_exp,
                                                     language_exp->ne[0], 1, len, 0);
        ggml_tensor *language_exp_left = ggml_view_2d(ctx->ctx_b.get(), language_exp,
                                                      language_exp->ne[0], language_exp->ne[1] - 1, len, len);

        ggml_tensor *mixed_raw_layer = ggml_concat(ctx->ctx_b.get(), language_exp_eop, vision_exp, 1);
        mixed_raw_layer = ggml_concat(ctx->ctx_b.get(), mixed_raw_layer, language_exp_left, 1);
        mixed_raw_layer = ggml_cont(ctx->ctx_b.get(), mixed_raw_layer);
        int factor = mixed_raw_layer->ne[0] / (stride[0] + stride[1] + stride[2]);

        // split mixed_raw_layer to get query_states,key_states,value_states
        ggml_tensor *query_states, *key_states, *value_states;
        query_states = ggml_view_2d(gctx, mixed_raw_layer, factor * stride[0], mixed_raw_layer->ne[1],
                                    factor * stride[1] * ggml_element_size(mixed_raw_layer), 0);
        query_states = ggml_cont(ctx->ctx_b.get(), query_states);

        key_states = ggml_view_2d(gctx, mixed_raw_layer, factor * stride[1], mixed_raw_layer->ne[1],
                                  factor * stride[1] * ggml_element_size(mixed_raw_layer),
                                  mixed_raw_layer->ne[1] * factor * stride[0] * ggml_element_size(mixed_raw_layer));
        key_states = ggml_cont(ctx->ctx_b.get(), key_states);

        value_states = ggml_view_2d(gctx, mixed_raw_layer, factor * stride[2], mixed_raw_layer->ne[1],
                                    factor * stride[2] * ggml_element_size(mixed_raw_layer),
                                    mixed_raw_layer->ne[1] * factor * (stride[0] + stride[1]) * ggml_element_size(mixed_raw_layer));
        value_states = ggml_cont(ctx->ctx_b.get(), value_states);

        query_states = ggml_cont(ctx->ctx_b.get(), _transpose_for_scores(ctx, query_states)); //[H,L,HD]
        key_states = ggml_cont(ctx->ctx_b.get(), _transpose_for_scores(ctx, key_states));     //[H,L,HD]
        value_states = ggml_cont(ctx->ctx_b.get(), _transpose_for_scores(ctx, value_states)); //[H,L,HD]
        // ggml_build_forward_expand(ctx->gf, query_states);
        // ggml_build_forward_expand(ctx->gf, key_states);

        query_states = roper_(ctx, query_states, position_ids, n_ctx);
        key_states = roper_(ctx, key_states, position_ids, n_ctx);
        query_states = ggml_cont(ctx->ctx_b.get(), ggml_permute(ctx->ctx_b.get(), query_states, 0, 2, 1, 3));
        key_states = ggml_cont(ctx->ctx_b.get(), ggml_permute(ctx->ctx_b.get(), key_states, 0, 2, 1, 3));
        value_states = ggml_cont(ctx->ctx_b.get(), ggml_permute(ctx->ctx_b.get(), value_states, 0, 2, 1, 3));

        if (past_key_value != nullptr)
        {
            key_states = ggml_concat(ctx->ctx_kv.get(), past_key_value[0], key_states, 1);
            value_states = ggml_concat(ctx->ctx_kv.get(), past_key_value[1], value_states, 1);
        }
        if (use_cache)
        {
            if (past_key_value == nullptr)
            {
                past_key_value = new ggml_tensor *[2];
            }
            past_key_value[0] = key_states;
            past_key_value[1] = value_states;
        }
        ggml_tensor *tmp = ggml_new_tensor_4d(ctx->ctx_b.get(),
                                              key_states->type, key_states->ne[0], key_states->ne[1],
                                              num_attention_heads, bsz);

        key_states = ggml_repeat(ctx->ctx_b.get(), key_states, tmp);
        value_states = ggml_repeat(ctx->ctx_b.get(), value_states, tmp);
        
        ggml_tensor *context_layer = attention_fn(
            ctx, query_states, key_states, value_states,
            attention_mask, true);

        context_layer = ggml_cont(ctx->ctx_b.get(),
                                  ggml_permute(ctx->ctx_b.get(), context_layer, 0, 2, 1, 3));
        context_layer = ggml_cont(ctx->ctx_b.get(),
                                  ggml_reshape_3d(ctx->ctx_b.get(), context_layer, hidden_size, q_len, bsz));

        vision_input = ggml_get_rows(ctx->ctx_b.get(), context_layer, mask[0]);
        language_input = ggml_get_rows(ctx->ctx_b.get(), context_layer, mask[1]);

        vision_exp = vision_expert_dense.forward(ctx, vision_input);
        language_exp = language_expert_dense.forward(ctx, language_input);

        len = language_exp->ne[0] * ggml_element_size(language_exp);
        language_exp_eop = ggml_view_2d(ctx->ctx_b.get(), language_exp,
                                        language_exp->ne[0], 1, len, 0);
        language_exp_left = ggml_view_2d(ctx->ctx_b.get(), language_exp,
                                         language_exp->ne[0], language_exp->ne[1] - 1, len, len);

        ggml_tensor *attn_output = ggml_concat(ctx->ctx_b.get(), language_exp_eop, vision_exp, 1);
        attn_output = ggml_concat(ctx->ctx_b.get(), attn_output, language_exp_left, 1);
        cogvlm_attention_output output{attn_output, past_key_value};
        return output;
    }

    cogvlm_attention_output CogVLMDecoderLayer::forward(const ModelContext *ctx,
                                                        ggml_tensor *hidden_states, ggml_tensor *token_type_ids,
                                                        ggml_tensor *position_ids, ggml_tensor *attention_mask,
                                                        ggml_tensor **past_key_value,
                                                        int n_ctx, bool use_cache)
    {
        ggml_tensor *residual = hidden_states;
        hidden_states = input_layernorm.forward(ctx, hidden_states);
        // ggml_build_forward_expand(ctx->gf, hidden_states);
        cogvlm_attention_output output = self_attn.forward(ctx,
                                                           hidden_states, token_type_ids, position_ids, attention_mask,
                                                           past_key_value, n_ctx, use_cache);

        hidden_states = ggml_add_inplace(ctx->ctx_b.get(),
                                         output.attn_output, residual);
        // Fully Connected
        residual = hidden_states;
        hidden_states = post_attention_layernorm.forward(ctx, hidden_states);
        hidden_states = mlp.forward(ctx, hidden_states, token_type_ids);
        hidden_states = ggml_add_inplace(ctx->ctx_b.get(),
                                         hidden_states, residual);
        output.attn_output = hidden_states;
        return output;
    }

    struct BaseModelOutputWithPast CogVLMModel::forward(const ModelContext *ctx, ggml_tensor *input_ids,
                                                        ggml_tensor *token_type_ids, int *attention_mask, ggml_tensor *images,
                                                        ggml_tensor *position_ids, std::vector<ggml_tensor **> past_key_values,
                                                        int n_ctx, bool use_cache)
    {
        ggml_tensor *inputs_embeds = nullptr;
        ggml_tensor *images_features = nullptr;
        COGVLM_CHECK(input_ids != nullptr) << "input_ids is required";
        printf("past_key_values.size()=%d\n",past_key_values.size());
        int n_threads = get_default_num_threads();
        if (past_key_values.size() == 0 && images!=nullptr)
        {
            // input_ids 仅包含文字
            ggml_tensor *language_embeds = embed_tokens.forward(ctx, input_ids);

            size_t len = language_embeds->ne[0] * ggml_element_size(language_embeds);

            ggml_tensor *language_embeds_eop = ggml_view_2d(ctx->ctx_b.get(), language_embeds,
                                                            language_embeds->ne[0], 1, len, 0);

            ggml_tensor *language_embeds_left = ggml_view_2d(ctx->ctx_b.get(), language_embeds,
                                                             language_embeds->ne[0], language_embeds->ne[1] - 1, len, len);
            
            images_features = vision.forward(ctx, images);
            // images_features = compute(images_features, ctx->galloc, *ctx->backend, n_threads);
            // ggml_tensor* images_features_new = ggml_reshape_3d(ctx->ctx_b.get(), images_features,
            //                                   images_features->ne[0], images_features->ne[1], images_features->ne[2]);
            // ctx->malloc(images_features_new,images_features->data);
            inputs_embeds = ggml_concat(ctx->ctx_b.get(), language_embeds_eop, images_features, 1);
            inputs_embeds = ggml_concat(ctx->ctx_b.get(), inputs_embeds, language_embeds_left, 1);
        }
        else
            inputs_embeds = embed_tokens.forward(ctx, input_ids);
        // ggml_set_name(inputs_embeds,"inputs_embeds");
        // inputs_embeds = compute(inputs_embeds, ctx->galloc, *ctx->backend, n_threads);
        // ggml_set_input(inputs_embeds);
        input_ids = nullptr;
        COGVLM_CHECK(inputs_embeds != nullptr) << "inputs_embeds is required";

        printf("llm_forward enter\n");
        return llm_forward(ctx, token_type_ids, position_ids, attention_mask,
                           inputs_embeds, past_key_values, n_ctx, use_cache);
    }

    struct BaseModelOutputWithPast CogVLMModel::llm_forward(const ModelContext *ctx,
                                                            ggml_tensor *token_type_ids, ggml_tensor *position_ids,
                                                            int *attention_mask, ggml_tensor *inputs_embeds,
                                                            std::vector<ggml_tensor **> past_key_values, int n_ctx,
                                                            bool use_cache)
    {
        int batch_size = inputs_embeds->ne[2];
        int seq_length = inputs_embeds->ne[1];
        int past_key_values_length = 0, seq_length_with_past = seq_length;
        int n_threads = get_default_num_threads();
        if (past_key_values.size() > 0)
        {
            past_key_values_length = past_key_values[0][0]->ne[1];
            seq_length_with_past = seq_length_with_past + past_key_values_length;
        }
        if (position_ids == nullptr)
        {
            position_ids = ggml_new_tensor_1d(ctx->ctx_b.get(), GGML_TYPE_I32, seq_length);
            int *position_id = new int[seq_length];
            for(int i=0;i<seq_length;i++)
            {
                position_id[i] = i + past_key_values_length;
            }
            ctx->malloc(position_ids, position_id,false);
        }
        if (attention_mask == nullptr)
        {
            attention_mask = new int[seq_length];
            std::fill(attention_mask, attention_mask + seq_length, 1);
        }
        // TODO: implement
        ggml_tensor *attention_mask_tensor = _prepare_decoder_attention_mask(
            ctx, attention_mask, batch_size,
            seq_length, inputs_embeds,
            past_key_values_length);
        ggml_tensor *hidden_states = inputs_embeds;
        ggml_tensor **past_key_value = nullptr;
        std::vector<ggml_tensor **> next_decoder_cache;
        struct cogvlm_attention_output layer_outputs;
        for (int idx = 0; idx < layers.size(); idx++)
        {
            if (!past_key_values.empty() && past_key_values.size() > idx)
            {
                past_key_value = past_key_values[idx];
            }
            layer_outputs = layers[idx].forward(
                ctx,
                hidden_states,
                token_type_ids,
                position_ids,
                attention_mask_tensor,
                past_key_value,
                n_ctx,
                use_cache);

            hidden_states = layer_outputs.attn_output;

            if (use_cache)
            {
                next_decoder_cache.emplace_back(layer_outputs.past_key_value);
            }
        }

        hidden_states = norm.forward(ctx, hidden_states);
        return BaseModelOutputWithPast{
            hidden_states,
            next_decoder_cache,
        };
    }

    ggml_tensor *CogVLMModel::_prepare_decoder_attention_mask(const ModelContext *ctx,
                                                              int *attention_mask, int batch_size,
                                                              int seq_length, ggml_tensor *inputs_embeds,
                                                              int past_key_values_length)
    {
        ggml_tensor *combined_attention_mask = nullptr;
        if (seq_length > 1)
            combined_attention_mask = _make_causal_mask(
                ctx, batch_size, seq_length, past_key_values_length);
        if (attention_mask == nullptr)
        {
            return combined_attention_mask;
        }
        ggml_tensor *expanded_attn_mask = _expand_mask(
            ctx, attention_mask, seq_length, seq_length);

        if (combined_attention_mask == nullptr)
        {
            return expanded_attn_mask;
        }

        combined_attention_mask = ggml_add_inplace(ctx->ctx_b.get(),
                                                   combined_attention_mask, expanded_attn_mask);
        return combined_attention_mask;
    }

    void BaseModelForCausalLM::sampling_repetition_penalty(float *first, float *last, const std::vector<int> &input_ids,
                                                           float penalty)
    {
        COGVLM_CHECK(penalty > 0) << "penalty must be a positive float, but got " << penalty;
        const float inv_penalty = 1.f / penalty;
        const int vocab_size = last - first;
        std::vector<bool> occurrence(vocab_size, false);
        for (const int id : input_ids)
        {
            if (!occurrence[id])
            {
                first[id] *= (first[id] > 0) ? inv_penalty : penalty;
            }
            occurrence[id] = true;
        }
    }

    void BaseModelForCausalLM::sampling_temperature(float *first, float *last, float temp)
    {
        const float inv_temp = 1.f / temp;
        for (float *it = first; it != last; it++)
        {
            *it *= inv_temp;
        }
    }

    void BaseModelForCausalLM::sampling_top_k(TokenIdScore *first, TokenIdScore *kth, TokenIdScore *last)
    {
        std::nth_element(first, kth, last, std::greater<TokenIdScore>());
    }

    TokenIdScore *BaseModelForCausalLM::sampling_top_p(TokenIdScore *first, TokenIdScore *last, float top_p)
    {
        // fast top_p in expected O(n) time complexity
        sampling_softmax_inplace(first, last);

        while (first + 1 < last)
        {
            const float pivot_score = (last - 1)->score; // use mid score?
            TokenIdScore *mid =
                std::partition(first, last - 1, [pivot_score](const TokenIdScore &x)
                               { return x.score > pivot_score; });
            std::swap(*mid, *(last - 1));

            const float prefix_sum =
                std::accumulate(first, mid, 0.f, [](float sum, const TokenIdScore &x)
                                { return sum + x.score; });
            if (prefix_sum >= top_p)
            {
                last = mid;
            }
            else if (prefix_sum + mid->score < top_p)
            {
                first = mid + 1;
                top_p -= prefix_sum + mid->score;
            }
            else
            {
                return mid + 1;
            }
        }
        return last;
    }

    void BaseModelForCausalLM::sampling_softmax_inplace(TokenIdScore *first, TokenIdScore *last)
    {
        float max_score = std::max_element(first, last)->score;
        float sum = 0.f;
        for (TokenIdScore *p = first; p != last; p++)
        {
            float s = std::exp(p->score - max_score);
            p->score = s;
            sum += s;
        }
        float inv_sum = 1.f / sum;
        for (TokenIdScore *p = first; p != last; p++)
        {
            p->score *= inv_sum;
        }
    }

    CogVLMForCausalLM::CogVLMForCausalLM(const ModelConfig *config) : BaseModelForCausalLM(config, MEM_SIZE, 0, num_weights(config->num_hidden_layers)),
                                                                      vocab_size(config->vocab_size),
                                                                      model(&ctx_, config),
                                                                      lm_head(&ctx_, config->hidden_size, config->vocab_size, false)
    {
        this->config = config;
    };

    StateDict CogVLMForCausalLM::state_dict() const
    {
        StateDict sd;
        sd.reserve(1189 + 20);
        sd.emplace_back("model.embed_tokens.weight", model.embed_tokens.weight);

        for (int i = 0; i < config->num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(i) + '.';
            sd.emplace_back(layer_prefix + "self_attn.vision_expert_query_key_value.weight",
                            model.layers[i].self_attn.vision_expert_query_key_value.weight);
            sd.emplace_back(layer_prefix + "self_attn.vision_expert_query_key_value.bias",
                            model.layers[i].self_attn.vision_expert_query_key_value.bias);

            sd.emplace_back(layer_prefix + "self_attn.vision_expert_dense.weight",
                            model.layers[i].self_attn.vision_expert_dense.weight);
            sd.emplace_back(layer_prefix + "self_attn.language_expert_query_key_value.weight",
                            model.layers[i].self_attn.language_expert_query_key_value.weight);
            sd.emplace_back(layer_prefix + "self_attn.language_expert_dense.weight",
                            model.layers[i].self_attn.language_expert_dense.weight);

            sd.emplace_back(layer_prefix + "mlp.language_mlp.gate_proj.weight",
                            model.layers[i].mlp.language_mlp.gate_proj.weight);
            sd.emplace_back(layer_prefix + "mlp.language_mlp.up_proj.weight",
                            model.layers[i].mlp.language_mlp.up_proj.weight);
            sd.emplace_back(layer_prefix + "mlp.language_mlp.down_proj.weight",
                            model.layers[i].mlp.language_mlp.down_proj.weight);
            sd.emplace_back(layer_prefix + "mlp.vision_mlp.gate_proj.weight",
                            model.layers[i].mlp.vision_mlp.gate_proj.weight);
            sd.emplace_back(layer_prefix + "mlp.vision_mlp.up_proj.weight",
                            model.layers[i].mlp.vision_mlp.up_proj.weight);
            sd.emplace_back(layer_prefix + "mlp.vision_mlp.down_proj.weight",
                            model.layers[i].mlp.vision_mlp.down_proj.weight);

            sd.emplace_back(layer_prefix + "input_layernorm.weight",
                            model.layers[i].input_layernorm.weight);
            sd.emplace_back(layer_prefix + "post_attention_layernorm.weight",
                            model.layers[i].post_attention_layernorm.weight);
        }

        sd.emplace_back("model.norm.weight", model.norm.weight);
        sd.emplace_back("model.vision.patch_embedding.proj.weight",
                        model.vision.patch_embedding.proj.weight);
        sd.emplace_back("model.vision.patch_embedding.proj.bias",
                        model.vision.patch_embedding.proj.bias);
        sd.emplace_back("model.vision.patch_embedding.position_embedding.weight",
                        model.vision.patch_embedding.position_embedding.weight);
        sd.emplace_back("model.vision.patch_embedding.cls_embedding",
                        model.vision.patch_embedding.cls_embedding);

        for (int i = 0; i < config->model_vision_config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.vision.transformer.layers." + std::to_string(i) + '.';
            sd.emplace_back(layer_prefix + "input_layernorm.weight",
                            model.vision.transformer.layers[i].input_layernorm.weight);
            sd.emplace_back(layer_prefix + "input_layernorm.bias",
                            model.vision.transformer.layers[i].input_layernorm.bias);
            sd.emplace_back(layer_prefix + "attention.query_key_value.bias",
                            model.vision.transformer.layers[i].attention.query_key_value.bias);
            sd.emplace_back(layer_prefix + "attention.query_key_value.weight",
                            model.vision.transformer.layers[i].attention.query_key_value.weight);
            sd.emplace_back(layer_prefix + "attention.dense.bias",
                            model.vision.transformer.layers[i].attention.dense.bias);
            sd.emplace_back(layer_prefix + "attention.dense.weight",
                            model.vision.transformer.layers[i].attention.dense.weight);
            sd.emplace_back(layer_prefix + "mlp.fc1.bias",
                            model.vision.transformer.layers[i].mlp.fc1.bias);
            sd.emplace_back(layer_prefix + "mlp.fc1.weight",
                            model.vision.transformer.layers[i].mlp.fc1.weight);
            sd.emplace_back(layer_prefix + "mlp.fc2.bias",
                            model.vision.transformer.layers[i].mlp.fc2.bias);
            sd.emplace_back(layer_prefix + "mlp.fc2.weight",
                            model.vision.transformer.layers[i].mlp.fc2.weight);
            sd.emplace_back(layer_prefix + "post_attention_layernorm.bias",
                            model.vision.transformer.layers[i].post_attention_layernorm.bias);
            sd.emplace_back(layer_prefix + "post_attention_layernorm.weight",
                            model.vision.transformer.layers[i].post_attention_layernorm.weight);
        }

        sd.emplace_back("model.vision.linear_proj.linear_proj.weight",
                        model.vision.linear_proj.linear_proj.weight);
        sd.emplace_back("model.vision.linear_proj.norm1.weight",
                        model.vision.linear_proj.norm1.weight);
        sd.emplace_back("model.vision.linear_proj.norm1.bias",
                        model.vision.linear_proj.norm1.bias);
        sd.emplace_back("model.vision.linear_proj.dense_h_to_4h.weight",
                        model.vision.linear_proj.dense_h_to_4h.weight);
        sd.emplace_back("model.vision.linear_proj.gate_proj.weight",
                        model.vision.linear_proj.gate_proj.weight);
        sd.emplace_back("model.vision.linear_proj.dense_4h_to_h.weight",
                        model.vision.linear_proj.dense_4h_to_h.weight);
        sd.emplace_back("model.vision.conv.weight",
                        model.vision.conv.weight);
        sd.emplace_back("model.vision.conv.bias",
                        model.vision.conv.bias);
        sd.emplace_back("model.vision.boi",
                        model.vision.boi);
        sd.emplace_back("model.vision.eoi",
                        model.vision.eoi);

        sd.emplace_back("lm_head.weight",
                        lm_head.weight);

        return sd;
    }

    void CogVLMForCausalLM::load(ModelLoader &loader)
    {
        // *ctx_.buffer = ggml_backend_alloc_ctx_tensors(ctx_.ctx_w.get(), *ctx_.backend);
        *ctx_.alloc = ggml_tallocr_new(*ctx_.buffer);
        state_dict_ = state_dict();
        for (auto &item : state_dict_)
        {
            void *data = loader.read_tensor(item.first, item.second);
            if(item.first[0]=='m')
                ggml_set_name(item.second, item.first.substr(5).c_str());
            else
                ggml_set_name(item.second, item.first.c_str());
            COGVLM_CHECK(item.second != NULL);
            // std::cout<<item.first<<":"<<item.second<<std::endl;
            ctx_.malloc(item.second, data, false);
        }
        // to_device();
        ctx_.weight_buffer = std::string_view(loader.data, loader.size);
        ctx_.init_device_context();
    }

    struct CausalLMOutputWithPast CogVLMForCausalLM::forward(
        ggml_tensor *input_ids,
        ggml_tensor *images,
        ggml_tensor *token_type_ids,
        int *attention_mask,
        ggml_tensor *position_ids,
        std::vector<ggml_tensor **> past_key_values,
        int n_ctx, bool use_cache)
    {
        BaseModelOutputWithPast output = model.forward(&ctx_, input_ids,
                                                       token_type_ids, attention_mask, images, position_ids,
                                                       past_key_values, n_ctx, use_cache);

        ggml_tensor *hidden_states = output.last_hidden_state;
        ggml_tensor *logits = lm_head.forward(&ctx_, hidden_states);
        PRINT_SHAPE("logits", logits);
        return CausalLMOutputWithPast{hidden_states, logits, output.past_key_values};
    }

    ggml_tensor *CogVLMForCausalLM::vision_forward(std::vector<float> buff)
    {
        int image_size = config->model_vision_config.image_size;
        ggml_tensor *images = ggml_new_tensor_3d(ctx_.ctx_temp,
                                                 GGML_TYPE_F32, image_size, image_size, 3);
        ctx_.malloc(images, buff.data());
        int n_threads = get_default_num_threads(); // default thread num
        ctx_.ctx_b = make_unique_ggml_context(ctx_.compute_buffer.size(), ctx_.compute_buffer.data(), true);
        ggml_tensor *result;
        result = model.vision.forward(&ctx_, images);
        PRINT_SHAPE("images", images);
        // ggml_gallocr_free(ctx_.galloc);
        // ctx_.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(*ctx_.backend));
        result = compute(result,
                         ctx_.galloc, *ctx_.backend, n_threads);
        printf("ctx_.alloc->offset=%ld\n", ctx_.alloc->offset);
        printf("success!\n");
        ctx_.ctx_b.get_deleter();
        return result;
    };

    CausalLMOutputWithPast CogVLMForCausalLM::forward_graph_compute(
        ggml_tensor *input_ids,
        ggml_tensor *images,
        ggml_tensor *token_type_ids,
        int *attention_mask,
        ggml_tensor *position_ids,
        std::vector<ggml_tensor **> past_key_values,
        int n_ctx, bool use_cache)
    {
        ctx_.ctx_b = make_unique_ggml_context(ctx_.compute_buffer.size(), ctx_.compute_buffer.data(), true);
        // ctx_.gf = ggml_new_graph_custom(ctx_.ctx_b.get(), 4096, false);
        int n_threads = get_default_num_threads(); // default thread num
        printf("n_threads %d\n", n_threads);
        if(ctx_.galloc==NULL)
        {
            ctx_.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(*ctx_.backend));
        }
        CausalLMOutputWithPast output = forward(
            input_ids, images, token_type_ids,
            attention_mask, position_ids, past_key_values, n_ctx, use_cache);
        ggml_tensor *lm_logits = output.logits;
        if(ctx_.galloc==NULL)
        {
            ctx_.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(*ctx_.backend));
            struct ggml_cgraph * gf = build_graph(lm_logits);
             // pre-allocate the compute buffer for the worst case (optional)
            ggml_gallocr_reserve(ctx_.galloc, gf);
            size_t mem_size =  ggml_gallocr_get_buffer_size(ctx_.galloc, 0);
            fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0/1024.0);
        }
        ggml_tensor *output_ = compute(lm_logits,
                                       ctx_.galloc, *ctx_.backend, n_threads);
        output.logits = output_;
        // ctx_.ctx_b.get_deleter();
        return output;
    }

    GenerateOutput CogVLMForCausalLM::generate_next_token(
        std::vector<int> &input_ids_vector,
        ggml_tensor *images,
        std::vector<int> &token_type_ids,
        int *attention_mask,
        ggml_tensor *position_ids,
        std::vector<ggml_tensor **> past_key_values,
        int n_ctx, bool use_cache)
    {
        // set CogVLM input
        ggml_tensor *input_ids = ggml_new_tensor_1d(ctx_.ctx_temp,
                                                    GGML_TYPE_I32, input_ids_vector.size());
        ctx_.malloc(input_ids, input_ids_vector.data());

        ggml_tensor *token_type_ids_tensor = ggml_new_tensor_1d(ctx_.ctx_temp,
                                                                GGML_TYPE_I32, token_type_ids.size());
        ctx_.malloc(token_type_ids_tensor, token_type_ids.data());
        CausalLMOutputWithPast output = forward_graph_compute(
            input_ids, images, token_type_ids_tensor,
            attention_mask, position_ids, past_key_values, n_ctx, true);

        ggml_tensor *lm_logits = output.logits;
        PRINT_SHAPE("lm_logits", lm_logits);
        printf("gen_config.do_sample %d\n", gen_config.do_sample);
        int vocab_size = lm_logits->ne[0];
        float *next_token_logits = new float[vocab_size];
        ggml_backend_tensor_get(lm_logits, (void *)next_token_logits,
                                ggml_nbytes(lm_logits) - vocab_size * ggml_type_size(lm_logits->type),
                                vocab_size * ggml_type_size(lm_logits->type));
        // check nan
        for (int i = 0; i < vocab_size; i++)
        {

            COGVLM_CHECK(std::isfinite(next_token_logits[i])) << "nan/inf encountered at lm_logits[" << i << "]";
        }

        // logits pre-process
        if (gen_config.repetition_penalty != 1.f)
        {
            sampling_repetition_penalty(next_token_logits, next_token_logits + vocab_size, input_ids_vector,
                                        gen_config.repetition_penalty);
        }

        int next_token_id;

        if (gen_config.do_sample)
        {
            // temperature sampling
            if (gen_config.temperature > 0)
            {
                sampling_temperature(next_token_logits, next_token_logits + vocab_size, gen_config.temperature);
            }

            std::vector<TokenIdScore> token_scores(vocab_size);
            for (int i = 0; i < vocab_size; i++)
            {
                token_scores[i] = TokenIdScore(i, next_token_logits[i]);
            }

            // top_k sampling
            if (0 < gen_config.top_k && gen_config.top_k < (int)token_scores.size())
            {
                sampling_top_k(token_scores.data(), token_scores.data() + gen_config.top_k,
                               token_scores.data() + token_scores.size());
                token_scores.resize(gen_config.top_k);
            }

            // top_p sampling
            if (0.f < gen_config.top_p && gen_config.top_p < 1.f)
            {
                auto pos = sampling_top_p(token_scores.data(), token_scores.data() + token_scores.size(), gen_config.top_p);
                token_scores.resize(pos - token_scores.data());
            }

            // sample next token
            sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
            for (size_t i = 0; i < token_scores.size(); i++)
            {
                next_token_logits[i] = token_scores[i].score;
            }

            thread_local std::random_device rd;
            thread_local std::mt19937 gen(rd());

            std::discrete_distribution<> dist(next_token_logits, next_token_logits + token_scores.size());
            next_token_id = token_scores[dist(gen)].id;
        }
        else
        {
            // greedy search
            next_token_id = std::max_element(next_token_logits, next_token_logits + vocab_size) - next_token_logits;
        }

        // int next_token_id;
        return GenerateOutput{output.past_key_values,next_token_id};
    }

    std::vector<int> CogVLMForCausalLM::generate(std::vector<int> &text_ids,
                                                 std::vector<float> &images_vector)
    {
        std::vector<int> input_ids_vector;
        std::vector<int> token_type_ids;
        input_ids_vector.emplace_back(config->bos_token_id);
        token_type_ids.clear();
        token_type_ids.emplace_back(LANGUAGE_TOKEN_TYPE);

        if (images_vector.size() > 0)
        {
            int image_size = config->model_vision_config.image_size;
            int patch_size = config->model_vision_config.patch_size;
            int vision_token_num = (image_size / patch_size / 2) * (image_size / patch_size / 2) + 2;
            token_type_ids.insert(token_type_ids.end(), vision_token_num, VISION_TOKEN_TYPE);
        }

        input_ids_vector.insert(input_ids_vector.end(), text_ids.begin(), text_ids.end());
        token_type_ids.insert(token_type_ids.end(), text_ids.size(), LANGUAGE_TOKEN_TYPE);
        const int n_ctx = token_type_ids.size();
        std::vector<int> attention_mask_vector = std::vector<int>(token_type_ids.size(), 1);
        printf("CogVLMForCausalLM::generate\n");
        ggml_tensor *position_ids = build_position_ids(&ctx_, token_type_ids, attention_mask_vector);

        printf("token_type_ids sum = %d\n", std::accumulate(token_type_ids.begin(), token_type_ids.end(), 0));
        
        ggml_tensor *images = nullptr;
        if (images_vector.size() > 0)
        {
            int image_size = config->model_vision_config.image_size;
            images = ggml_new_tensor_3d(ctx_.ctx_temp,
                    GGML_TYPE_F32, image_size, image_size, 3);
            ctx_.malloc(images, images_vector.data());
        }
        

        int *attention_mask = attention_mask_vector.data();

        std::vector<int> output_ids;
        output_ids.reserve(gen_config.max_length - n_ctx);

        std::vector<ggml_tensor **> past_key_values;
        while ((int)output_ids.size() < gen_config.max_length - n_ctx)
        {
            GenerateOutput output = generate_next_token(
                input_ids_vector, images, token_type_ids,
                attention_mask, position_ids, past_key_values, n_ctx, false);
            
            int next_token_id = output.next_token_id;
            // past_key_values = output.past_key_values;
            
            position_ids = nullptr;
            images = nullptr;
            position_ids = nullptr;
            input_ids_vector.clear();
            input_ids_vector.emplace_back(next_token_id);
            token_type_ids.clear();
            token_type_ids.emplace_back(LANGUAGE_TOKEN_TYPE);
            output_ids.emplace_back(next_token_id);
            printf("next_toekn %d\n", next_token_id);
            if (next_token_id == config->eos_token_id[0] ||
                next_token_id == config->eos_token_id[1] ||
                std::find(config->extra_eos_token_ids.begin(), config->extra_eos_token_ids.end(), next_token_id) !=
                    config->extra_eos_token_ids.end())
            {
                break;
            }
        }

        return output_ids;
    }

    // ===== pipeline =====
    Pipeline::Pipeline(const std::string &path)
    {
        mapped_file = std::make_unique<MappedFile>(path);
        ModelLoader loader(mapped_file->data, mapped_file->size);
        // load magic
        std::string magic = loader.read_string(4);
        COGVLM_CHECK(magic == "ggml")
            << "model file is broken (bad magic)";

        // load model type
        ModelType model_type = (ModelType)loader.read_basic<int>();
        COGVLM_CHECK(model_type == ModelType::COGVLM2)
            << "model type is not COGVLM2";
        // load version
        int version = loader.read_basic<int>();

        if (model_type == ModelType::COGVLM2)
        {
            // load config
            COGVLM_CHECK(version == 1)
                << "only support version 1 for now but got " << version;
            config = new ModelConfig(
                model_type, loader.read_basic<ConfigRecord>());
            std::cout << config->model_to_string() << std::endl;

            // load model
            const int image_size = config->model_vision_config.image_size;
            this->model = std::make_unique<cogvlm::CogVLMForCausalLM>(config);
            printf("model load\n");
            this->model->load(loader);
            // ggml_tensor *x = this->model->vision_forward(images);
            // PRINT_SHAPE("images", images);
            // PRINT_SHAPE("x", x);
        }
    }
}