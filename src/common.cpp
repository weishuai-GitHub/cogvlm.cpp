#include "common.h"
#include "loader.h"
#include <thread>

#ifdef GGML_USE_CUBLAS
#include <ggml-cuda.h>
#endif
static std::string shape_to_string(ggml_tensor *tensor)
{
    std::ostringstream oss;
    oss << '[';
    for (int i = ggml_n_dims(tensor) - 1; i >= 0; i--)
    {
        oss << tensor->ne[i] << (i > 0 ? ", " : "");
    }
    oss << ']';
    return oss.str();
}

static std::string strides_to_string(ggml_tensor *tensor)
{
    std::ostringstream oss;
    oss << '[';
    for (int i = ggml_n_dims(tensor) - 1; i >= 0; i--)
    {
        oss << tensor->nb[i] << (i > 0 ? ", " : "");
    }
    oss << ']';
    return oss.str();
}

std::string to_string(ggml_tensor *tensor, bool with_data)
{
    std::ostringstream oss;
    oss << "ggml_tensor(";

    if (with_data)
    {
        if (ggml_n_dims(tensor) > 3)
            oss << "[";
        for (int i3 = 0; i3 < tensor->ne[3]; i3++)
        {
            if (ggml_n_dims(tensor) > 2)
                oss << (i3 > 0 ? ",\n\n[" : "[");
            for (int i2 = 0; i2 < tensor->ne[2]; i2++)
            {
                if (ggml_n_dims(tensor) > 1)
                    oss << (i2 > 0 ? ",\n\n[" : "[");
                for (int i1 = 0; i1 < tensor->ne[1]; i1++)
                {
                    oss << (i1 > 0 ? ",\n[" : "[");
                    for (int i0 = 0; i0 < tensor->ne[0]; i0++)
                    {
                        auto ptr = (char *)tensor->data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] +
                                   i0 * tensor->nb[0];
                        oss << (i0 > 0 ? ", " : "");
                        if (tensor->type == GGML_TYPE_I32)
                        {
                            oss << *(int *)ptr;
                        }
                        else
                        {
                            float val;
                            if (tensor->type == GGML_TYPE_F32)
                            {
                                val = *(float *)ptr;
                            }
                            else if (tensor->type == GGML_TYPE_F16)
                            {
                                val = ggml_fp16_to_fp32(*(ggml_fp16_t *)ptr);
                            }
                            else
                            {
                                CHATGLM_THROW << "unimplemented";
                            }
                            oss << std::setw(7) << std::fixed << std::setprecision(4) << val;
                        }
                    }
                    oss << "]";
                }
                if (ggml_n_dims(tensor) > 1)
                    oss << "]";
            }
            if (ggml_n_dims(tensor) > 2)
                oss << "]";
        }
        if (ggml_n_dims(tensor) > 3)
            oss << "]";
        oss << ", ";
    }

    oss << "shape=" << shape_to_string(tensor) << ", stride=" << strides_to_string(tensor) << ")";
    return oss.str();
}

ggml_tensor *tensor_assign_buffers(ggml_tensor *tensor)
{
// #ifdef GGML_USE_CUBLAS
//     ggml_cuda_assign_buffers(tensor);
// #endif
    return tensor;
}

ggml_tensor *tensor_to_device(ggml_tensor *tensor)
{
#ifdef GGML_USE_CUBLAS
    if (tensor->backend == GGML_BACKEND_TYPE_CPU)
    {
        tensor->backend = GGML_BACKEND_TYPE_GPU;
        ggml_backend_tensor_set(tensor,tensor->data,0, ggml_nbytes(tensor));
    }
#endif
    return tensor;
}

ggml_tensor *tensor_to_cpu(ggml_tensor *tensor)
{
#ifdef GGML_USE_CUBLAS
    if (tensor->backend != GGML_BACKEND_TYPE_CPU)
    {
        ggml_backend_tensor_get(tensor,tensor->data,0, ggml_nbytes(tensor));
        tensor->backend = GGML_BACKEND_TYPE_CPU;
    }
#endif
    return tensor;
}

// Adapted from https://github.com/ggerganov/llama.cpp/blob/master/llama.cpp
void ggml_graph_compute_helper(std::vector<uninitialized_char> &buf, ggml_cgraph *graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);
    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = (uint8_t *)buf.data();
    }
    printf("plan.work_size %d\n",plan.work_size);
    // ggml_status status = ggml_graph_compute(graph, &plan);
    ggml_status status = ggml_graph_compute(graph, &plan);
}
// Adapted from https://github.com/ggerganov/llama.cpp/blob/master/examples/common.cpp

int get_num_physical_cores() {
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

int get_default_num_threads() {
#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_METAL)
    return 1;
#else
    return std::min(get_num_physical_cores(), 16);
#endif
}