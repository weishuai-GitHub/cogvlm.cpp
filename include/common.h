#ifndef COMMON_H
#define COMMON_H
#include <ggml.h>
#include <iomanip>
#include <sstream>
#include <vector>
#include <unordered_map>

// ===== common =====
std::string to_string(ggml_tensor *tensor, bool with_data = true);

ggml_tensor *tensor_assign_buffers(ggml_tensor *tensor);

ggml_tensor *tensor_to_device(ggml_tensor *tensor);

ggml_tensor *tensor_to_cpu(ggml_tensor *tensor);

// reference: https://stackoverflow.com/questions/11149665/c-vector-that-doesnt-initialize-its-members
struct uninitialized_char
{
    char m;
    uninitialized_char() {}
};

void ggml_graph_compute_helper(std::vector<uninitialized_char> &buf, ggml_cgraph *graph, int n_threads);

// Adapted from https://github.com/ggerganov/llama.cpp/blob/master/examples/common.cpp
int get_num_physical_cores();

int get_default_num_threads();
#endif