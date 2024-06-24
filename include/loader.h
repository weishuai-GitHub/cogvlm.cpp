#ifndef LOADER_H
#define LOADER_H
#include <ggml.h>
#include <iomanip>
#include <vector>
#include <unordered_map>

class MappedFile
{
public:
    MappedFile(const std::string &path);
    ~MappedFile();
    int64_t tell() const { return ptr - data; }

    void seek(int64_t offset, int whence);

    template <typename T>
    T read_basic()
    {
        T obj = *(T *)ptr;
        ptr += sizeof(T);
        return obj;
    }

    std::string read_string(size_t length);

    void checked_read_tensor_meta(const std::string &name, int ndim, int64_t *ne, ggml_type dtype);

    void *read_tensor_data(size_t nbytes);

    void read_tensor(const std::string &name, ggml_tensor *tensor);

public:
    char *data;
    size_t size;
    char *ptr;
};

// ===== model_loader =====
class ModelLoader
{
public:
    ModelLoader(char *data, size_t size) : data(data), size(size), ptr(data) {}

    int64_t tell() const { return ptr - data; }

    void seek(int64_t offset, int whence);

    template <typename T>
    T read_basic()
    {
        T obj = *(T *)ptr;
        ptr += sizeof(T);
        return obj;
    }

    std::string read_string(size_t length);

    void checked_read_tensor_meta(const std::string &name, int ndim, int64_t *ne, ggml_type dtype);

    void *read_tensor_data(size_t nbytes);

    void *read_tensor(const std::string &name, ggml_tensor *tensor);

public:
    char *data;
    size_t size;
    char *ptr;
};

static constexpr size_t MB = 1024 * 1024;

class LogMessageFatal
{
public:
    LogMessageFatal(const char *file, int line) { oss_ << file << ':' << line << ' '; }
    [[noreturn]] ~LogMessageFatal() noexcept(false) { throw std::runtime_error(oss_.str()); }
    std::ostringstream &stream() { return oss_; }

private:
    std::ostringstream oss_;
};
#define CHATGLM_THROW LogMessageFatal(__FILE__, __LINE__).stream()
#define COGVLM_CHECK(cond) \
    if (!(cond))            \
    CHATGLM_THROW << "check failed (" #cond ") "
#define COGVLM_CHECK_CUDA(call)                                                            \
    do                                                                                      \
    {                                                                                       \
        cudaError_t error = (call);                                                         \
        COGVLM_CHECK(error == cudaSuccess) << "CUDA error: " << cudaGetErrorString(error); \
    } while (0)
# define PRINT_SHAPE(name,tensor) \
    std::cout << name << ".shape: ["; \
    for (int i = 0; i < 4; i++) \
    { \
        if(i<3) std::cout << tensor->ne[i] << ","; \
        else std::cout << tensor->ne[i]; \
    } \
    std::cout <<"]"<<std::endl;

#endif