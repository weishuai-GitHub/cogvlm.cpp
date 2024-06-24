#include "loader.h"
#include "common.h"
#include <algorithm>
#include <codecvt>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <locale>
#include <numeric>
#include <random>
#include <regex>
#include <string>
#include <sys/stat.h>
#include <thread>
#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif
#ifdef _POSIX_MAPPED_FILES

MappedFile::MappedFile(const std::string &path)
{
    int fd = open(path.c_str(), O_RDONLY);
    COGVLM_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

    struct stat sb;
    COGVLM_CHECK(fstat(fd, &sb) == 0) << strerror(errno);
    size = sb.st_size;

    data = (char *)mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    COGVLM_CHECK(data != MAP_FAILED) << strerror(errno);

    COGVLM_CHECK(close(fd) == 0) << strerror(errno);
}

MappedFile::~MappedFile() { COGVLM_CHECK(munmap(data, size) == 0) << strerror(errno); }

#elif defined(_WIN32)
MappedFile::MappedFile(const std::string &path)
{

    int fd = open(path.c_str(), O_RDONLY);
    COGVLM_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

    struct _stat64 sb;
    COGVLM_CHECK(_fstat64(fd, &sb) == 0) << strerror(errno);
    size = sb.st_size;

    HANDLE hFile = (HANDLE)_get_osfhandle(fd);

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    COGVLM_CHECK(hMapping != NULL) << strerror(errno);

    data = (char *)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);

    COGVLM_CHECK(data != NULL) << strerror(errno);

    COGVLM_CHECK(close(fd) == 0) << strerror(errno);
}

MappedFile::~MappedFile() { COGVLM_CHECK(UnmapViewOfFile(data)) << strerror(errno); }
#endif

void ModelLoader::seek(int64_t offset, int whence)
{
    if (whence == SEEK_SET)
    {
        ptr = data + offset;
    }
    else if (whence == SEEK_CUR)
    {
        ptr += offset;
    }
    else if (whence == SEEK_END)
    {
        ptr = data + size + offset;
    }
    else
    {
        CHATGLM_THROW << "invalid seek mode " << whence;
    }
}

std::string ModelLoader::read_string(size_t length)
{
    std::string s(ptr, ptr + length);
    ptr += length;
    return s;
}

void ModelLoader::checked_read_tensor_meta(const std::string &name, int target_ndim, int64_t *target_ne,
                                           ggml_type target_dtype)
{
    // read and check tensor name
    {
        int name_size = read_basic<int>();
        COGVLM_CHECK(name_size == (int)name.size())
            << "tensor " << name << " name size mismatch: expect " << name.size() << " but got " << name_size;
        std::string weight_name = read_string(name_size);
        COGVLM_CHECK(weight_name == name) << "tensor name mismatch: expect " << name << " but got " << weight_name;
    }

    // read and check tensor shape
    {
        int ndim = read_basic<int>();
        COGVLM_CHECK(ndim == target_ndim)
            << "tensor " << name << " ndim mismatch: expect " << target_ndim << " but got " << ndim;
        for (int i = ndim - 1; i >= 0; i--)
        {
            int dim_size = read_basic<int>();
            COGVLM_CHECK(dim_size == target_ne[i]) << "tensor " << name << " shape mismatch at dim " << i
                                                    << ": expect " << target_ne[i] << " but got " << dim_size;
        }
    }

    // read and check tensor dtype
    {
        ggml_type dtype = (ggml_type)read_basic<int>();
        COGVLM_CHECK(dtype == target_dtype)
            << "tensor " << name << " dtype mismatch: expect " << target_dtype << " but got " << dtype;
    }
}

void *ModelLoader::read_tensor_data(size_t nbytes) {
    constexpr int64_t MEM_ALIGNED = 16;
    const int64_t data_offset = (tell() + (MEM_ALIGNED - 1)) & ~(MEM_ALIGNED - 1);
    void *tensor_data = data + data_offset;
    seek(data_offset + nbytes, SEEK_SET);
    return tensor_data;
}

void* ModelLoader::read_tensor(const std::string &name, ggml_tensor *tensor) {
    checked_read_tensor_meta(name, ggml_n_dims(tensor), tensor->ne, tensor->type);
    void* data = read_tensor_data(ggml_nbytes(tensor));
    return data;
}
