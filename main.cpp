#include<cogvlm.h>
#include<iostream>
#include <fstream>
#include"visual.h"
int main()
{
    cogvlm::Pipeline pipeline("cogvlm-ggml.bin");
    std::vector<int> text_ids = { 14924, 25, 91967, 66201, 9554, 44915, 22559, 25};
    std::ifstream infile("image.bin");
    if (!infile.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    // 获取文件大小
    infile.seekg(0, std::ios::end);
    std::streamsize fileSize = infile.tellg();
    infile.seekg(0, std::ios::beg);
    // 计算浮点数的数量
    std::size_t numFloats = fileSize / sizeof(float);

    // 创建向量来保存图像数据
    std::vector<float> buffer(numFloats);
     // 读取文件内容到缓冲区
    if (infile.read(reinterpret_cast<char*>(buffer.data()), fileSize)) {
        std::cout << "Image data read successfully!" << std::endl;
    } else {
        std::cerr << "Error reading the file!" << std::endl;
        return 1;
    }
    pipeline.generate(text_ids, buffer);
    // ggml_tensor* result = pipeline.model->vision_forward(buffer);
    // PRINT_SHAPE("result",result);
    return 0;
}