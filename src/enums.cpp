#include <string>
#include "enums.h"

std::string to_string(ActivationType activation_type)
{
    switch (activation_type)
    {
    case ActivationType::GELU:
        return "GELU";
    case ActivationType::SILU:
        return "SILU";
    default:
        return "UNKNOWN";
    }
}
std::string to_string(ModelType modelType)
{
    switch (modelType)
    {
    case ModelType::CHATGLM:
        return "CHATGLM";
    case ModelType::CHATGLM2:
        return "CHATGLM2";
    case ModelType::CHATGLM3:
        return "CHATGLM3";
    case ModelType::COGVLM2:
        return "COGVLM2";
    case ModelType::BAICHUAN7B:
        return "BAICHUAN7B";
    case ModelType::BAICHUAN13B:
        return "BAICHUAN13B";
    case ModelType::INTERNLM:
        return "INTERNLM";
    default:
        return "UNKNOWN";
    }
}