#ifndef ENUMS_H
#define ENUMS_H
enum class ModelType
{
    CHATGLM = 1,
    CHATGLM2 = 2,
    CHATGLM3 = 3,
    COGVLM2 = 33,
    BAICHUAN7B = 1024,
    BAICHUAN13B = 1025,
    INTERNLM = 1280,
};

enum class ActivationType
{
    GELU,
    SILU,
};

enum RopeType {
    ROPE_TYPE_DEFAULT = 0,
    ROPE_TYPE_NEOX = 2,
    ROPE_TYPE_CHATGLM = 4,
};
std::string to_string(ActivationType activation_type);
std::string to_string(ModelType modelType);
// std::string to_string(RopeType ropeType)
// {
//     switch (ropeType)
//     {
//     case RopeType::ROPE_TYPE_DEFAULT:
//         return "ROPE_TYPE_DEFAULT";
//     case RopeType::ROPE_TYPE_NEOX:
//         return "ROPE_TYPE_NEOX";
//     case RopeType::ROPE_TYPE_CHATGLM:
//         return "ROPE_TYPE_CHATGLM";
//     default:
//         return "UNKNOWN";
//     }
// }
#endif