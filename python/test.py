import torch
def f(x,n):
    return (((x) + (n) - 1) & ~((n) - 1))
print(f(2315,32))
# # 创建一个四维张量
# cls_embedding = torch.randn(2, 3)
# # 使用 flatten(2) 函数
# cls_token  = cls_embedding.reshape(3,2)
# cls_token[2][0]=10
# print(cls_embedding)  # 输出: torch.Size([2, 3, 4, 5])
# print(cls_token)  # 输出: torch.Size([2, 3, 20]