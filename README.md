## 部署步骤

1. 将模型转化成ggml.bin格式
2. 初始化设备，指定设备号，分配context,backend的内存
3. 构建model框架
4. 构建计算图
5. 推理

## 运行
```shell
mkdir build
cd build && cmake .. && make
cd  ..
./build/bin/main
```