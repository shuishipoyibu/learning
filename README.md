# Beginner FlashAttention CUDA

这是一个精简 FlashAttention 入门项目，保留两条核心主线：

- `standard_decode_attention`
- `flash_attention（split+combine）` 
## 具体内容
- `standard_decode_attention`：采取片上动态softmax，每新来一个token，算出score并动态更新全局最大值，并动态更新softmax函数分母的值，以及旧有token的加权attention结果。
- `flash_attention`：先对token序列进行分块，进行分块的softmax计算，并算出分块的加权输出，然后进行combine，得出每个分块的权重分数，计算出最后的输出。
- 最后还有采取pytorch官方实现的一个版本，这个无cuda代码了。
  
## 整体项目
- 一个没训练过的简易的小模型，mini_model.py,采取了CUDA代码中的flash_attention部分，可以在demo_mini_model.py中运行推理。
- 一个测试脚本，benchmark.py, 其中包含了上述三种attention的测试的性能对比。
- test_correctness.py, 用于测试三种路径的输出结果的一致性。
## 最核心文件

```text
beginner_flashattention_cuda/
  README.md
  beginner_flashatt/
    __init__.py
    flashattention.cu
    mini_model.py
  tests/
    test_correctness.py
    test_input_validation.py
    test_mini_model.py
  demo_mini_model.py
```

`flashattention.cu` 是核心 CUDA 实现，`mini_model.py` 负责把 attention 接进最小模型，`tests/` 负责验证，`demo_mini_model.py` 负责端到端演示。
