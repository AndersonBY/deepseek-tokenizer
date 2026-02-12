# DeepSeek Tokenizer

[English](README.md) | 中文

## 介绍

DeepSeek Tokenizer 是一个高效且轻量级的分词库，运行时不依赖任何第三方库，是分词任务中更精简和高效的选择。


## 安装

要安装 DeepSeek Tokenizer，请使用以下命令：

```bash
pip install deepseek_tokenizer
```


## 基本用法

下面是一个简单的例子，演示了如何使用 DeepSeek Tokenizer 来计算文本的 Token：

```python
from deepseek_tokenizer import ds_token

# 示例文本
text = "Hello! 毕老师！1 + 1 = 2 ĠÑĤÐ²ÑĬÑĢ"

# 编码文本
result = ds_token.encode(text)

# 打印结果
print(result)
```


### 输出

```
[19923, 3, 223, 5464, 5008, 1175, 19, 940, 223, 19, 438, 223, 20, 6113, 257, 76589, 131, 100, 76032, 1628, 76589, 131, 108, 76589, 131, 98]
```


## 许可

本项目采用 MIT 许可证。
