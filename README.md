# DeepSeek Tokenizer

English | [中文](README_ZH.md)

## Introduction

DeepSeek Tokenizer is an efficient and lightweight tokenization library with no third-party runtime dependencies, making it a streamlined and efficient choice for tokenization tasks.

This release uses the DeepSeek-V4 series tokenizer shared by `deepseek-ai/DeepSeek-V4-Pro` and `deepseek-ai/DeepSeek-V4-Flash`. The bundled tokenizer supports the V4 special tokens, including thinking, DSML, quick-instruction, and multimodal placeholder tokens, with a `model_max_length` of 1,048,576 tokens.

## Installation

To install DeepSeek Tokenizer, use the following command:

```bash
pip install deepseek_tokenizer
```

## Basic Usage

Below is a simple example demonstrating how to use DeepSeek Tokenizer to encode text:

```python
from deepseek_tokenizer import ds_token

# Sample text
text = "Hello! 毕老师！1 + 1 = 2 ĠÑĤÐ²ÑĬÑĢ"

# Encode text
result = ds_token.encode(text)

# Print result
print(result)
```

### Output

```
[19923, 3, 223, 5464, 5008, 1175, 19, 940, 223, 19, 438, 223, 20, 6113, 257, 76589, 131, 100, 76032, 1628, 76589, 131, 108, 76589, 131, 98]
```

## DeepSeek-V4 Tokens

```python
from deepseek_tokenizer import ds_token

print(ds_token.model_max_length)
print(ds_token.convert_tokens_to_ids("<think>"))
print(ds_token.convert_tokens_to_ids("</think>"))
print(ds_token.convert_tokens_to_ids("｜DSML｜"))
print(ds_token.convert_tokens_to_ids("<｜latest_reminder｜>"))
```

### Output

```
1048576
128821
128822
128825
128828
```

## License

This project is licensed under the MIT License.
