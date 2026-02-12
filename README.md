# DeepSeek Tokenizer

English | [中文](README_ZH.md)

## Introduction

DeepSeek Tokenizer is an efficient and lightweight tokenization library with no third-party runtime dependencies, making it a streamlined and efficient choice for tokenization tasks.

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

## License

This project is licensed under the MIT License.
