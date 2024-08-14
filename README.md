# DeepSeek Tokenizer

## Introduction

DeepSeek Tokenizer is an efficient and lightweight tokenization libraries which doesn't require heavy dependencies like the `transformers` library, DeepSeek Tokenizer solely relies on the `tokenizers` library, making it a more streamlined and efficient choice for tokenization tasks.

## Installation

To install DeepSeek Tokenizer, use the following command:

```bash
pip install deepseek_tokenizer
```

## Basic Usage

Below is a simple example demonstrating how to use DeepSeek Tokenizer to encode text:

```python
from deepseek_tokenizer import deepseek_tokenizer

# Sample text
text = "Hello! 毕老师！1 + 1 = 2 ĠÑĤÐ²ÑĬÑĢ"

# Encode text
result = deepseek_tokenizer.encode(text)

# Print result
print(result)
```

### Output

```
[17464, 0, 207, 7689, 7842, 2160, 16, 919, 207, 16, 403, 207, 17, 20877, 241, 65469, 126, 97, 70795, 12094, 65469, 126, 105, 65469, 126, 95]
```

## License

This project is licensed under the MIT License.
