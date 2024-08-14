# @Author: Bi Ying
# @Date:   2024-08-14 15:49:21
from deepseek_tokenizer import deepseek_tokenizer


chat_tokenizer_dir = "./"
text = "Hello! 毕老师！1 + 1 = 2 ĠÑĤÐ²ÑĬÑĢ"


result = deepseek_tokenizer.encode(text)
print(result)
