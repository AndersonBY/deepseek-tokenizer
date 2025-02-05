# @Author: Bi Ying
# @Date:   2024-08-14 15:49:21
from deepseek_tokenizer import ds_token


chat_tokenizer_dir = "./"
text = "Hello! 毕老师！1 + 1 = 2 ĠÑĤÐ²ÑĬÑĢ"


result = ds_token.encode(text)
print(result)
print(f"len(result): {len(result)}")
