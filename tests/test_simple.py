# @Author: Bi Ying
# @Date:   2024-08-14 15:49:21
from deepseek_tokenizer import ds_token


text = "Hello! 毕老师！1 + 1 = 2 ĠÑĤÐ²ÑĬÑĢ"
expected = [
    19923,
    3,
    223,
    5464,
    5008,
    1175,
    19,
    940,
    223,
    19,
    438,
    223,
    20,
    6113,
    257,
    76589,
    131,
    100,
    76032,
    1628,
    76589,
    131,
    108,
    76589,
    131,
    98,
]

result = ds_token.encode(text)
assert result == expected, f"unexpected encoding: {result}"
