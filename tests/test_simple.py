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


assert ds_token.model_max_length == 1048576
assert ds_token.vocab_size == 129283

expected_special_tokens = {
    "<think>": 128821,
    "</think>": 128822,
    "｜DSML｜": 128825,
    "<｜latest_reminder｜>": 128828,
    "<｜action｜>": 128829,
    "<｜query｜>": 128830,
    "<｜authority｜>": 128831,
    "<｜domain｜>": 128832,
    "<｜title｜>": 128836,
    "<｜read_url｜>": 128845,
    "<｜image｜>": 129279,
}

for token, token_id in expected_special_tokens.items():
    assert ds_token.convert_tokens_to_ids(token) == token_id

v4_prompt = (
    "<｜User｜>你好<｜Assistant｜><think>推理</think>答案"
    "<｜end▁of▁sentence｜>"
)
expected_v4_prompt = [128803, 30594, 128804, 128821, 36101, 128822, 7624, 1]
assert ds_token.encode(v4_prompt) == expected_v4_prompt
assert ds_token.decode(expected_v4_prompt) == v4_prompt
