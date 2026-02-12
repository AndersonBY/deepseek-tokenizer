import json
from pathlib import Path
from typing import List, Optional, Union

from .py_tokenizer import PurePythonTokenizer


BASE_FOLDER = Path(__file__).parent


class DeepSeekTokenizer:
    def __init__(self, tokenizer: PurePythonTokenizer, model_max_length: int = int(1e30)):
        self._tokenizer = tokenizer
        self.model_max_length = model_max_length

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path]):
        path = Path(pretrained_model_name_or_path)
        tokenizer = PurePythonTokenizer(path / "tokenizer.json")

        # 从配置文件中获取model_max_length
        config_file = path / "tokenizer_config.json"
        model_max_length = int(1e30)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                model_max_length = config.get("model_max_length", model_max_length)

        return cls(tokenizer, model_max_length=model_max_length)

    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], List[List[int]]]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return self._clean_up(text) if clean_up_tokenization_spaces else text

    def _clean_up(self, text: str) -> str:
        return text.replace("Ġ", " ").replace(" ", " ").strip()

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[Optional[int], List[Optional[int]]]:
        if isinstance(tokens, str):
            return self._tokenizer.token_to_id(tokens)
        return [self._tokenizer.token_to_id(t) for t in tokens]

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self._tokenizer.id_to_token_str(ids) or ""
        return [self._tokenizer.id_to_token_str(i) or "" for i in ids]


ds_token = deepseek_tokenizer = DeepSeekTokenizer.from_pretrained(BASE_FOLDER)
