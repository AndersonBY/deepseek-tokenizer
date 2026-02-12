import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _bytes_to_unicode() -> Dict[int, str]:
    # Mirrors tokenizers' byte-level mapping (GPT-2 byte encoder).
    bs = list(range(ord("!"), ord("~") + 1))
    bs.extend(range(0xA1, 0xAC + 1))
    bs.extend(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTES_TO_UNICODE: Dict[int, str] = _bytes_to_unicode()
_UNICODE_TO_BYTES: Dict[str, int] = {v: k for k, v in _BYTES_TO_UNICODE.items()}


def _byte_level_encode(text: str) -> str:
    return "".join(_BYTES_TO_UNICODE[b] for b in text.encode("utf-8"))


def _byte_level_decode(tokens: Sequence[str]) -> str:
    out_bytes: List[int] = []
    for token in tokens:
        # If any char is not in the byte-level alphabet, fall back to raw bytes.
        token_bytes: Optional[List[int]] = []
        for ch in token:
            b = _UNICODE_TO_BYTES.get(ch)
            if b is None:
                token_bytes = None
                break
            token_bytes.append(b)
        if token_bytes is None:
            out_bytes.extend(token.encode("utf-8"))
        else:
            out_bytes.extend(token_bytes)
    return bytes(out_bytes).decode("utf-8", errors="replace")


def _cat(ch: str) -> str:
    return unicodedata.category(ch)


def _is_letter(ch: str) -> bool:
    return _cat(ch).startswith("L")


def _is_mark(ch: str) -> bool:
    return _cat(ch).startswith("M")


def _is_number(ch: str) -> bool:
    return _cat(ch).startswith("N")


def _is_punct(ch: str) -> bool:
    return _cat(ch).startswith("P")


def _is_symbol(ch: str) -> bool:
    return _cat(ch).startswith("S")


def _is_ascii_letter(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


_PUNCT_SET = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FA5
        or 0x3040 <= code <= 0x309F
        or 0x30A0 <= code <= 0x30FF
    )


def _split_isolated(text: str, match_len_fn) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    i = 0
    last = 0
    n = len(text)
    while i < n:
        mlen = match_len_fn(text, i)
        if mlen:
            if last < i:
                out.append(text[last:i])
            out.append(text[i : i + mlen])
            i += mlen
            last = i
        else:
            i += 1
    if last < n:
        out.append(text[last:])
    return [s for s in out if s]


def _match_digits(text: str, i: int) -> int:
    if not _is_number(text[i]):
        return 0
    j = i
    count = 0
    n = len(text)
    while j < n and _is_number(text[j]) and count < 3:
        j += 1
        count += 1
    return j - i


def _match_cjk(text: str, i: int) -> int:
    if not _is_cjk(text[i]):
        return 0
    j = i
    n = len(text)
    while j < n and _is_cjk(text[j]):
        j += 1
    return j - i


def _match_pattern3(text: str, i: int) -> int:
    n = len(text)
    ch = text[i]

    # 1) Punct + ASCII letters: [punct][A-Za-z]+
    if ch in _PUNCT_SET and i + 1 < n and _is_ascii_letter(text[i + 1]):
        j = i + 1
        while j < n and _is_ascii_letter(text[j]):
            j += 1
        return j - i

    # 2) Optional not CR/LF and not L/P/S, then [L/M]+
    if _is_letter(ch) or _is_mark(ch):
        j = i
        while j < n and (_is_letter(text[j]) or _is_mark(text[j])):
            j += 1
        return j - i
    if ch not in "\r\n" and not (_is_letter(ch) or _is_punct(ch) or _is_symbol(ch)):
        if i + 1 < n and (_is_letter(text[i + 1]) or _is_mark(text[i + 1])):
            j = i + 1
            while j < n and (_is_letter(text[j]) or _is_mark(text[j])):
                j += 1
            return j - i

    # 3) Optional space + [P/S]+ + CR/LF*
    j = i
    if j < n and text[j] == " ":
        j += 1
    if j < n and (_is_punct(text[j]) or _is_symbol(text[j])):
        while j < n and (_is_punct(text[j]) or _is_symbol(text[j])):
            j += 1
        while j < n and text[j] in "\r\n":
            j += 1
        return j - i

    # 4) \s*[\r\n]+
    if ch.isspace():
        j = i
        while j < n and text[j].isspace():
            j += 1
        k = None
        for t in range(i, j):
            if text[t] in "\r\n":
                k = t
                break
        if k is not None:
            end_idx = k
            while end_idx < n and text[end_idx] in "\r\n":
                end_idx += 1
            return end_idx - i

    # 5) \s+(?!\S)
    if ch.isspace():
        if all(c.isspace() for c in text[i:]):
            return n - i

    # 6) \s+
    if ch.isspace():
        j = i
        while j < n and text[j].isspace():
            j += 1
        return j - i

    return 0


@dataclass(frozen=True)
class AddedToken:
    content: str
    special: bool


class _TrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self) -> None:
        self.children: Dict[str, _TrieNode] = {}
        self.token_id: Optional[int] = None


class AddedTokenTrie:
    def __init__(self) -> None:
        self._root = _TrieNode()

    def add(self, token: str, token_id: int) -> None:
        node = self._root
        for ch in token:
            node = node.children.setdefault(ch, _TrieNode())
        node.token_id = token_id

    def longest_match(self, text: str, start: int) -> Optional[Tuple[int, int]]:
        node = self._root
        match_id: Optional[int] = None
        match_end: Optional[int] = None
        i = start
        n = len(text)
        while i < n:
            child = node.children.get(text[i])
            if child is None:
                break
            node = child
            i += 1
            if node.token_id is not None:
                match_id = node.token_id
                match_end = i
        if match_id is None or match_end is None:
            return None
        return match_id, match_end


def _split_added_tokens(
    text: str, trie: AddedTokenTrie, special_ids: Optional[set]
) -> List[Tuple[str, Optional[int]]]:
    if not text:
        return []
    out: List[Tuple[str, Optional[int]]] = []
    i = 0
    n = len(text)
    buffer_start = 0
    while i < n:
        match = trie.longest_match(text, i)
        if match is None:
            i += 1
            continue
        token_id, end = match
        if special_ids is not None and token_id in special_ids:
            # Skip matching special tokens if requested.
            i += 1
            continue
        if buffer_start < i:
            out.append((text[buffer_start:i], None))
        out.append((text[i:end], token_id))
        i = end
        buffer_start = i
    if buffer_start < n:
        out.append((text[buffer_start:], None))
    return out


def _get_pairs(word: Sequence[str]) -> set:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class BPE:
    def __init__(self, vocab: Dict[str, int], merges: Iterable[str]) -> None:
        self.vocab = vocab
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        for rank, merge in enumerate(merges):
            parts = merge.split()
            if len(parts) != 2:
                continue
            self.bpe_ranks[(parts[0], parts[1])] = rank
        self.cache: Dict[str, List[str]] = {}
        self._max_cache = 50000

    def bpe(self, token: str) -> List[str]:
        cached = self.cache.get(token)
        if cached is not None:
            return cached

        word = tuple(token)
        if len(word) == 1:
            out = [token]
            self._cache_set(token, out)
            return out

        pairs = _get_pairs(word)
        while True:
            min_pair = None
            min_rank = None
            for pair in pairs:
                rank = self.bpe_ranks.get(pair)
                if rank is None:
                    continue
                if min_rank is None or rank < min_rank:
                    min_rank = rank
                    min_pair = pair
            if min_pair is None:
                break

            first, second = min_pair
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                if j < len(word) - 1 and word[j] == first and word[j + 1] == second:
                    new_word.append(first + second)
                    i = j + 2
                else:
                    new_word.append(word[j])
                    i = j + 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        out = list(word)
        self._cache_set(token, out)
        return out

    def _cache_set(self, token: str, out: List[str]) -> None:
        if len(self.cache) >= self._max_cache:
            self.cache.clear()
        self.cache[token] = out


class PurePythonTokenizer:
    def __init__(self, tokenizer_json: Path) -> None:
        with tokenizer_json.open("r", encoding="utf-8") as f:
            data = json.load(f)

        model = data.get("model", {})
        if model.get("type") != "BPE":
            raise ValueError("Only BPE model is supported")

        self.vocab: Dict[str, int] = model.get("vocab", {})
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.bpe = BPE(self.vocab, model.get("merges", []))

        # Added tokens
        self.added_tokens: Dict[int, AddedToken] = {}
        self.added_token_ids: Dict[str, int] = {}
        self.special_token_ids: set = set()
        self.trie = AddedTokenTrie()
        for item in data.get("added_tokens", []):
            content = item["content"]
            token_id = int(item["id"])
            special = bool(item.get("special", False))
            added = AddedToken(content=content, special=special)
            self.added_tokens[token_id] = added
            self.added_token_ids[content] = token_id
            if special:
                self.special_token_ids.add(token_id)
            self.trie.add(content, token_id)

        # Validate expected pre-tokenizer structure for this project.
        pretok = data.get("pre_tokenizer", {})
        if pretok.get("type") != "Sequence":
            raise ValueError("Unsupported pre_tokenizer type")

        # We will use the configured patterns as-is, but only support the exact config.
        pretoks = pretok.get("pretokenizers", [])
        if len(pretoks) < 4:
            raise ValueError("Unexpected pretokenizers length")
        if pretoks[-1].get("type") != "ByteLevel":
            raise ValueError("Expected ByteLevel pretokenizer at the end")
        self.byte_level_add_prefix_space = bool(pretoks[-1].get("add_prefix_space", False))
        self.byte_level_use_regex = bool(pretoks[-1].get("use_regex", False))

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        # Added tokens are matched first (encode_special_tokens is False by default).
        pieces = _split_added_tokens(text, self.trie, special_ids=None)

        tokens: List[Tuple[str, Optional[int]]] = []
        for piece, token_id in pieces:
            if token_id is not None:
                tokens.append((piece, token_id))
                continue
            for part in self._pre_tokenize(piece):
                tokens.append((part, None))

        ids: List[int] = []
        for token, token_id in tokens:
            if token_id is not None:
                ids.append(token_id)
                continue
            for bpe_token in self.bpe.bpe(token):
                tok_id = self.vocab.get(bpe_token)
                if tok_id is None:
                    # This should not happen for byte-level vocab; fall back to unknown behavior.
                    continue
                ids.append(tok_id)
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        tokens: List[str] = []
        for token_id in ids:
            if skip_special_tokens and token_id in self.special_token_ids:
                continue
            added = self.added_tokens.get(token_id)
            if added is not None:
                tokens.append(added.content)
                continue
            token = self.id_to_token.get(int(token_id))
            if token is not None:
                tokens.append(token)
        return _byte_level_decode(tokens)

    def get_vocab_size(self) -> int:
        return len(self.vocab) + len(self.added_tokens)

    def token_to_id(self, token: str) -> Optional[int]:
        if token in self.added_token_ids:
            return self.added_token_ids[token]
        return self.vocab.get(token)

    def id_to_token_str(self, token_id: int) -> Optional[str]:
        added = self.added_tokens.get(token_id)
        if added is not None:
            return added.content
        return self.id_to_token.get(token_id)

    def _pre_tokenize(self, text: str) -> List[str]:
        # Apply the 3 Split pretokenizers in sequence, then byte-level transform.
        parts = [text]
        parts = [p for t in parts for p in _split_isolated(t, _match_digits)]
        parts = [p for t in parts for p in _split_isolated(t, _match_cjk)]
        parts = [p for t in parts for p in _split_isolated(t, _match_pattern3)]

        if self.byte_level_add_prefix_space and parts:
            if not parts[0].startswith(" "):
                parts[0] = " " + parts[0]

        # ByteLevel pretokenizer in this config uses_regex=False, so only transform.
        return [_byte_level_encode(p) for p in parts if p]
