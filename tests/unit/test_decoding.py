import pytest

from self_judged_refusal_direction.decoding import EnumTrieConstrainedDecoder
from self_judged_refusal_direction.errors import ConfigurationError, InvariantError


class CharacterTokenizer:
    eos_token_id = 3
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids if token_id > 3)


def test_common_prefix_and_terminal_eos_only() -> None:
    decoder = EnumTrieConstrainedDecoder(
        {"AB": (10, 20), "AC": (10, 30)},
        eos_token_id=99,
        generation_start_lengths=(2,),
    )
    assert decoder.allowed_tokens(0, [7, 8]) == [10]
    assert decoder.allowed_tokens(0, [7, 8, 10]) == [20, 30]
    assert decoder.allowed_tokens(0, [7, 8, 10, 20]) == [99]
    assert decoder.parse_suffix([10, 20, 99]) == ("AB", "AB")


def test_out_of_language_and_missing_eos_are_errors() -> None:
    decoder = EnumTrieConstrainedDecoder({"X": (40,)}, 99, (1,))
    with pytest.raises(InvariantError):
        decoder.allowed_tokens(0, [1, 41])
    with pytest.raises(InvariantError):
        decoder.parse_suffix([40])


def test_prefix_labels_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="prefix"):
        EnumTrieConstrainedDecoder({"A": (1,), "AB": (1, 2)}, 99, (0,))


def test_compile_uses_exact_round_trip_label_sequences() -> None:
    tokenizer = CharacterTokenizer()
    decoder = EnumTrieConstrainedDecoder.compile(
        tokenizer,
        ("REFUSAL", "NON_REFUSAL", "UNCERTAIN"),
        ((ord("P"),),),
    )
    assert decoder.max_new_tokens == len("NON_REFUSAL") + 1
    assert decoder.allowed_tokens(0, [ord("P")]) == [ord("N"), ord("R"), ord("U")]
