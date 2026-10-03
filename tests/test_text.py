"""Text normalisation: the shared tokenisation both retrieval and the offline
model depend on. If they disagree, retrieval and generation optimise for
different things, so this is pinned down explicitly."""

from __future__ import annotations

from app.core.text import STOP_WORDS, stem, token_set, tokenize


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("Refunds, Refunds!") == ["refund", "refund"]


def test_tokenize_drops_stop_words() -> None:
    assert tokenize("What is the capital of France?") == ["capital", "france"]


def test_stop_words_do_not_create_false_overlap() -> None:
    query = token_set("What is the capital of France?")
    unrelated = token_set("The quarterly earnings call is on Tuesday.")
    assert query & unrelated == set()


def test_stem_matches_related_word_forms() -> None:
    assert stem("refunds") == "refund"
    assert stem("encrypted") == "encrypt"
    assert stem("encrypting") == "encrypt"
    assert stem("policies") == "polic"
    assert stem("policy") == "policy"


def test_stem_never_reduces_below_the_minimum_length() -> None:
    # "is" -> "i" and "days" -> "day" would be harmless, but over-stemming short
    # words collapses distinct terms. Short words are left alone.
    assert stem("is") == "is"
    assert stem("days") == "days"
    assert stem("data") == "data"


def test_related_questions_and_answers_share_tokens() -> None:
    question = token_set("How long do I have to request a refund?")
    answer = token_set("You can request a refund within 30 days of your first payment.")
    assert {"request", "refund"} <= question & answer


def test_tokenize_handles_numbers_and_punctuation() -> None:
    assert tokenize("Refund within 30 days!") == ["refund", "within", "30", "days"]


def test_tokenize_handles_empty_input() -> None:
    assert tokenize("") == []
    assert tokenize("   \n\t ") == []
    assert token_set("") == set()


def test_stop_word_list_contains_only_lowercase_words() -> None:
    assert all(word == word.lower() and word.isalpha() for word in STOP_WORDS)
