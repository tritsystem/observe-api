import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from observe_search_tools.cost_guard import looks_like_exact_identifier  # noqa: E402


def test_bare_snake_case_identifier_flagged():
    assert looks_like_exact_identifier("retry_upload") is True


def test_bare_camel_case_identifier_flagged():
    assert looks_like_exact_identifier("RetryUpload") is True


def test_single_plain_word_flagged():
    assert looks_like_exact_identifier("useState") is True


def test_file_path_flagged():
    assert looks_like_exact_identifier("lib/retry.js") is True
    assert looks_like_exact_identifier("src\\utils\\retry.py") is True


def test_natural_language_query_not_flagged():
    assert looks_like_exact_identifier("where does this handle retrying a failed upload") is False


def test_multi_word_query_never_flagged_even_if_short():
    assert looks_like_exact_identifier("retry logic") is False


def test_empty_and_whitespace_not_flagged():
    assert looks_like_exact_identifier("") is False
    assert looks_like_exact_identifier("   ") is False


def test_leading_trailing_whitespace_stripped_before_check():
    assert looks_like_exact_identifier("  retry_upload  ") is True
