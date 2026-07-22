# backend/tests/test_categories.py
import numpy as np
import pytest

from labeling.categories import (
    CATEGORY_ARTIFACT,
    CATEGORY_EXCLUDED_REVIEW,
    CATEGORY_NAMES,
    CATEGORY_NONE,
    CATEGORY_TRANSIENT,
    category_histogram,
    parse_category,
)


def test_names_round_trip():
    for value, name in CATEGORY_NAMES.items():
        assert parse_category(name) == value
        assert parse_category(value) == value


def test_parse_accepts_case_and_whitespace():
    assert parse_category(" Artifact ") == CATEGORY_ARTIFACT


@pytest.mark.parametrize("bad", ["thing", "", 4, -1, None, True])
def test_parse_rejects_unknown(bad):
    with pytest.raises(ValueError):
        parse_category(bad)


def test_histogram_has_every_key_including_zeros():
    cats = np.array([CATEGORY_NONE, CATEGORY_ARTIFACT, CATEGORY_ARTIFACT,
                     CATEGORY_EXCLUDED_REVIEW], dtype=np.int8)
    hist = category_histogram(cats)
    assert hist == {"none": 1, "artifact": 2, "transient": 0, "excluded_review": 1}
    assert set(hist) == set(CATEGORY_NAMES.values())
    assert CATEGORY_TRANSIENT not in cats  # sanity: the zero above is real
