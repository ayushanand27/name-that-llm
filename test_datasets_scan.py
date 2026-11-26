import pandas as pd

import datasets_scan as ds

LABELS = ["glm-5.3", "glm-5.3-flash", "kimi-k3", "gemini-3.7-flash"]


def test_parse_list_csv():
    csv = "ref,title,size,lastUpdated,downloadCount,voteCount,usabilityRating\na/b,T,1MB,2026-11-01,3,1,0.5\n"
    assert ds.parse_list(csv)[0]["ref"] == "a/b"


def test_find_columns_picks_text_and_label():
    df = pd.DataFrame({"id": [1, 2], "generated_text": ["a b c", "d e f"], "model_name": ["kimi-k3", "glm-5.3"]})
    assert ds.find_columns(df, LABELS) == ("generated_text", "model_name")


def test_find_columns_none_when_no_label_column():
    df = pd.DataFrame({"text": ["a", "b"], "x": ["foo", "bar"]})
    assert ds.find_columns(df, LABELS) == ("text", None)


def test_label_map_is_exact_after_light_cleanup():
    m = ds.map_label
    assert m("Kimi-K3", LABELS) == "kimi-k3"
    assert m("moonshotai/kimi-k3", LABELS) == "kimi-k3"
    assert m("glm-5.3-flash", LABELS) == "glm-5.3-flash"
    assert m("glm-5.3-air", LABELS) is None and m("gemini-3.7-pro", LABELS) is None


def test_overlap_counts_normalised_duplicates():
    assert ds.overlap(pd.Series(["Hello  World", "new one"]), {ds.norm_key("hello world")}) == 1
