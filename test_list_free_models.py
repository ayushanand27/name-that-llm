import list_free_models as lf

LABELS = ["glm-5.3", "glm-5.3-flash", "kimi-k3", "qwen3.8-max", "gemini-3.8-flash"]


def test_exact_name_maps():
    assert lf.match("z-ai/glm-5.3:free", LABELS) == ("glm-5.3", "same")
    assert lf.match("moonshotai/kimi-k3:free", LABELS) == ("kimi-k3", "same")


def test_similar_models_are_never_mapped():
    assert lf.match("z-ai/glm-5.3-air:free", LABELS) == (None, "near glm-5.3")
    assert lf.match("qwen/qwen3.8-max-preview:free", LABELS) == (None, "near qwen3.8-max")
    assert lf.match("google/gemini-3.8-flash-lite:free", LABELS) == (None, "near gemini-3.8-flash")
    assert lf.match("meta/llama-5:free", LABELS) == (None, "")


def test_only_free_ids_kept():
    models = [{"id": "a/x:free"}, {"id": "a/y"}, {"id": "z-ai/glm-5.3:free"}]
    assert [m["id"] for m in lf.free_only(models)] == ["a/x:free", "z-ai/glm-5.3:free"]


def test_longest_near_label_wins():
    assert lf.match("z-ai/glm-5.3-flash-2:free", LABELS) == (None, "near glm-5.3-flash")
