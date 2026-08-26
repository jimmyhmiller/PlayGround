import os

from mill.engine import Engine


def test_hooks_applied():
    os.environ["MILL_MAX_RETRIES"] = "7"
    try:
        eng = Engine("settings.yml")
        assert eng.limits.max_retries == 7
    finally:
        del os.environ["MILL_MAX_RETRIES"]


def test_probe_blocks_padded_keys():
    eng = Engine("settings.yml")
    assert eng.process({" name": "Ada"}) is False


def test_process_cleans():
    eng = Engine("settings.yml")
    assert eng.process({"name": "  Ada   Lovelace "}) is True
    assert eng.writer.rows[-1] == {"name": "Ada Lovelace"}


def test_process_pair_merges():
    eng = Engine("settings.yml")
    assert eng.process_pair({"name": "Ada"}, {"city": " Paris "}) is True
    assert eng.writer.rows[-1] == {"name": "Ada", "city": "Paris"}
