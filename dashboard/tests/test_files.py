import gzip
import os
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import files  # noqa: E402


@pytest.fixture
def root(tmp_path):
    for layer in files.LAYERS:
        (tmp_path / layer).mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "secret.json").write_text("{}")
    (tmp_path / "bronze" / "dwd" / "recents").mkdir(parents=True)
    (tmp_path / "bronze" / "dwd" / "a.csv").write_text("x,y\n1,2\n")
    (tmp_path / "bronze" / "dwd" / "dwd_crawl.log").write_text("log")
    (tmp_path / "bronze" / "dwd" / "dwd_crawl.log.1.gz").write_bytes(gzip.compress(b"old"))
    return tmp_path


@pytest.mark.parametrize(
    "rel",
    [
        "..",
        "../config/secret.json",
        "dwd/../../config",
        "/etc/passwd",
        "C:/x",
        "..\\config",
        "a\x00b",
    ],
)
def test_traversal_is_rejected(root, rel):
    with pytest.raises(files.FileAccessError):
        files.safe_path("bronze", rel, root)


@pytest.mark.parametrize("layer", ["config", "..", "", "bronze/../config"])
def test_only_known_layers(root, layer):
    with pytest.raises(files.FileAccessError):
        files.safe_path(layer, "", root)


def test_sibling_with_same_prefix_is_rejected(root):
    (root / "bronze_private").mkdir()
    with pytest.raises(files.FileAccessError):
        files.safe_path("bronze", "../bronze_private", root)


@pytest.mark.skipif(os.name == "nt", reason="symlinks need extra rights on Windows")
def test_outside_symlink_is_hidden_and_rejected(root):
    (root / "bronze" / "link.json").symlink_to(root / "config" / "secret.json")
    assert "link.json" not in [e.name for e in files.list_dir("bronze", "", True, root)]
    with pytest.raises(files.FileAccessError):
        files.read_download("bronze", "link.json", root)


def test_raw_files_hidden_by_default(root):
    assert [e.name for e in files.list_dir("bronze", "dwd", root=root)] == ["a.csv"]
    names = {e.name for e in files.list_dir("bronze", "dwd", True, root)}
    assert names == {"recents", "a.csv", "dwd_crawl.log", "dwd_crawl.log.1.gz"}


@pytest.mark.parametrize(
    "raw",
    [
        b"d,v\n1,\xc3\xa4\n",
        b"\xef\xbb\xbfd;v\r\n1;\xc3\xa4\r\n",
        b"d;v\n1;\xe4\n",  # cp1252
        b'"d";"v"\n"1";"\xc3\xa4"\n',
    ],
)
def test_csv_formats(root, raw):
    (root / "gold" / "f.csv").write_bytes(raw)
    df = files.preview_csv("gold", "f.csv", root=root)
    assert list(df.columns) == ["d", "v"]
    assert df.iloc[0].tolist() == ["1", "ä"]


def test_head_and_tail_on_large_file(root):
    lines = "\n".join(f"{i},{'x' * 50}" for i in range(60_000))
    (root / "gold" / "big.csv").write_text("i,v\n" + lines + "\n")
    head = files.preview_csv("gold", "big.csv", rows=10, root=root)
    tail = files.preview_csv("gold", "big.csv", rows=10, tail=True, root=root)
    assert head["i"].tolist() == [str(i) for i in range(10)]
    assert tail["i"].tolist() == [str(i) for i in range(59_990, 60_000)]


def test_header_only_and_empty_csv(root):
    (root / "gold" / "h.csv").write_text("a,b\n")
    assert files.preview_csv("gold", "h.csv", root=root).empty
    (root / "gold" / "e.csv").write_text("")
    with pytest.raises(files.FileAccessError):
        files.preview_csv("gold", "e.csv", root=root, wait=0)


def test_retry_until_file_is_written(root, monkeypatch):
    path = root / "gold" / "r.csv"
    path.write_text("")
    monkeypatch.setattr(files.time, "sleep", lambda _: path.write_text("a,b\n1,2\n"))
    assert len(files.preview_csv("gold", "r.csv", root=root)) == 1


def test_download_limit(root, monkeypatch):
    (root / "gold" / "d.bin").write_bytes(b"x" * 11)
    assert files.read_download("gold", "d.bin", root) == b"x" * 11
    monkeypatch.setattr(files, "MAX_DOWNLOAD_BYTES", 10)
    with pytest.raises(files.FileAccessError):
        files.read_download("gold", "d.bin", root)


def test_archives(root):
    with zipfile.ZipFile(root / "gold" / "z.zip", "w") as z:
        z.writestr("inner.txt", "hello")
    assert files.preview_zip("gold", "z.zip", root)["Datei"].tolist() == ["inner.txt"]
    (root / "gold" / "bad.zip").write_bytes(b"nope")
    with pytest.raises(files.FileAccessError):
        files.preview_zip("gold", "bad.zip", root)
    (root / "gold" / "g.gz").write_bytes(gzip.compress(b"a" * (2 * files.MAX_PREVIEW_BYTES)))
    text, truncated = files.preview_gz("gold", "g.gz", root)
    assert truncated and len(text) == files.MAX_PREVIEW_BYTES


def test_pickle_has_no_preview():
    assert files.kind("ecoli_model.pkl") == "none"
