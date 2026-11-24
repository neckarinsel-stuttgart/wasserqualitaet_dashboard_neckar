import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import files  # noqa: E402
import important  # noqa: E402


@pytest.fixture
def root(tmp_path, monkeypatch):
    for group in important.GROUPS:
        for item in group.items:
            folder = tmp_path / item.layer / item.folder
            folder.mkdir(parents=True, exist_ok=True)
            name = item.pattern.replace("[0-9][0-9][0-9][0-9]", "2026")
            (folder / name).write_text("a,b\n1,2\n")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "x.json").write_text("{}")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    return tmp_path


def run(**params):
    app = AppTest.from_file(str(HERE / "app.py"), default_timeout=20)
    for key, value in params.items():
        app.query_params[key] = value
    return app.run()


def test_every_item_resolves(root):
    for group in important.GROUPS:
        assert group.folder[0] in files.LAYERS
        for item in group.items:
            assert important.resolve(item, root), item.pattern


def test_new_year_appears(root):
    (root / "bronze" / "messungen" / "messungen_2027.csv").write_text("a,b\n1,2\n")
    (root / "bronze" / "messungen" / "messungen_alt.csv").write_text("a,b\n1,2\n")
    item = next(i for g in important.GROUPS for i in g.items if "[0-9]" in i.pattern)
    assert [e.name for e in important.resolve(item, root)] == [
        "messungen_2027.csv",
        "messungen_2026.csv",
    ]


def test_start_page_lists_groups(root):
    app = run()
    assert not app.exception
    assert [h.value for h in app.subheader] == [g.title for g in important.GROUPS]


def test_start_page_survives_missing_files(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert not run().exception


def test_open_file_and_go_back(root):
    app = run()
    app.button(key="open:gold/datasets/gold_daily_dataset.csv").click().run()
    assert not app.exception
    assert app.query_params["datei"] == "gold/datasets/gold_daily_dataset.csv"
    assert app.dataframe[0].value.shape == (1, 2)
    app.button(key="back").click().run()
    assert "datei" not in app.query_params
    assert app.subheader


def test_open_folder(root):
    app = run()
    app.button(key="folder:bronze/lubw").click().run()
    assert not app.exception
    assert app.session_state["layer"] == "bronze"
    assert app.session_state["folder"] == "lubw"


@pytest.mark.parametrize(
    "link",
    [
        "config/x.json",
        "gold/../config/x.json",
        "../config/x.json",
        "gold",
        "gold/nope.csv",
        "/etc/passwd",
        "",
    ],
)
def test_bad_links_show_error_only(root, link):
    app = run(datei=link)
    assert not app.exception
    assert not app.dataframe and not app.json


def test_layers_still_work(root):
    app = run()
    app.button(key="folder:gold/datasets").click().run()
    assert list(app.dataframe[0].value["Datei"]) == [
        "data_full.csv",
        "gold_daily_dataset.csv",
        "masterdata.csv",
    ]
