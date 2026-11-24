from __future__ import annotations

import json
from datetime import datetime
from functools import partial
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import files

TZ = ZoneInfo("Europe/Berlin")


def human_size(size: float) -> str:
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def go_to(rel: str) -> None:
    st.session_state["folder"] = rel


def breadcrumbs(layer: str, folder: str) -> None:
    parts = folder.split("/") if folder else []
    crumbs = [(layer, "")] + [(p, "/".join(parts[: i + 1])) for i, p in enumerate(parts)]
    with st.container(horizontal=True):
        for label, rel in crumbs:
            st.button(
                label,
                key=f"crumb:{layer}:{rel}",
                on_click=go_to,
                args=(rel,),
                disabled=rel == folder,
                type="tertiary",
                icon=":material/folder_open:",
            )


def show_preview(layer: str, entry: files.Entry) -> None:
    file_kind = files.kind(entry.name)
    if file_kind == "csv":
        left, right = st.columns(2)
        rows = left.select_slider("Zeilen", [50, 200, 500, 1000], value=200)
        tail = right.radio("Ausschnitt", ["Ende", "Anfang"], horizontal=True) == "Ende"
        df = files.preview_csv(layer, entry.rel, rows=rows, tail=tail)
        columns = st.multiselect("Spalten (leer = alle)", list(df.columns), key=f"cols:{entry.rel}")
        st.dataframe(df[columns] if columns else df, hide_index=True, width="stretch")
        st.caption(f"{len(df)} Zeilen angezeigt, Werte unverändert wie in der Datei")
    elif file_kind == "json":
        text, truncated = files.preview_text(layer, entry.rel)
        try:
            st.json(json.loads(text))
        except ValueError:
            st.code(text, language=None)
        if truncated:
            st.caption("Vorschau gekürzt")
    elif file_kind in ("text", "gz"):
        reader = files.preview_gz if file_kind == "gz" else files.preview_text
        text, truncated = reader(layer, entry.rel)
        st.code(text[:100_000] or "(leer)", language=None, height=500)
        if truncated or len(text) > 100_000:
            st.caption("Vorschau gekürzt")
    elif file_kind == "zip":
        st.dataframe(files.preview_zip(layer, entry.rel), hide_index=True, width="stretch")
    else:
        st.info("Für diesen Dateityp gibt es keine Vorschau.")


def show_file(layer: str, entry: files.Entry) -> None:
    st.subheader(entry.name)
    if entry.size > files.MAX_DOWNLOAD_BYTES:
        st.warning("Die Datei ist zu groß für den Download.")
    else:
        st.download_button(
            f"Herunterladen ({human_size(entry.size)})",
            data=partial(files.read_download, layer, entry.rel),  # read only when clicked
            file_name=entry.name,
            mime="application/octet-stream",
            icon=":material/download:",
            on_click="ignore",
            key=f"download:{layer}:{entry.rel}",
        )
    try:
        show_preview(layer, entry)
    except (files.FileAccessError, OSError):
        st.warning("Datei ist leer, nicht lesbar oder wird gerade aktualisiert. Bitte neu laden.")


def main() -> None:
    st.set_page_config(page_title="Neckarinsel Daten", page_icon=":material/water:", layout="wide")
    st.title("Neckarinsel Daten")

    layer = st.segmented_control(
        "Ebene", files.LAYERS, default="gold", required=True, key="layer", on_change=go_to, args=("",)
    )
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    query = left.text_input("Dateiname filtern", placeholder="z. B. predictions")
    show_raw = right.toggle("Rohdateien anzeigen", help="Logs, Metadaten, historical/ und recents/")

    folder = st.session_state.setdefault("folder", "")
    breadcrumbs(layer, folder)
    try:
        entries = files.list_dir(layer, folder, show_raw)
    except (files.FileAccessError, OSError):
        st.error("Ordner nicht gefunden.")
        return

    folders = [e for e in entries if e.is_dir]
    shown = [e for e in entries if not e.is_dir and query.lower() in e.name.lower()]

    if folders:
        with st.container(horizontal=True):
            for e in folders:
                st.button(
                    e.name,
                    key=f"dir:{layer}:{e.rel}",
                    on_click=go_to,
                    args=(e.rel,),
                    icon=":material/folder:",
                )
    if not shown:
        if query:
            st.info("Keine passenden Dateien in diesem Ordner.")
        elif not folders:
            st.info("Keine Dateien in diesem Ordner.")
        return

    table = pd.DataFrame(
        {
            "Datei": [e.name for e in shown],
            "Größe (KB)": [round(e.size / 1024, 1) for e in shown],
            "Geändert": [datetime.fromtimestamp(e.mtime, TZ).replace(tzinfo=None) for e in shown],
        }
    )
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=f"files:{layer}:{folder}:{query}:{show_raw}",
        column_config={"Geändert": st.column_config.DatetimeColumn(format="DD.MM.YYYY HH:mm")},
    )
    selected = event.selection.rows
    if selected and selected[0] < len(shown):
        show_file(layer, shown[selected[0]])
    else:
        st.caption("Datei in der Tabelle ganz links markieren, um sie anzusehen oder herunterzuladen.")


main()
