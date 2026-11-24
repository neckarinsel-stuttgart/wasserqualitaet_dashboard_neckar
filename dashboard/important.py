"""Curated list for the "Wichtige Dateien" page. Only points to files, nothing is copied."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

import files

OVERVIEW = """
| Ebene | Was liegt dort? |
|---|---|
| **bronze** | Daten, wie sie ankommen: Laborwerte, Wetter vom DWD, Flusswerte der LUBW |
| **silver** | Bereinigt und vereinheitlicht: alle Laborwerte in einer Datei, bereinigtes Wetter |
| **gold** | Zusammengeführt: fertige Datensätze, Modell und Vorhersagen |

**Ablauf:** Laborwerte + DWD + LUBW (bronze) → bereinigen (silver) → zusammenführen (gold) → Modell
"""


@dataclass(frozen=True)
class Item:
    layer: str
    folder: str
    pattern: str  # file name, may contain * (new yearly files show up by themselves)
    text: str


@dataclass(frozen=True)
class Group:
    title: str
    icon: str
    intro: str
    folder: tuple[str, str]  # layer, folder for the "open folder" button
    items: list[Item] = field(default_factory=list)


GROUPS = [
    Group(
        "Messungen (Laborwerte)",
        ":material/science:",
        "E. coli (`ecoli`) und Enterokokken (`entro`) aus den Wasserproben, pro 100 ml.",
        ("bronze", "messungen"),
        [
            Item(
                "silver",
                "messungen",
                "messungen_komplett.csv",
                "**Alle Jahre zusammen**, einheitliches Format. Für Auswertungen diese Datei nehmen.",
            ),
            Item(
                "bronze",
                "messungen",
                "messungen_[0-9][0-9][0-9][0-9].csv",
                "Originalwerte eines Jahres, eine Zeile pro Probe. Trennzeichen und Datumsformat "
                "unterscheiden sich je nach Jahr.",
            ),
        ],
    ),
    Group(
        "Wetter (DWD, bronze)",
        ":material/partly_cloudy_day:",
        "Stündliche Werte der Station Stuttgart-Schnarrenberg (4928). Dateien mit `clean_` sind "
        "aufbereitet (Fehlwerte leer). Ohne `clean_` sind es die Rohdaten mit `;` als "
        "Trennzeichen und `-999` für Fehlwerte. `QN_…`-Spalten sind Qualitätsstufen des DWD.",
        ("bronze", "dwd"),
        [
            Item(
                "bronze",
                "dwd",
                "clean_wetter_komplett.csv",
                "**Alle Wetterwerte in einer Tabelle**, eine Zeile pro Stunde ab 2023. Die letzten "
                "Stunden sind leer, bis der DWD sie liefert.",
            ),
            Item(
                "bronze",
                "dwd",
                "clean_schnarrenberg_dwd_regen.csv",
                "Niederschlag: `R1` in mm pro Stunde.",
            ),
            Item(
                "bronze",
                "dwd",
                "clean_schnarrenberg_dwd_lufttemp.csv",
                "Luft: Temperatur `TT_TU` (°C), relative Feuchte `RF_TU` (%).",
            ),
            Item(
                "bronze",
                "dwd",
                "clean_schnarrenberg_dwd_boden.csv",
                "Bodentemperatur (°C) in 2 bis 100 cm Tiefe: `V_TE002` … `V_TE100`.",
            ),
            Item(
                "bronze",
                "dwd",
                "clean_schnarrenberg_dwd_druck.csv",
                "Luftdruck (hPa): auf Meereshöhe `P`, an der Station `P0`.",
            ),
            Item(
                "bronze",
                "dwd",
                "clean_schnarrenberg_dwd_feuchtigkeit.csv",
                "Feuchtewerte, z. B. Taupunkt `TD_STD` (°C) und relative Feuchte `RF_STD` (%).",
            ),
            Item(
                "bronze",
                "dwd",
                "clean_schnarrenberg_dwd_sun.csv",
                "Sonnenscheindauer: `SD_SO` in Minuten pro Stunde.",
            ),
            Item(
                "bronze",
                "dwd",
                "clean_schnarrenberg_dwd_strahlung.csv",
                "Sonnenstrahlung, z. B. Globalstrahlung `FG_LBERG` (J/cm²). Große Datei ab 1979, "
                "die neuesten Wochen kommen später als beim übrigen Wetter.",
            ),
        ],
    ),
    Group(
        "Flusswerte (LUBW, bronze)",
        ":material/water:",
        "Tagesmittelwerte der Messstationen an Neckar, Fils, Lauter und Körsch.",
        ("bronze", "lubw"),
        [
            Item(
                "bronze",
                "lubw",
                "lubw_download_latest.csv",
                "Wassertemperatur, pH-Wert, Sauerstoff, Leitfähigkeit und Trübung ab 2024. Eine "
                "Zeile pro Station, Parameter und Tag. Dezimaltrennzeichen ist das Komma.",
            ),
        ],
    ),
    Group(
        "Finaler Datensatz (gold)",
        ":material/star:",
        "Hier laufen Messungen, Wetter und Flusswerte zusammen.",
        ("gold", "datasets"),
        [
            Item(
                "gold",
                "datasets",
                "gold_daily_dataset.csv",
                "**Finaler Datensatz für das Modell.** Eine Zeile pro Messtag: Laborwerte plus "
                "Wetter- und Flussmerkmale. `pos_neg`: 1 = E. coli höchstens 500, 0 = darüber.",
            ),
            Item(
                "gold",
                "datasets",
                "masterdata.csv",
                "Vorstufe: **alle Tage** ab 2024, auch ohne Messung. Pro Wert Mittel, Minimum, "
                "Maximum sowie Stand um 0, 6, 12 und 18 Uhr.",
            ),
            Item(
                "gold",
                "datasets",
                "data_full.csv",
                "Vorstufe: **stündlich** ab 2024, Wetter, Flusswerte und Laborwerte in einer Tabelle.",
            ),
        ],
    ),
]


def resolve(item: Item, root: Path | None = None) -> list[files.Entry]:
    """Existing files for an item, newest name first. Empty if none is there."""
    try:
        entries = files.list_dir(item.layer, item.folder, root=root)
    except (files.FileAccessError, OSError):
        return []
    found = [e for e in entries if not e.is_dir and fnmatch.fnmatchcase(e.name, item.pattern)]
    return sorted(found, key=lambda e: e.name, reverse=True)
