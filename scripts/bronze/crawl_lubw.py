"""Bronze step: scrape and export LUBW Fließgewässer dashboard CSV.

This `.py` file is the canonical pipeline implementation.
The notebook `scripts/bronze/crawl_lubw.ipynb` is for convenience only and may drift.
"""

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    def display(x=None):
        print(x)

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / '.git').exists():
            return parent
    return cur

def resolve_path(env_value: str | None, default: Path, root: Path) -> Path:
    if env_value is None or env_value.strip() == '':
        return default
    p = Path(env_value)
    return p if p.is_absolute() else (root / p)


ROOT = find_repo_root()
DATA_BRONZE_DIR = resolve_path(os.getenv('DATA_BRONZE'), ROOT / 'data' / 'bronze', ROOT)
BRONZE_LUBW_DIR = resolve_path(os.getenv('BRONZE_LUBW_DIR'), DATA_BRONZE_DIR / 'lubw', ROOT)
BRONZE_LUBW_DIR.mkdir(parents=True, exist_ok=True)
print(f'BRONZE_LUBW_DIR={BRONZE_LUBW_DIR}')


"""
Playwright crawler (Notebook‑compatible) for LUBW Fließgewässer‑Dashboard.

- wählt Kreise: Esslingen, Stuttgart, Landeshauptstadt
- setzt Datumsfeld: "2024 - <aktuelles Jahr>"
- exportiert CSV (Exportieren → CSV-Datei)

Hinweis zu Notebooks/Windows:
- Der Jupyter Kernel läuft oft bereits in einer asyncio-Loop.
- Playwright Sync API bricht dann ab.
- Playwright Async API kann auf Windows in manchen Loops keine Subprozesse starten.

Daher: wir führen Playwright Async API in einem separaten Thread
mit eigenem (Proactor-)Event-Loop aus.

Erstinstallation:
    pip install playwright
    python -m playwright install chromium

Aufruf:
    main()
    # sichtbar: setze env LUBW_HEADLESS=0
"""

import asyncio
import os
import threading
from datetime import datetime

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

URL = (
    "https://umweltdaten.lubw.baden-wuerttemberg.de/repositories/"
    "wasser_gewaesserguete,yUm3pYRGyaP7hFRow7Rf/workbooks/"
    "Fliessgewaesserguete,8SMZrw9xObs2ChqTSHk1/worksheets/"
    "Daten-der-Online-Messstationen,XucPlXdlipPF63XPBcOz"
    "?workbookHash=d7e9GgWwzUgJQK5oRyJbAwEQa-fEcb54KVJ-nA2GRzL45K5G"
    "&embeddingTargetId=chemie-messstellen"
)

KREISE = ["Esslingen", "Stuttgart, Landeshauptstadt"]


def _env_flag(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


async def choose_kreise(page) -> None:
    """Wählt die gewünschten Kreise im Select2-Filter aus."""
    print("🔎 Wähle Kreise:", KREISE)

    await page.wait_for_load_state("domcontentloaded")

    # Optional: dismiss common consent/overlay buttons if present.
    for btn_sel in [
        "button:has-text('Akzeptieren')",
        "button:has-text('Zustimmen')",
        "button:has-text('Accept')",
        "button:has-text('Agree')",
    ]:
        try:
            btn = page.locator(btn_sel).first
            if await btn.count():
                await btn.click(timeout=1500)
                await page.wait_for_timeout(250)
                break
        except Exception:
            pass

    async def _kreis_field():
        await page.wait_for_selector("input.select2-search__field:visible", timeout=30000)
        fields = page.locator("input.select2-search__field:visible")
        for i in range(await fields.count()):
            label_id = await fields.nth(i).get_attribute("aria-labelledby")
            if not label_id:
                continue
            label = page.locator(f"[id='{label_id}']")
            if await label.count() and (await label.first.inner_text()).strip() == "Kreis":
                return page.locator(
                    f"input.select2-search__field[aria-labelledby='{label_id}']"
                ).first
        raise RuntimeError("Visible 'Kreis' filter not found on LUBW dashboard")

    def _kreis_chip(field, kreis: str):
        selection = field.locator(
            "xpath=ancestor::*[contains(concat(' ', @class, ' '), ' select2-selection ')][1]"
        )
        return selection.locator(".select2-selection__choice", has_text=kreis)

    attempts = 4
    for kreis in KREISE:
        last_exc: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                field = await _kreis_field()
                if await _kreis_chip(field, kreis).count():
                    last_exc = None
                    break
                await field.click(timeout=10000)
                await field.fill(kreis, timeout=10000)
                option = page.locator(".select2-results__option", has_text=kreis).first
                await option.wait_for(state="visible", timeout=10000)
                await option.click(timeout=10000)
                await _kreis_chip(await _kreis_field(), kreis).first.wait_for(
                    state="visible", timeout=15000
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                print(
                    f"⚠️  Kreis '{kreis}' attempt {attempt}/{attempts} failed "
                    f"({type(exc).__name__}). Retrying..."
                )
                try:
                    await page.keyboard.press("Escape")
                except Exception:
                    pass
                await page.wait_for_timeout(1500)
        if last_exc is not None:
            raise last_exc
        print(f"  ✔ {kreis}")
        await page.wait_for_timeout(1000)

    print("✅ Kreise erfolgreich gewählt.")


async def set_datum_filter(page) -> None:
    """Setzt das Datumsfeld direkt per Texteingabe."""
    current_year = datetime.now().year
    date_range = f"2024 - {current_year}"

    selector = "textarea.d-condition-date-picker--input.form-control[placeholder='yyyy - yyyy']"

    # The dashboard re-renders parts of the filter UI after interactions.
    # This can detach the textarea between locating it and scrolling/filling.
    last_exc: BaseException | None = None
    for attempt in range(1, 6):
        date_input = page.locator(selector).first
        try:
            await date_input.wait_for(state="visible", timeout=15000)
            await date_input.scroll_into_view_if_needed(timeout=10000)
            await date_input.fill(date_range)
            await date_input.press("Enter")
            print(f"✔ Datum gesetzt: {date_range}")
            return
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if "not attached" in msg.lower() or "detached" in msg.lower():
                # Give the UI a moment to settle, then re-locate and retry.
                await page.wait_for_timeout(500)
                continue
            raise

    assert last_exc is not None
    raise last_exc


async def export_csv(page) -> None:
    """Hovert über Spaltenkopf 'Parameter' und exportiert CSV."""
    export_btn = page.locator("button[aria-label='Exportieren']").first
    csv_entry = page.locator("role=menuitem >> text=CSV-Datei").first

    attempts = 5
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            header = page.locator("th:has-text('Parameter'):visible").first
            await header.wait_for(state="visible", timeout=60000)
            await header.hover(timeout=30000)
            await page.wait_for_timeout(500)
            await export_btn.wait_for(state="visible", timeout=10000)
            await export_btn.click(timeout=10000)
            await csv_entry.wait_for(state="visible", timeout=10000)
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            print(
                f"⚠️  Export menu attempt {attempt}/{attempts} failed "
                f"({type(exc).__name__}). Retrying..."
            )
            try:
                await page.keyboard.press("Escape")
            except Exception:
                pass
            await page.wait_for_timeout(3000)
    if last_exc is not None:
        raise last_exc

    download_timeout_ms = int(os.getenv("LUBW_DOWNLOAD_TIMEOUT_MS") or "180000")
    export_retries = int(os.getenv("LUBW_EXPORT_RETRIES") or "2")

    last_exc: BaseException | None = None
    for attempt in range(1, export_retries + 1):
        try:
            async with page.expect_download(timeout=download_timeout_ms) as dl_info:
                await csv_entry.click()

            download = await dl_info.value
            out_path = BRONZE_LUBW_DIR / "lubw_download_latest.csv"
            await download.save_as(str(out_path))
            print(f"✔ CSV erfolgreich gespeichert: {out_path}")
            return
        except PlaywrightTimeoutError as exc:
            last_exc = exc
            if attempt >= export_retries:
                break
            print(
                f"⚠️  CSV download timeout after {download_timeout_ms}ms (attempt {attempt}/{export_retries}). Retrying..."
            )
            # Re-open export menu and try again.
            await export_btn.click()
            await csv_entry.wait_for(state="visible", timeout=10000)

    assert last_exc is not None
    raise last_exc


async def main_async(headless: bool = True) -> None:
    async with async_playwright() as pw:
        launch_args = []
        if os.name != "nt":
            launch_args = ["--no-sandbox", "--disable-dev-shm-usage"]

        browser = await pw.chromium.launch(headless=headless, args=launch_args)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        await page.goto(URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)

        await choose_kreise(page)
        await page.wait_for_timeout(2000)

        await set_datum_filter(page)
        await page.wait_for_timeout(1500)

        await export_csv(page)

        print("✅ Ablauf abgeschlossen")
        await browser.close()


def main(headless: bool = True) -> None:
    """Notebook-friendly entrypoint (runs async Playwright in a fresh thread/loop)."""

    error: list[BaseException] = []

    run_attempts = max(1, int(os.getenv("LUBW_RUN_ATTEMPTS") or "3"))

    def runner() -> None:
        if os.name == "nt":
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            except Exception:
                pass
        for attempt in range(1, run_attempts + 1):
            try:
                asyncio.run(main_async(headless=headless))
                return
            except BaseException as exc:
                if attempt >= run_attempts or isinstance(exc, KeyboardInterrupt):
                    error.append(exc)
                    return
                print(
                    f"⚠️  LUBW run attempt {attempt}/{run_attempts} failed "
                    f"({type(exc).__name__}: {str(exc).splitlines()[0][:200]}). Retrying..."
                )

    t = threading.Thread(target=runner, name="lubw_playwright")
    t.start()
    t.join()

    if error:
        raise error[0]


if __name__ == "__main__":
    headless_default = _env_flag("LUBW_HEADLESS", True)
    main(headless=headless_default)
