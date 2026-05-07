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

    async def _open_select2_dropdown() -> None:
        # The search input exists in the DOM but is often hidden until the dropdown is opened.
        try:
            any_field = page.locator("input.select2-search__field").first
            if await any_field.count():
                described_by = await any_field.get_attribute("aria-describedby")
                if described_by:
                    await page.locator(f"#{described_by}").click(timeout=5000)
                    return
        except Exception:
            pass

        # Fallbacks: click the first available Select2 container/selection.
        for sel in [
            ".select2-container:visible",
            "span.select2-selection:visible",
            ".select2-selection--multiple:visible",
            ".select2-selection--single:visible",
            ".select2-container",
            "span.select2-selection",
        ]:
            try:
                loc = page.locator(sel).first
                if await loc.count():
                    await loc.scroll_into_view_if_needed(timeout=5000)
                    await loc.click(timeout=5000)
                    return
            except Exception:
                continue

    # Try a few times to ensure a *visible* select2 search field.
    last_exc: BaseException | None = None
    for _ in range(5):
        try:
            await _open_select2_dropdown()
            await page.wait_for_selector("input.select2-search__field:visible", timeout=15000)
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            await page.wait_for_timeout(500)
    if last_exc is not None:
        raise last_exc

    for kreis in KREISE:
        # Always use the visible field (there may be multiple hidden ones in the DOM).
        field = page.locator("input.select2-search__field:visible").first
        await field.click(timeout=10000)
        await field.fill(kreis, timeout=10000)
        await page.wait_for_selector(".select2-results__option", timeout=5000)
        await page.locator(".select2-results__option", has_text=kreis).click()
        await page.wait_for_timeout(400)

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
    header = page.locator("th:has-text('Parameter')").first
    await header.scroll_into_view_if_needed()
    await header.hover()
    await page.wait_for_timeout(500)

    export_btn = page.locator("button[aria-label='Exportieren']").first
    await export_btn.wait_for(state="visible", timeout=4000)
    await export_btn.click()

    csv_entry = page.locator("role=menuitem >> text=CSV-Datei").first
    await csv_entry.wait_for(state="visible", timeout=4000)

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

    def runner() -> None:
        try:
            if os.name == "nt":
                try:
                    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                except Exception:
                    pass
            asyncio.run(main_async(headless=headless))
        except BaseException as exc:
            error.append(exc)

    t = threading.Thread(target=runner, name="lubw_playwright")
    t.start()
    t.join()

    if error:
        raise error[0]


if __name__ == "__main__":
    headless_default = _env_flag("LUBW_HEADLESS", True)
    main(headless=headless_default)
