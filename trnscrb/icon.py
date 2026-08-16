"""Generate the Trnscrb menu bar icon (mic silhouette PNG).

Run once after install:
    python -m trnscrb.icon

Saves a 44x44 template PNG to ~/.local/share/trnscrb/mic.png
"""

import os
import tempfile
from pathlib import Path

ICON_DIR = Path.home() / ".local" / "share" / "trnscrb"
ICON_IDLE = ICON_DIR / "mic.png"
ICON_RECORDING = ICON_DIR / "mic_active.png"


def generate_icons() -> None:

    ICON_DIR.mkdir(parents=True, exist_ok=True)

    _make_mic(ICON_IDLE, fill=(0, 0, 0, 255))  # black  — idle (macOS template image)
    _make_mic(ICON_RECORDING, fill=(220, 38, 38, 255))  # red    — recording


def _make_mic(path: Path, fill: tuple) -> None:
    from PIL import Image, ImageDraw

    S = 44  # canvas size (retina menu bar = 22 pt @ 2x)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # ── waveform bars — mirrors the trnscrb logo icon ─────────────────────────
    # Logo bar heights (relative): 32, 52, 72, 48, 24  (tallest = 72)
    # Scale tallest bar to 34px to fill 44px canvas with 5px top/bottom margin
    scale = 34 / 72
    bar_w = 4
    gap = 3
    total_w = 5 * bar_w + 4 * gap  # = 32px
    x0 = (S - total_w) // 2  # left edge, centred
    cy = S // 2

    heights = [int(h * scale) for h in [32, 52, 72, 48, 24]]
    opacities = [0.45, 0.72, 1.0, 0.72, 0.45]

    r, g, b, a = fill
    for i, (h, op) in enumerate(zip(heights, opacities)):
        x = x0 + i * (bar_w + gap)
        y0 = cy - h // 2
        y1 = cy + h // 2
        color = (r, g, b, int(a * op))
        d.rounded_rectangle([x, y0, x + bar_w, y1], radius=bar_w // 2, fill=color)

    # Atomic write: render to a temp file in the same dir, then rename into
    # place. The menu bar reads this PNG at launch; a torn read of a
    # half-written file makes ImageIO SIGBUS-crash the app in a loop.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".png")
    os.close(fd)
    try:
        img.save(tmp)
        os.replace(tmp, path)  # atomic on the same filesystem
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _is_readable_png(path: Path) -> bool:
    """True when this file parses as a PNG all the way through.

    Handing a damaged one to the menu bar is not a cosmetic problem: AppKit
    reads it through ImageIO, which faults on a torn or truncated file and
    takes the whole process down with SIGBUS — no traceback, no Python error,
    just a dead app that launchd starts again ten seconds later. Parsing it
    here first costs microseconds on a 250-byte file and turns the worst case
    into a plain emoji icon.
    """
    try:
        from PIL import Image

        with Image.open(path) as img:
            img.verify()  # checks CRCs and that every chunk is complete
        return True
    except Exception:
        return False


def icon_path(recording: bool = False) -> str | None:
    """Path to the icon PNG, or None to fall back to the emoji title.

    A file that will not parse is regenerated once — the usual cause is a
    write interrupted by a crash or a full disk — and skipped if that fails.
    """
    p = ICON_RECORDING if recording else ICON_IDLE
    if not p.exists():
        return None
    if _is_readable_png(p):
        return str(p)
    try:
        generate_icons()
    except Exception:
        return None
    return str(p) if _is_readable_png(p) else None


if __name__ == "__main__":
    generate_icons()
    print(f"Icons written to {ICON_DIR}")


def generate_icons_cli() -> None:
    """Entry point called from the trnscrb CLI (uses the uv tool's Python with PIL)."""
    try:
        generate_icons()
        print(f"✓ Icons written to {ICON_DIR}")
    except ImportError:
        print("Pillow not available — menu bar will use emoji fallback (🎙 / 🔴). That's fine.")
    except Exception as e:
        print(f"Icon generation failed: {e}")
