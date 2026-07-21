"""Render the Trnscrb app icon and assemble a .icns for the app bundle.

The design carries the menu-bar identity (5-bar waveform, heights
32/52/72/48/24) onto a macOS squircle: diagonal indigo→navy gradient, a
tight glow behind the bars, soft bar shadows, and a top-edge highlight.
Rendered with Pillow at 4x and downsampled; assembled with the system
`iconutil`. Both are best-effort — without them the bundle simply ships
without an icon.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from trnscrb.log import get_logger

_log = get_logger("trnscrb.app_icon")

_MASTER = 4096  # 4x the largest iconset size
_ICONSET_SIZES = (16, 32, 64, 128, 256, 512, 1024)


def _render_master():
    from PIL import Image, ImageDraw, ImageFilter

    S = _MASTER
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))

    # Apple icon grid: content rect inside a transparent margin, squircle corners
    margin = int(100 / 1024 * S)
    rect = [margin, margin, S - margin, S - margin]
    content = S - 2 * margin
    radius = int(0.225 * content)
    cx, cy = S // 2, S // 2

    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle(rect, radius=radius, fill=255)

    # Diagonal gradient: vivid indigo → deep navy
    grad = Image.new("RGBA", (S, S))
    gd = ImageDraw.Draw(grad)
    c0, c1 = (109, 96, 250), (12, 8, 40)
    for i in range(2 * S):
        t = i / (2 * S)
        color = tuple(int(c0[k] + (c1[k] - c0[k]) * t) for k in range(3)) + (255,)
        gd.line([(0, i), (i, 0)], fill=color, width=3)
    img.paste(grad, (0, 0), mask)

    # Tight, dim radial glow behind the bars
    glow = Image.new("L", (S, S), 0)
    gr = int(S * 0.22)
    ImageDraw.Draw(glow).ellipse(
        [cx - gr, cy - int(gr * 0.9), cx + gr, cy + int(gr * 0.9)], fill=70
    )
    glow = glow.filter(ImageFilter.GaussianBlur(S // 16))
    glow_rgba = Image.merge(
        "RGBA",
        [
            Image.new("L", (S, S), 168),
            Image.new("L", (S, S), 162),
            Image.new("L", (S, S), 255),
            glow,
        ],
    )
    img.paste(glow_rgba, (0, 0), Image.composite(glow, Image.new("L", (S, S), 0), mask))

    # Top-edge inner highlight for depth
    hl = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(hl).rounded_rectangle(
        [rect[0] + 6, rect[1] + 6, rect[2] - 6, rect[3] - 6],
        radius=radius - 6,
        outline=(255, 255, 255, 60),
        width=int(S * 0.004),
    )
    hl = hl.filter(ImageFilter.GaussianBlur(S // 512))
    half = Image.new("L", (S, S), 0)
    ImageDraw.Draw(half).rectangle([0, 0, S, cy - int(content * 0.18)], fill=255)
    half = half.filter(ImageFilter.GaussianBlur(S // 24))
    hl.putalpha(Image.composite(hl.split()[3], Image.new("L", (S, S), 0), half))
    img = Image.alpha_composite(img, hl)

    # Waveform bars (brand heights) with a soft drop shadow
    heights = [32, 52, 72, 48, 24]
    tallest = 72
    bar_h_max = int(content * 0.58)
    bar_w = int(content * 0.088)
    gap = int(content * 0.062)
    x0 = (S - (5 * bar_w + 4 * gap)) // 2
    alphas = [150, 208, 255, 208, 150]

    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    bars = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bars)
    for i, (h, a) in enumerate(zip(heights, alphas)):
        bh = int(h / tallest * bar_h_max)
        x = x0 + i * (bar_w + gap)
        y0 = cy - bh // 2
        off = int(S * 0.008)
        sd.rounded_rectangle(
            [x + off, y0 + 2 * off, x + bar_w + off, y0 + bh + 2 * off],
            radius=bar_w // 2,
            fill=(0, 0, 0, 90),
        )
        bd.rounded_rectangle(
            [x, y0, x + bar_w, y0 + bh], radius=bar_w // 2, fill=(255, 255, 255, a)
        )
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(S // 160)))
    img = Image.alpha_composite(img, bars)
    return img


def build_icns(dest: Path) -> bool:
    """Render the icon and write a .icns at ``dest``. False if unavailable."""
    iconutil = shutil.which("iconutil")
    if iconutil is None:
        _log.info("iconutil not available — bundle ships without an icon")
        return False
    try:
        from PIL import Image
    except ImportError:
        _log.info("Pillow not available — bundle ships without an icon")
        return False

    try:
        master = _render_master()
        with tempfile.TemporaryDirectory() as tmp:
            iconset = Path(tmp) / "Trnscrb.iconset"
            iconset.mkdir()
            for size in _ICONSET_SIZES:
                scaled = master.resize((size, size), Image.LANCZOS)
                if size <= 512:
                    scaled.save(iconset / f"icon_{size}x{size}.png")
                if 32 <= size:
                    scaled.save(iconset / f"icon_{size // 2}x{size // 2}@2x.png")
            subprocess.run(
                [iconutil, "-c", "icns", str(iconset), "-o", str(dest)],
                check=True,
                capture_output=True,
                timeout=120,
            )
        _log.info("App icon written to %s", dest)
        return True
    except Exception:
        _log.warning("App icon generation failed — bundle ships without an icon", exc_info=True)
        return False
