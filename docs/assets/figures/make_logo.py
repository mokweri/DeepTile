#!/usr/bin/env python3
"""Generate the DeepTile logo mark in light and dark variants.

    python3 docs/assets/figures/make_logo.py

Writes deeptile-logo-light.svg and deeptile-logo-dark.svg next to this script.
The mark is a chip outline holding a 3x3 grid of tiles, one of them orange: the
FPGA target executing a tiled schedule, one tile at a time. It deliberately
shares the chip outline and palette of the FixQuant logo, so the two projects
read as a pair.

Built on a 256-unit canvas and checked at 32 px. make_hero.py reuses
mark_elements() for the FPGA icon, so edit the geometry here.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

THEMES = {
    # Blue tiles, orange active tile: the pair passes the colour-blind
    # separation, lightness and contrast checks against each page background.
    'light': dict(chip='#1F2328', tile='#2A62A8', active='#C2701A'),
    'dark': dict(chip='#E6EDF3', tile='#5A94E0', active='#CF7B2B'),
}

S = 256
BODY = (36, 220)
BODY_STROKE = 12
PIN_LEN, PIN_STROKE = 18, 11
PINS = (92, 128, 164)

GRID_N, TILE, TILE_GAP = 3, 38, 12        # 3x3 tiles, centred in the body
ACTIVE = (2, 2)                           # (row, column) of the orange tile


def mark_elements(t):
    """The logo's SVG elements on the 256-unit canvas, without the <svg> wrapper."""
    lo, hi = BODY
    out = [f'<rect x="{lo}" y="{lo}" width="{hi-lo}" height="{hi-lo}" rx="22" '
           f'fill="none" stroke="{t["chip"]}" stroke-width="{BODY_STROKE}"/>']
    edge = BODY_STROKE / 2
    pin = f'stroke="{t["chip"]}" stroke-width="{PIN_STROKE}" stroke-linecap="round"'
    for p in PINS:
        out.append(f'<line x1="{p}" y1="{lo-edge}" x2="{p}" y2="{lo-edge-PIN_LEN}" {pin}/>')
        out.append(f'<line x1="{p}" y1="{hi+edge}" x2="{p}" y2="{hi+edge+PIN_LEN}" {pin}/>')
        out.append(f'<line x1="{lo-edge}" y1="{p}" x2="{lo-edge-PIN_LEN}" y2="{p}" {pin}/>')
        out.append(f'<line x1="{hi+edge}" y1="{p}" x2="{hi+edge+PIN_LEN}" y2="{p}" {pin}/>')
    span = GRID_N * TILE + (GRID_N - 1) * TILE_GAP
    x0 = (S - span) / 2
    for r in range(GRID_N):
        for c in range(GRID_N):
            fill = t['active'] if (r, c) == ACTIVE else t['tile']
            out.append(f'<rect x="{x0 + c*(TILE+TILE_GAP)}" y="{x0 + r*(TILE+TILE_GAP)}" '
                       f'width="{TILE}" height="{TILE}" rx="6" fill="{fill}"/>')
    return out


def mark(t):
    return '\n'.join([f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}" '
                      f'width="{S}" height="{S}" role="img" aria-label="DeepTile logo">',
                      *mark_elements(t), '</svg>'])


if __name__ == '__main__':
    for name, theme in THEMES.items():
        path = HERE / f'deeptile-logo-{name}.svg'
        path.write_text(mark(theme) + '\n')
        print(f'wrote {path.relative_to(HERE.parents[2])}')
