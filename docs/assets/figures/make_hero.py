#!/usr/bin/env python3
"""Generate the README hero figure in light and dark variants.

    python3 docs/assets/figures/make_hero.py

Writes deeptile-hero-light.svg and deeptile-hero-dark.svg next to this script.
The figure carries DeepTile's one-line pitch rather than its internals: trained
networks pass through FixQuant into verified ModelPackages, and one
target-aware compiler maps each onto a single finished FPGA target, so
switching networks never rebuilds the bitstream. The paper's architecture
figure stays in the README for readers who want the detail.

Layout follows a PaperBanana ideation round (many-to-one composition, chevron
compiler stages), redrawn as vector. The two project logos serve as icons: the
FPGA card uses the DeepTile logo from make_logo.py, and the FixQuant card embeds
the FixQuant logo read from the submodule's committed SVG, so neither can drift
from its source. The canvas is 1600 units wide; at GitHub's ~880 px README
width the 20-unit captions display near 11 px, and nothing is set smaller.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from make_logo import mark_elements  # noqa: E402

THEMES = {
    'light': dict(card='#FFFFFF', edge='#D0D7DE', ink='#1F2328', muted='#59636E',
                  accent='#2A62A8', accent_fill='#EEF4FB', chev='#DCE8F6',
                  orange='#C2701A', orange_fill='#FBF0E4', arrow='#59636E',
                  chip='#1F2328', tile='#2A62A8', active='#C2701A'),
    'dark': dict(card='#161B22', edge='#30363D', ink='#E6EDF3', muted='#9198A1',
                 accent='#5A94E0', accent_fill='#12243A', chev='#1B3252',
                 orange='#CF7B2B', orange_fill='#2B1D10', arrow='#9198A1',
                 chip='#E6EDF3', tile='#5A94E0', active='#CF7B2B'),
}
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
W, H = 1600, 560
AY = 280                                  # the main flow runs along this line

# Geometry, left to right.
MX, MW, MH = 28, 236, 92                  # trained-network cards
M_YS = (124, 234, 344)
MERGE_X = 312                             # where the three inputs converge
QX, QW, QY, QH = 348, 166, 172, 216       # FixQuant card
CX, CW, CY, CH = 660, 420, 186, 190       # compiler card
TPW, TPY, TPH = 250, 66, 76               # TargetPackage card, centred on compiler
FX, FW, FY, FH = 1200, 226, 116, 328      # FPGA target card
RX, RW, RY, RH = 1462, 116, 214, 132      # RunRecord card


def text(x, y, s, size, fill, weight=400, anchor='middle'):
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}">{s}</text>')


def card(x, y, w, h, t, fill=None, edge=None, width=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" '
            f'fill="{fill or t["card"]}" stroke="{edge or t["edge"]}" '
            f'stroke-width="{width}"{d}/>')


def arrow(x1, x2, t):
    return (f'<line x1="{x1}" y1="{AY}" x2="{x2}" y2="{AY}" stroke="{t["arrow"]}" '
            f'stroke-width="3.5" marker-end="url(#head)"/>')


def network_icon(cx, cy, c):
    cols = [(-16, [-12, 0, 12]), (0, [-16, -5, 5, 16]), (16, [-6, 6])]
    out = []
    for (x1, ys1), (x2, ys2) in zip(cols, cols[1:]):
        for y1 in ys1:
            for y2 in ys2:
                out.append(f'<line x1="{cx+x1}" y1="{cy+y1}" x2="{cx+x2}" y2="{cy+y2}" '
                           f'stroke="{c}" stroke-width="1.2" opacity="0.55"/>')
    for x, ys in cols:
        for y in ys:
            out.append(f'<circle cx="{cx+x}" cy="{cy+y}" r="3.6" fill="{c}"/>')
    return ''.join(out)


def fixquant_mark(theme):
    """Inner elements of the FixQuant logo, from the submodule's committed SVG."""
    svg = (REPO / f'FixQuant/docs/images/fixquant-logo-{theme}.svg').read_text()
    return re.sub(r'^.*?<svg[^>]*>|</svg>\s*$', '', svg, flags=re.S)


def figure(name, t):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" font-family="{FONT}" role="img" aria-label="DeepTile: trained networks '
         f'such as ResNet-50 and MobileNetV2 are quantized and verified by FixQuant into '
         f'ModelPackages, compiled by one target-aware compiler, and run on a single ZCU104 '
         f'FPGA target. The same bitstream serves every network, with no FPGA rebuild, and '
         f'every run produces a RunRecord.">']
    o.append(f'<defs><marker id="head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
             f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
             f'fill="{t["arrow"]}"/></marker></defs>')

    # Column headers: the many-to-one framing.
    o.append(text(MX + MW / 2, 92, 'Many networks', 25, t['muted'], 600))
    o.append(text(FX + FW / 2, 92, 'One FPGA target', 25, t['muted'], 600))

    # Trained networks, converging on FixQuant.
    names = ('ResNet-50', 'MobileNetV2', 'Any supported CNN')
    for i, (y, nm) in enumerate(zip(M_YS, names)):
        last = i == len(names) - 1
        o.append(card(MX, y, MW, MH, t, dash='8 6' if last else None))
        cy = y + MH / 2
        o.append(network_icon(MX + 40, cy, t['accent']))
        if last:   # two lines: the qualifier 'supported' is the reuse boundary
            o.append(text(MX + 72, cy - 4, 'Any supported', 20, t['muted'], 400, 'start'))
            o.append(text(MX + 72, cy + 20, 'CNN', 20, t['muted'], 400, 'start'))
        else:
            o.append(text(MX + 72, cy + 8, nm, 24, t['ink'], 700, 'start'))
        o.append(f'<path d="M{MX+MW+6},{cy} C{MERGE_X-24},{cy} {MERGE_X-24},{AY} '
                 f'{MERGE_X},{AY}" fill="none" stroke="{t["arrow"]}" stroke-width="3"/>')
    o.append(arrow(MERGE_X, QX - 8, t))
    o.append(text(MX + MW / 2, 480, 'trained models', 20, t['muted']))

    # FixQuant: quantize and verify, emitting a ModelPackage.
    o.append(card(QX, QY, QW, QH, t))
    s = 92 / 256
    o.append(f'<g transform="translate({QX + QW/2 - 46},{QY + 18}) scale({s})">'
             + fixquant_mark(name) + '</g>')
    o.append(text(QX + QW / 2, QY + 142, 'FixQuant', 26, t['ink'], 700))
    o.append(text(QX + QW / 2, QY + 170, 'quantize to INT8,', 19, t['muted']))
    o.append(text(QX + QW / 2, QY + 194, 'verify bit-exactly', 19, t['muted']))
    o.append(arrow(QX + QW + 6, CX - 8, t))
    o.append(text((QX + QW + CX) / 2, AY - 16, 'ModelPackage', 19, t['muted']))

    # TargetPackage, read by the compiler.
    tpx = CX + CW / 2 - TPW / 2
    o.append(card(tpx, TPY, TPW, TPH, t))
    o.append(text(CX + CW / 2, TPY + 34, 'TargetPackage', 25, t['ink'], 700))
    o.append(text(CX + CW / 2, TPY + 62, 'describes the finished target', 19, t['muted']))
    o.append(f'<line x1="{CX+CW/2}" y1="{TPY+TPH+4}" x2="{CX+CW/2}" y2="{CY-8}" '
             f'stroke="{t["arrow"]}" stroke-width="3" stroke-dasharray="7 6" '
             f'marker-end="url(#head)"/>')

    # Compiler: one card with four chevron stages.
    o.append(card(CX, CY, CW, CH, t, t['accent_fill'], t['accent'], 3))
    o.append(text(CX + CW / 2, CY + 48, 'DeepTile compiler', 29, t['ink'], 700))
    stages = ('Validate', 'Lower', 'Tile', 'Schedule')
    cw, tip, x = (CW - 36) / len(stages), 14, CX + 18
    y0, y1 = CY + 76, CY + 140
    for i, st in enumerate(stages):
        notch = 0 if i == 0 else tip
        o.append(f'<path d="M{x},{y0} h{cw-tip} l{tip},{(y1-y0)/2} l-{tip},{(y1-y0)/2} '
                 f'h-{cw-tip} l{notch},-{(y1-y0)/2} z" fill="{t["chev"]}" '
                 f'stroke="{t["accent"]}" stroke-width="2"/>')
        o.append(text(x + cw / 2 + (notch - tip) / 4 + 2, (y0 + y1) / 2 + 7, st, 18,
                      t['ink'], 600))
        x += cw
    o.append(text(CX + CW / 2, CY + CH - 18, 'target-aware, per network', 20, t['muted']))

    # Descriptor stream to the FPGA.
    o.append(arrow(CX + CW + 6, FX - 8, t))
    o.append(text((CX + CW + FX) / 2, AY - 16, 'descriptors', 19, t['muted']))

    # FPGA target, with the DeepTile logo as its icon.
    o.append(card(FX, FY, FW, FH, t))
    s = 108 / 256
    o.append(f'<g transform="translate({FX + FW/2 - 54},{FY + 30}) scale({s})">'
             + ''.join(mark_elements(t)) + '</g>')
    o.append(text(FX + FW / 2, FY + 186, 'ZCU104', 27, t['ink'], 700))
    o.append(text(FX + FW / 2, FY + 222, 'tiled INT8 accelerator', 20, t['muted']))
    o.append(text(FX + FW / 2, FY + 250, 'driven by descriptors', 20, t['muted']))
    o.append(text(FX + FW / 2, FY + 294, 'one bitstream', 22, t['accent'], 700))

    # The key message, directly under the target.
    by, bh = FY + FH + 20, 64
    o.append(card(FX, by, FW, bh, t, t['orange_fill'], t['orange'], 2.5))
    o.append(text(FX + FW / 2, by + 28, 'No FPGA rebuild', 23, t['orange'], 700))
    o.append(text(FX + FW / 2, by + 52, 'when you switch networks', 18, t['ink']))

    # RunRecord.
    o.append(arrow(FX + FW + 6, RX - 8, t))
    o.append(card(RX, RY, RW, RH, t))
    cxr = RX + RW / 2
    o.append(f'<rect x="{cxr-16}" y="{RY+18}" width="32" height="40" rx="4" fill="none" '
             f'stroke="{t["accent"]}" stroke-width="3"/>'
             f'<rect x="{cxr-8}" y="{RY+13}" width="16" height="9" rx="2" fill="{t["accent"]}"/>'
             + ''.join(f'<line x1="{cxr-9}" y1="{RY+32+k*9}" x2="{cxr+9}" y2="{RY+32+k*9}" '
                       f'stroke="{t["accent"]}" stroke-width="2.5"/>' for k in range(3)))
    o.append(text(cxr, RY + 90, 'RunRecord', 19, t['ink'], 700))
    o.append(text(cxr, RY + 116, 'every run', 18, t['muted']))

    o.append('</svg>')
    return '\n'.join(o)


for name, theme in THEMES.items():
    path = HERE / f'deeptile-hero-{name}.svg'
    path.write_text(figure(name, theme) + '\n')
    print(f'wrote {path.relative_to(REPO)}')
