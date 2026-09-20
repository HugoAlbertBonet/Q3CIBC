#!/usr/bin/env python3
"""Rebuild REAL_ROBOT_SETUP.pdf from REAL_ROBOT_SETUP.md.

Usage, from this directory:

    python build_pdf.py
    uvx --from weasyprint weasyprint -u . setup.html REAL_ROBOT_SETUP.pdf

Needs `markdown` (pip) and WeasyPrint; the latter is easiest via uvx, which
needs no install. `-u .` is what makes the relative assets/ image paths
resolve.
"""
import re
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SRC = HERE / "REAL_ROBOT_SETUP.md"
OUT = HERE / "setup.html"

text = SRC.read_text()

# Split the title off so it can be rendered as a cover-ish header block.
body_md = re.sub(r"^# .*\n", "", text, count=1)
title = re.match(r"^# (.*)", text).group(1)

html_body = markdown.markdown(
    body_md,
    extensions=["tables", "fenced_code", "codehilite", "attr_list", "toc"],
    extension_configs={"codehilite": {"noclasses": True,
                                      "pygments_style": "friendly"}},
)

# GitHub and python-markdown disagree on slugs: GitHub keeps runs of hyphens
# ("and---control-z"), python-markdown collapses them ("and-control-z"). Keep
# the source GitHub-correct and remap the hrefs here instead.
ids = set(re.findall(r'<h[1-6][^>]*\bid="([^"]+)"', html_body))
collapsed = {re.sub(r"-{2,}", "-", i): i for i in ids}


def _fix_href(m):
    target = m.group(1)
    if target in ids:
        return m.group(0)
    hit = collapsed.get(re.sub(r"-{2,}", "-", target))
    if hit is None:
        print(f"  WARNING unresolved internal link: #{target}", file=sys.stderr)
        return m.group(0)
    return f'href="#{hit}"'


html_body = re.sub(r'href="#([^"]+)"', _fix_href, html_body)

CSS = """
@page {
  size: A4;
  margin: 20mm 18mm 18mm 18mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font: 8.5pt "DejaVu Sans", sans-serif;
    color: #888;
  }
}
@page :first { margin-top: 24mm; }

html { font-size: 10.5pt; }
body {
  font-family: "DejaVu Serif", Georgia, serif;
  line-height: 1.45;
  color: #1a1a1a;
  hyphens: auto;
}

h1.doc-title {
  font-family: "DejaVu Sans", sans-serif;
  font-size: 21pt;
  line-height: 1.2;
  margin: 0 0 4mm 0;
  padding-bottom: 3mm;
  border-bottom: 2.5pt solid #1a1a1a;
}

h1 {
  font-family: "DejaVu Sans", sans-serif;
  font-size: 16pt;
  margin: 9mm 0 4mm 0;
  padding-bottom: 1.5mm;
  border-bottom: 1pt solid #bbb;
  page-break-after: avoid;
  break-after: avoid;
}
h1:first-of-type { margin-top: 2mm; }

h2 {
  font-family: "DejaVu Sans", sans-serif;
  font-size: 12.5pt;
  margin: 6mm 0 2.5mm 0;
  color: #111;
  page-break-after: avoid;
  break-after: avoid;
}

h3 {
  font-family: "DejaVu Sans", sans-serif;
  font-size: 10.5pt;
  margin: 4.5mm 0 2mm 0;
  color: #333;
  page-break-after: avoid;
  break-after: avoid;
}

p { margin: 0 0 2.5mm 0; orphans: 2; widows: 2; }
ul, ol { margin: 0 0 3mm 0; padding-left: 6mm; }
li { margin-bottom: 1.5mm; }

hr { border: none; border-top: 1pt solid #ddd; margin: 5mm 0; }

code {
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 0.85em;
  background: #f2f2f0;
  padding: 0.5pt 2pt;
  border-radius: 2pt;
}
pre {
  background: #f7f7f5;
  border: 0.5pt solid #ddd;
  border-left: 2.5pt solid #999;
  padding: 2.5mm 3mm;
  margin: 0 0 3.5mm 0;
  font-size: 7.8pt;
  line-height: 1.35;
  white-space: pre-wrap;
  overflow-wrap: break-word;
  page-break-inside: avoid;
  break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 1em; }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 0 0 4mm 0;
  font-size: 9pt;
  page-break-inside: avoid;
  break-inside: avoid;
}
th, td {
  border: 0.5pt solid #ccc;
  padding: 1.6mm 2.2mm;
  text-align: left;
  vertical-align: top;
}
th {
  background: #ececea;
  font-family: "DejaVu Sans", sans-serif;
  font-size: 8.5pt;
  font-weight: bold;
}
tr:nth-child(even) td { background: #fafaf8; }
td code, th code { font-size: 0.88em; }

img {
  max-width: 62%;
  display: block;
  margin: 2mm auto 1.5mm auto;
  border: 0.5pt solid #ccc;
  page-break-inside: avoid;
  break-inside: avoid;
}

h1 code, h2 code, h3 code {
  background: none;
  padding: 0;
  font-size: 0.92em;
}

a { color: #1a1a1a; text-decoration: none; border-bottom: 0.4pt dotted #999; }

strong { font-weight: bold; }
"""

OUT.write_text(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<h1 class="doc-title">{title}</h1>
{html_body}
</body>
</html>
""")
print(f"wrote {OUT}")
