"""Rebuild the HTML article using the local (untracked) TeX and Pandoc.
Run with the bundled Python, or Python with Pillow installed.
"""
from pathlib import Path
import re
import subprocess
import tempfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'docs/papers/atlas'
source = (ROOT / 'paper_fr.tex').read_text()
# Print-layout overrides redefine section commands and confuse Pandoc's reader.
source = re.sub(r'\\makeatletter.*?\\makeatother', '', source, flags=re.S)
source = source.replace(r'\date{\today}', r'\date{September 22, 2026}')

# Preserve the manuscript's manually numbered bibliography and citation links.
keys = re.findall(r'\\bibitem\{([^}]+)\}', source)
numbers = {key: str(i + 1) for i, key in enumerate(keys)}
def citation(keys):
    return ', '.join(r'\hyperlink{ref-' + key.strip() + '}{[' + numbers[key.strip()] + ']}'
                     for key in keys.split(','))
source = re.sub(r'\\cite(?:\[([^]]*)\])?\{([^}]+)\}',
                lambda m: citation(m.group(2))
                + (' (' + m.group(1) + ')' if m.group(1) else ''), source)
source = source.replace(r'\begin{thebibliography}{99}', r'\section*{References}')
source = source.replace(r'\end{thebibliography}', '')
source = re.sub(r'\\bibitem\{([^}]+)\}',
                lambda m: '\n\n' + r'\hypertarget{ref-' + m.group(1) + '}{}[' + numbers[m.group(1)] + '] ', source)

# Move labels outside their environments so browser cross-references resolve.
for environment in ('equation', 'table'):
    pattern = r'\\begin\{' + environment + r'\}(.*?)\\end\{' + environment + r'\}'
    def anchor_environment(match):
        labels = re.findall(r'\\label\{([^}]+)\}', match.group(1))
        return ''.join(r'\hypertarget{' + label + '}{}' for label in labels) + match.group(0)
    source = re.sub(pattern, anchor_environment, source, flags=re.S)
# Pandoc does not render TikZ. Replace the manuscript diagram with a PDF crop.
source = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}',
                lambda m: r'\includegraphics{visualization_scripts/output/helmholtz_projection.png}',
                source, flags=re.S)

with tempfile.TemporaryDirectory() as directory:
    clean = Path(directory) / 'paper.tex'
    clean.write_text(source)
    result = subprocess.run(['pandoc', str(clean), '--from=latex', '--to=html5',
                             '--mathml', '--section-divs', '--number-sections', '--shift-heading-level-by=1'],
                            check=True, capture_output=True, text=True)
    if result.stderr:
        print(result.stderr)
    body = result.stdout
    count = [0]
    def caption(match):
        count[0] += 1
        return '<figcaption>Figure ' + str(count[0]) + '. '
    body = re.sub(r'<figcaption>', caption, body)
assets = TARGET / 'figures'
assets.mkdir(exist_ok=True)
for name in sorted(set(re.findall(r'src="visualization_scripts/output/([^"<>]+)"', body))):
    image_source = ROOT / 'figures' / Path(name).with_suffix('.png')
    output_name = Path(name).with_suffix('.jpg').name
    if image_source.is_file():
        with Image.open(image_source) as image:
            image = image.convert('RGB')
            image.thumbnail((1600, 1600))
            image.save(assets / output_name, quality=90, optimize=True)
    elif name in ('pushforward_geodesic_paths.pdf', 'helmholtz_projection.png'):
        # The missing vector illustration is Figure 28 on manuscript page 70.
        # Crop only the figure, excluding the running header and caption.
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(ROOT / 'fisher_rao_wasserstein_hal_v1.pdf')
        page_number, box = ((69, (72, 100, 524, 326)) if name == 'pushforward_geodesic_paths.pdf'
                            else (18, (145, 548, 510, 656)))
        image = pdf[page_number].render(scale=3).to_pil().convert('RGB')
        image = image.crop(tuple(value * 3 for value in box))
        pdf.close()
        image.save(assets / output_name, quality=90, optimize=True)
    else:
        raise FileNotFoundError(image_source)
    with Image.open(assets / output_name) as rendered:
        dimensions = f'width="{rendered.width}" height="{rendered.height}" '
    body = body.replace('src="visualization_scripts/output/' + name + '"',
                        dimensions + 'loading="lazy" alt="' + Path(name).stem.replace('_', ' ') +
                        '" src="figures/' + output_name + '"')
body = re.sub(r'<embed(?=\s)', '<img', body)
# Pandoc otherwise prints raw label names for these manually anchored targets.
for label, display in [('tab:atlas-overview', '1'), ('eq:wo-primitives', '(1)'),
                       ('eq:wo-quantile-pullback', '(2)')]:
    body = body.replace('[' + label + ']', display)
    if label.startswith('eq:'):
        body = re.sub(r'(<span\s+id="' + re.escape(label) + r'"></span>)',
                      lambda m: m.group(1) + '<span class="equation-number">' + display + '</span>', body)

# The abstract page holds the authoritative shared metadata and title.
landing = (TARGET / 'index.html').read_text()
head = landing.split('<body>', 1)[0]
head = head.replace('href="https://pierreguilbertf.github.io/fisher-rao-wasserstein-atlas/papers/atlas/"',
                    'href="https://pierreguilbertf.github.io/fisher-rao-wasserstein-atlas/papers/atlas/full-text.html"', 1)
heading = re.search(r'<h1.*?</h1>', landing, re.S).group(0)
abstract = re.search(r'<h2>Abstract</h2>\s*<p>.*?</p>', landing, re.S).group(0)
article = (head + '<body><a class="skip" href="#main">Skip to content</a><main id="main" class="full-text">'
           '<nav aria-label="Breadcrumb"><a href="../../">Publications</a> · <a href="./">Abstract and citation</a> · '
           '<a href="paper.pdf">PDF</a></nav>' + heading +
           '<p class="byline">Pierre Guilbert · September 22, 2026 · Preprint</p>' + abstract + body +
           '</main><footer><a href="./">Abstract and citation</a> · <a href="paper.pdf">PDF version</a></footer></body></html>')
(TARGET / 'full-text.html').write_text(article)
print('Generated full-text.html with', len(list(assets.glob('*.jpg'))), 'figures.')
