# Fisher–Rao and Wasserstein–Otto: A Comparative Atlas

Python scripts, numerical experiments, and figures accompanying:

**Pierre Guilbert — *Fisher–Rao and Wasserstein–Otto Geometries of Parametric Probability Families: A Comparative Atlas* (2026).**

## Scope

The atlas compares the two metrics on the same probability families:

- one-dimensional location–scale families;
- multivariate Gaussian distributions;
- linear and nonlinear Gibbs families;
- push-forward families.

For a parameter direction inducing a density variation $u=d_\theta p_\theta(d\theta)$, the squared costs are

```math
g_\theta^{\mathrm{FR}}(d\theta,d\theta)
=\int\frac{u^2}{p_\theta}\,dx,
\qquad
g_\theta^{\mathrm{WO}}(d\theta,d\theta)
=\inf_{-\mathrm{div}(p_\theta v)=u}
  \int p_\theta\|v\|^2\,dx.
```

Fisher–Rao measures relative density change; Wasserstein–Otto measures the least transport energy needed to produce it. The figures illustrate metric costs, distances, geodesics, and velocity fields, distinguishing paths within a parametric family from paths in the full density space.

## Scripts and figures

```text
code/
├── *.py                 # Figure-generation scripts
└── *_caption.txt        # Corresponding captions
```

Generated figures and numerical diagnostics are in `figures/`. Each script documents its dependencies and usage. For example:

```bash
python code/pushforward_shear_metrics.py
```

## Paper and citation

[Read the paper and abstract](https://pierreguilbertf.github.io/fisher-rao-wasserstein-atlas/papers/atlas/) · [Zenodo DOI](https://doi.org/10.5281/zenodo.22921933)

If you use these scripts or figures, please cite the paper:

```bibtex
@misc{Guilbert2026FisherRaoWassersteinAtlas,
  author    = {Pierre Guilbert},
  title     = {Fisher--Rao and Wasserstein--Otto Geometries of
               Parametric Probability Families: A Comparative Atlas},
  year      = {2026},
  note      = {Preprint},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22921933},
  url       = {https://doi.org/10.5281/zenodo.22921933}
}
```

## Reuse and licensing

- **Source code:** MIT License.
- **Original figures and graphical material:** CC BY 4.0, unless otherwise stated.

Suggested figure attribution:

> Figure reproduced from P. Guilbert, *Fisher–Rao and Wasserstein–Otto Geometries of Parametric Probability Families: A Comparative Atlas*, 2026. CC BY 4.0.

For modified figures, replace “reproduced” with “adapted” and indicate the changes.

## Author

**Pierre Guilbert** — Independent researcher.

## Research website

The static site is in `docs/`. See [website setup](WEBSITE.md) for GitHub Pages activation, indexing checks, and how to add another publication. The site needs no build step or JavaScript.
