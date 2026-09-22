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
=\inf_{-\operatorname{div}(p_\theta v)=u}
  \int p_\theta\|v\|^2\,dx.
```

Fisher–Rao measures relative density change; Wasserstein–Otto measures the least transport energy needed to produce it. The figures illustrate metric costs, distances, geodesics, and velocity fields, distinguishing paths within a parametric family from paths in the full density space.

## Scripts and figures

```text
visualization_scripts/
├── *.py                 # Figure-generation scripts
├── *_caption.txt        # Corresponding captions
└── output/              # Generated figures and numerical diagnostics
```

Each script documents its dependencies and usage. For example:

```bash
python visualization_scripts/pushforward_shear_metrics.py
```

## Paper and citation

HAL and arXiv links will be added when available. If you use these scripts or figures, please cite the paper:

```bibtex
@misc{Guilbert2026FisherRaoWassersteinAtlas,
  author = {Pierre Guilbert},
  title  = {Fisher--Rao and Wasserstein--Otto Geometries of
            Parametric Probability Families: A Comparative Atlas},
  year   = {2026},
  note   = {Preprint}
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
