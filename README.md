# Sinkhorn-HTI

Entropic substitute for HTI's inner-loop $c$-transform. ENSAE Optimal Transport course project, 2026.

## What it does

HTI (Amad & van der Schaar, ICLR 2026) fits a conditional optimal-transport model that predicts how a neural network's output distribution changes with its hyperparameters. Training requires a Kantorovich $c$-transform on every batch; under HTI's Lagrangian cost this has no closed form, so HTI solves it with an inner L-BFGS loop. Switching to entropic OT collapses that inner loop into a single LogSumExp (the soft $c$-transform), with the transport map read off as the entropic barycentric projection (Pooladian & Niles-Weed, 2021).

## Headline results

Semicircles benchmark (Amad & van der Schaar, 2026, §5.1) at paper budget (`N_outer=2001`, `N_inner=10`, `maxiter=10` for L-BFGS, seed 5).

|          | NLL ↓ |   CD ↓ | Wall (s) |
|----------|------:|-------:|---------:|
| L-BFGS   |  6.98 | 0.0158 |    12461 |
| Entropic | 12.05 | 0.0130 |     3808 |

CD is the mean deviation of held-out samples from the underlying semicircle. The entropic variant trains 3.3× faster.

See [`report/main.pdf`](report/main.pdf) for the writeup and [`sinkhorn_hti.ipynb`](sinkhorn_hti.ipynb) for the companion notebook.

## Layout

```
sinkhorn_hti.ipynb     notebook
report/main.pdf        report PDF
report/main.tex        report source
sinkhorn-hti/          Python package, tests, scripts, cached runs
```

## Reproducing

Python 3.11+, `jax`, `equinox`, `optax`, `cloudpickle`, `matplotlib`, `numpy`.

```
cd sinkhorn-hti
pytest tests/                              # ~85 s
python run_headline.py                     # ~5 h CPU, both methods
python scripts/regenerate_figs.py          # figures + numbers.tex
bash scripts/run_notebook.sh               # re-execute notebook
bash scripts/build_report.sh               # rebuild report PDF
```

## References

- Amad, H., & van der Schaar, M. *Hyperparameter Trajectory Inference with Conditional Lagrangian Optimal Transport*. ICLR 2026 (oral). [van der Schaar Lab @ ICLR 2026](https://www.vanderschaar-lab.com/iclr-2026/). Official code: [harrya32/hyperparameter-trajectory-inference](https://github.com/harrya32/hyperparameter-trajectory-inference).
- Pooladian, A.-A., & Niles-Weed, J. *Entropic estimation of optimal transport maps*. arXiv:2109.12004, 2021.

## License

[MIT](LICENSE).
