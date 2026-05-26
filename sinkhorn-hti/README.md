# Sinkhorn-HTI

Entropic substitute for HTI's inner-loop c-transform (Amad &
van der Schaar, ICLR 2026). ENSAE OT course project.

## Layout

```
sinkhorn_hti/         library
runs/                 cached headline payloads
scripts/              regenerate_figs.py + shell wrappers
tests/                pytest suite (~85s)
run_headline.py       reproduce both runs (~5h CPU)
```

## Use

```
conda activate inf554
pytest tests/                          # tests
python run_headline.py                 # reproduce runs
python scripts/regenerate_figs.py      # report figures + numbers.tex
bash scripts/run_notebook.sh           # re-execute notebook
bash scripts/build_report.sh           # rebuild report PDF
```
