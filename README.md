# Post-Detection Anganwadi–Healthcare Referral DES

This project is a modular Discrete Event Simulation of the post-detection
MAM/SAM treatment and referral pathway.

## Model boundary

The model starts only after a child has already been detected as MAM or SAM.

It does not model:

- healthy-to-malnutrition incidence,
- screening coverage,
- screening accuracy,
- disease detection.

It models:

- community treatment,
- medical-complication routing,
- referral generation,
- family referral compliance,
- facility-arrival delays,
- NRC bed queueing,
- NRC treatment,
- return to community care,
- follow-up attendance,
- recovery, dropout and non-response,
- operational bottlenecks.

## Setup in VS Code

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Install packages:

```bash
pip install -r requirements.txt
```

Run:

```bash
python run_simulation.py
```

Generated files will appear in `outputs/`.

The default command preserves the 1,000-child run as the
"Low-demand / original baseline", searches the predefined demand-load range
with 30 replications per load, selects the lowest load with mean NRC
utilisation in the 70–90% band, and compares 5, 10, 12 and 20 beds with 100
common-seed replications. Use `--search-replications 50` or
`--search-replications 100` when runtime permits. The optional
`--include-existing-analyses` flag also regenerates the verified scenario,
sensitivity and ML CSV workflow.

This calibration matches simulated facility workload to a literature-informed
occupancy range; it does not imply that the selected number of detected
children is a national epidemiological estimate.

## Main files

- `config.py`: all model inputs
- `entities.py`: child states and entity structure
- `distributions.py`: reproducible random sampling
- `model.py`: SimPy process logic
- `metrics.py`: output metrics
- `validation.py`: logical validation checks
- `scenarios.py`: scenario and replication runner
- `visualization.py`: plots
- `run_simulation.py`: executable entry point

## Important modelling note

Some clinical and operational transition probabilities in `config.py` are
explicit modelling assumptions for sensitivity analysis. They must not be
presented as observed evidence unless separately supported by literature or
programme data.
