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
