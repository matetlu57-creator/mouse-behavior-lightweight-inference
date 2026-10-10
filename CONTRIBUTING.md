# Contributing

Use Python 3.10 and install both runtime and development dependencies:

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Keep changes inside the current architecture boundary: detection and visual tracking belong to Ultralytics, RFID association belongs to mTrack or another external identity provider, and this repository owns trajectory normalization plus behavior analysis.

Before submitting a change, run:

```powershell
python scripts/run_quality.py --ci
```

Do not add recorded videos, generated caches, outputs, private RFID data, credentials, or unregistered model weights to Git. A change to the tracked-cache or identity-map schema requires tests and an ADR or an update to the existing identity-boundary ADR.
