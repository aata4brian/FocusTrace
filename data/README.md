# Data directory

Human-participant research data are intentionally not included in the public repository.

Do **not** commit raw webcam recordings, participant responses, consent documents, participant-level features, databases, or processed participant datasets here.

For engineering tests, generate synthetic grouped data using:

```powershell
python scripts\generate_demo_data.py --output outputs\synthetic\dataset.npz --participants 18 --seconds-per-block 8
```

See [../DATA_AVAILABILITY.md](../DATA_AVAILABILITY.md) for the public-data policy.
