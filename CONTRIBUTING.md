# Contributing

Thanks for your interest in improving Interactive Pole-Zero Plotter.

## Development Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Run the desktop app:

```bash
python desktop_app.py
```

## Before Opening a PR

1. Ensure the app launches and core interactions work (dragging, table edits, load system).
2. Run a syntax check:

```bash
python -m py_compile desktop_app.py src/pzplotter/analysis.py
```

3. Keep changes scoped and avoid unrelated reformatting.
4. Update README if behavior or usage changed.

## Branch and Commit Guidance

- Use descriptive branch names like `feature/pz-table-editing` or `fix/conjugate-sync`.
- Prefer small commits with clear messages.

## Pull Request Checklist

- Describe what changed and why.
- Include manual test steps.
- Include screenshots or short recordings for UI changes.
- Link related issues if available.

## License Note

By contributing, you agree that your contributions are provided under the same license as the project (GPL-3.0-or-later).
