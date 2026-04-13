# Interactive Pole-Zero Plotter

A native desktop tool for control systems and filter design analysis, featuring real-time pole-zero dragging, system response visualization, and transfer function equation parsing.

## Features

- **Dual Perspective Analysis**
  - Control systems view: Step, impulse, and ramp response
  - Filter design view: Bode magnitude and phase plots

- **Interactive Pole-Zero Dragging**
  - Click and drag poles and zeros directly on the plot
  - Exact numeric coordinate editing for the selected point
  - Real-mode snapping and complex-mode freedom
  - Optional conjugate-pair mirroring for real-coefficient systems

- **Multiple Input Methods**
  - Numerator/denominator coefficients
  - Transfer function equations (e.g., `H(s) = (s + 1) / (s^2 + 1.4*s + 1)`)
  - Pole-zero table input with editable real/imaginary coordinates
  - Support for Laplace (s) or Z-transform (z) variables

- **System Analysis**
  - Stability classification from pole locations
  - Transfer function coefficients display
  - Adjustable simulation time horizon (1–40 seconds)
  - Control-system performance metrics including rise time, settling time, overshoot, damping ratio, and natural frequency
  - Canonical second-order prototype generation from damping ratio and natural frequency

## Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Setup on Windows (PowerShell)

1. **Clone or navigate to the project**
  ```powershell
  cd C:\path\to\PZ-plotter
  ```

2. **Create and activate a virtual environment**
  ```powershell
  py -3 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

3. **Upgrade pip and install dependencies**
  ```powershell
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  ```

### Setup on macOS (zsh/bash)

1. **Clone or navigate to the project**
  ```bash
  cd /path/to/PZ-plotter
  ```

2. **Create and activate a virtual environment**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

3. **Upgrade pip and install dependencies**
  ```bash
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  ```

### Setup on Linux (bash)

1. **Clone or navigate to the project**
   ```bash
   cd /path/to/PZ-plotter
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Upgrade pip and install dependencies**
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Usage

Run the Matplotlib desktop app from the project root with:

```bash
python desktop_app.py
```

This version opens a native desktop window and lets you click and drag poles/zeros directly on the plot.

### Quick Start
1. Start from the default second-order example or load your own system.
2. Choose an input mode: coefficients, equation, or pole-zero table.
3. Drag poles/zeros on the pole-zero map or edit selected points numerically.
4. Switch between Control systems and Signal processing analysis modes.
5. Use the Control design section to generate a second-order prototype from damping ratio and natural frequency.
6. Use visibility toggles and simulation horizon controls to focus analysis.

## Project Structure

```
PZ-plotter/
├── desktop_app.py              # Native Matplotlib/Tkinter desktop app
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
└── src/
    └── pzplotter/
        ├── __init__.py
        └── analysis.py         # Core analysis module (LTI models, parsing, responses)
```

## Dependencies

- **numpy** – Numerical computing
- **scipy** – Scientific computing (control systems, signal processing)
- **matplotlib** – Plotting and GUI interaction
- **sympy** – Symbolic math for equation parsing

See `requirements.txt` for pinned versions.

## Development

### Adding New Features
- Time-domain response plotting functions are in [src/pzplotter/analysis.py](src/pzplotter/analysis.py)
- Native drag-capable desktop UI is in [desktop_app.py](desktop_app.py)
- Add new response types by extending the `TimeSignalType` literal and `_input_signal()` function

### Testing Locally
Run the desktop app directly:
```bash
python desktop_app.py
```

Run a quick syntax check before committing:
```bash
python -m py_compile desktop_app.py src/pzplotter/analysis.py
```

## Future Enhancements

- Discrete-time stability checks (z-plane, unit circle)
- Automatic transient metrics (rise time, settling time, overshoot)
- System step-response specifications editor
- Export plots to PDF/PNG

## License

This project is licensed under the GNU General Public License v3.0 (or later).

This copyleft license helps keep redistributed modifications open-source, which aligns well with educational use and sharing.

See [LICENSE](LICENSE).

## Author

Maintained by repository contributors.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for workflow and PR guidance.

## Troubleshooting

**Q: Want real point dragging**
- Use the desktop app: `python desktop_app.py`
- Click a pole or zero and drag it directly in the plot window

**Q: Plots not updating after editing poles/zeros**
- Click **Update row** (table mode) and then **Load system**.
- Ensure conjugate pairs are valid in real-coefficient mode.

**Q: Import error with `pzplotter` module**
- Check that the working directory is the project root
- Verify `src/pzplotter/__init__.py` exists
