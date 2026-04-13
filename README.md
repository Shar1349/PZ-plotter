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

- **Multiple Input Methods**
  - Numerator/denominator coefficients
  - Transfer function equations (e.g., `H(s) = (s + 1) / (s^2 + 1.4*s + 1)`)
  - Support for Laplace (s) or Z-transform (z) variables

- **System Analysis**
  - Stability classification from pole locations
  - Transfer function coefficients display
  - Adjustable simulation time horizon (1–40 seconds)

## Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Setup on Linux Mint XFCE

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
1. Use the default second-order system or load your own via coefficients or equation
2. Edit poles and zeros in the left and center tables to see real-time response updates
3. Adjust the simulation time horizon with the slider
4. Observe stability classification and transfer function coefficients

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

## Future Enhancements

- Discrete-time stability checks (z-plane, unit circle)
- Automatic transient metrics (rise time, settling time, overshoot)
- System step-response specifications editor
- Export plots to PDF/PNG

## License

[Add your license here, e.g., MIT, GPL-3.0]

## Author

[Your Name or Organization]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a pull request

## Troubleshooting

**Q: Want real point dragging**
- Use the desktop app: `python desktop_app.py`
- Click a pole or zero and drag it directly in the plot window

**Q: Plots not updating after editing poles/zeros**
- Ensure you've completed your edits in the table and the app has recomputed
- Refresh the browser if needed (Ctrl+R or Cmd+R)

**Q: Import error with `pzplotter` module**
- Check that the working directory is the project root
- Verify `src/pzplotter/__init__.py` exists
