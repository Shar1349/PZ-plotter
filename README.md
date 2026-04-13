# Interactive Pole-Zero Plotter

An interactive web-based tool for control systems and filter design analysis, featuring real-time pole-zero manipulation, system response visualization, and transfer function equation parsing.

## Features

- **Dual Perspective Analysis**
  - Control systems view: Step, impulse, and ramp response
  - Filter design view: Bode magnitude and phase plots
  
- **Interactive Pole-Zero Editing**
  - Edit poles and zeros directly in tables (real and imaginary parts)
  - Live recomputation of transfer function from edited roots
  - Immediate plot updates as you modify poles/zeros
  
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

Start the app from the project root with:

```bash
python -m streamlit run app.py
```

Streamlit will open a browser window (usually at `http://localhost:8501`). If it doesn't open automatically, copy the URL from the terminal.

### Quick Start
1. Use the default second-order system or load your own via coefficients or equation
2. Edit poles and zeros in the left and center tables to see real-time response updates
3. Adjust the simulation time horizon with the slider
4. Observe stability classification and transfer function coefficients

## Project Structure

```
PZ-plotter/
├── app.py                      # Main Streamlit app
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
└── src/
    └── pzplotter/
        ├── __init__.py
        └── analysis.py         # Core analysis module (LTI models, parsing, responses)
```

## Dependencies

- **streamlit** – Web app framework
- **numpy** – Numerical computing
- **scipy** – Scientific computing (control systems, signal processing)
- **matplotlib** – (installed as scipy dependency)
- **plotly** – Interactive plotting
- **sympy** – Symbolic math for equation parsing

See `requirements.txt` for pinned versions.

## Development

### Adding New Features
- Time-domain response plotting functions are in [src/pzplotter/analysis.py](src/pzplotter/analysis.py)
- UI layout and interactivity are in [app.py](app.py)
- Add new response types by extending the `TimeSignalType` literal and `_input_signal()` function

### Testing Locally
Run with verbose logging:
```bash
python -m streamlit run app.py --logger.level=debug
```

## Future Enhancements

- Drag-and-drop pole/zero manipulation directly on the plot canvas
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

**Q: `ModuleNotFoundError: No module named 'streamlit'`**
- Ensure you've activated the virtual environment: `source .venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

**Q: App won't open in browser**
- Check the terminal output for the URL (usually `http://localhost:8501`)
- If port 8501 is busy, specify a different one: `streamlit run app.py --server.port 8502`

**Q: Plots not updating after editing poles/zeros**
- Ensure you've completed your edits in the table and the app has recomputed
- Refresh the browser if needed (Ctrl+R or Cmd+R)

**Q: Import error with `pzplotter` module**
- Check that the working directory is the project root
- Verify `src/pzplotter/__init__.py` exists
