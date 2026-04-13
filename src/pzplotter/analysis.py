from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import sympy as sp
from scipy import signal


TimeSignalType = Literal["step", "impulse", "ramp", "parabolic"]


@dataclass
class LTIModel:
    """Container for transfer function data and precomputed roots."""

    numerator: np.ndarray
    denominator: np.ndarray
    zeros: np.ndarray
    poles: np.ndarray


@dataclass
class FrequencyResponse:
    """Frequency response data for plotting Bode magnitude and phase."""

    omega: np.ndarray
    magnitude_db: np.ndarray
    phase_deg: np.ndarray


@dataclass
class TimeResponse:
    """Time-domain response output for plotting."""

    t: np.ndarray
    y: np.ndarray
    signal_type: TimeSignalType


def parse_coefficients(raw: str) -> np.ndarray:
    """Parse comma/space-separated numeric coefficients into a numpy array."""

    if not raw.strip():
        raise ValueError("Coefficient field is empty.")

    tokens = raw.replace(",", " ").split()
    try:
        values = np.array([float(token) for token in tokens], dtype=float)
    except ValueError as exc:
        raise ValueError("Coefficients must be valid numbers.") from exc

    if np.allclose(values, 0.0):
        raise ValueError("All coefficients cannot be zero.")

    return trim_leading_zeros(values)


def parse_transfer_function_equation(raw: str, variable: str = "s") -> tuple[np.ndarray, np.ndarray]:
    """Parse a transfer-function equation string into numerator and denominator coefficients."""

    if not raw.strip():
        raise ValueError("Equation field is empty.")

    expression_text = raw.strip()
    if "=" in expression_text:
        expression_text = expression_text.split("=", maxsplit=1)[1].strip()

    expression_text = expression_text.replace("^", "**")
    symbol = sp.Symbol(variable)

    try:
        expression = sp.sympify(expression_text, locals={variable: symbol})
    except (sp.SympifyError, TypeError) as exc:
        raise ValueError("Could not parse equation. Use a valid symbolic expression.") from exc

    num_expr, den_expr = sp.fraction(sp.cancel(expression))

    try:
        num_poly = sp.Poly(sp.expand(num_expr), symbol)
        den_poly = sp.Poly(sp.expand(den_expr), symbol)
    except sp.PolynomialError as exc:
        raise ValueError("Equation must reduce to a polynomial ratio in the selected variable.") from exc

    if den_poly.is_zero:
        raise ValueError("Equation denominator cannot be zero.")

    num = _poly_to_coeffs(num_poly, allow_complex=False)
    den = _poly_to_coeffs(den_poly, allow_complex=False)
    return trim_leading_zeros(num), trim_leading_zeros(den)


def parse_transfer_function_equation_with_mode(
    raw: str,
    variable: str = "s",
    allow_complex: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a transfer-function equation string with optional complex coefficients."""

    if not raw.strip():
        raise ValueError("Equation field is empty.")

    expression_text = raw.strip()
    if "=" in expression_text:
        expression_text = expression_text.split("=", maxsplit=1)[1].strip()

    expression_text = expression_text.replace("^", "**")
    symbol = sp.Symbol(variable)

    try:
        expression = sp.sympify(expression_text, locals={variable: symbol})
    except (sp.SympifyError, TypeError) as exc:
        raise ValueError("Could not parse equation. Use a valid symbolic expression.") from exc

    num_expr, den_expr = sp.fraction(sp.cancel(expression))

    try:
        num_poly = sp.Poly(sp.expand(num_expr), symbol)
        den_poly = sp.Poly(sp.expand(den_expr), symbol)
    except sp.PolynomialError as exc:
        raise ValueError("Equation must reduce to a polynomial ratio in the selected variable.") from exc

    if den_poly.is_zero:
        raise ValueError("Equation denominator cannot be zero.")

    num = _poly_to_coeffs(num_poly, allow_complex=allow_complex)
    den = _poly_to_coeffs(den_poly, allow_complex=allow_complex)
    return trim_leading_zeros(num), trim_leading_zeros(den)


def trim_leading_zeros(values: np.ndarray) -> np.ndarray:
    """Remove leading zeros while preserving a valid polynomial array."""

    non_zero_indices = np.where(np.abs(values) > 1e-12)[0]
    if non_zero_indices.size == 0:
        return np.array([0.0], dtype=float)

    return values[non_zero_indices[0] :]


def _poly_to_coeffs(poly: sp.Poly, allow_complex: bool = False) -> np.ndarray:
    """Convert SymPy polynomial coefficients to a numpy array."""

    coeffs = np.array([complex(coeff.evalf()) for coeff in poly.all_coeffs()], dtype=complex)
    coeffs = np.real_if_close(coeffs, tol=1000)

    if np.iscomplexobj(coeffs) and not allow_complex:
        raise ValueError(
            "Equation produced complex coefficients. Use conjugate root pairs for real systems."
        )

    if allow_complex:
        return coeffs.astype(complex)
    return coeffs.astype(float)


def coefficients_from_roots(
    roots: np.ndarray,
    scale: float = 1.0,
    allow_complex: bool = False,
) -> np.ndarray:
    """Build polynomial coefficients from roots and a scalar gain."""

    if roots.size == 0:
        if allow_complex:
            return np.array([complex(scale)], dtype=complex)
        return np.array([float(scale)], dtype=float)

    coeffs = np.poly(roots) * scale
    coeffs = np.real_if_close(coeffs, tol=1000)
    if np.iscomplexobj(coeffs) and not allow_complex:
        raise ValueError("Roots must produce real coefficients for this real-valued model.")

    if allow_complex:
        return trim_leading_zeros(coeffs.astype(complex))
    return trim_leading_zeros(coeffs.astype(float))


def build_lti_model(num: np.ndarray, den: np.ndarray, allow_complex: bool = False) -> LTIModel:
    """Build an LTI model and compute poles and zeros from transfer function coefficients."""

    if den.size == 0 or np.allclose(den, 0.0):
        raise ValueError("Denominator must contain at least one non-zero coefficient.")

    dtype = complex if allow_complex else float
    num = trim_leading_zeros(np.array(num, dtype=dtype))
    den = trim_leading_zeros(np.array(den, dtype=dtype))

    if abs(den[0]) < 1e-12:
        raise ValueError("Denominator leading coefficient cannot be zero.")

    zeros = np.roots(num) if num.size > 1 else np.array([], dtype=complex)
    poles = np.roots(den) if den.size > 1 else np.array([], dtype=complex)

    return LTIModel(numerator=num, denominator=den, zeros=zeros, poles=poles)


def frequency_response(model: LTIModel, n_points: int = 800) -> FrequencyResponse:
    """Compute Bode magnitude and phase over logarithmic frequency samples."""

    system = signal.TransferFunction(model.numerator, model.denominator)
    omega = np.logspace(-2, 3, n_points)
    omega, magnitude_db, phase_deg = signal.bode(system, w=omega)
    return FrequencyResponse(omega=omega, magnitude_db=magnitude_db, phase_deg=phase_deg)


def _input_signal(t: np.ndarray, signal_type: TimeSignalType) -> np.ndarray:
    if signal_type == "step":
        return np.ones_like(t)
    if signal_type == "ramp":
        return t
    if signal_type == "parabolic":
        return 0.5 * t**2
    return np.zeros_like(t)


def time_response(
    model: LTIModel,
    signal_type: TimeSignalType = "step",
    t_final: float = 10.0,
    n_points: int = 1200,
) -> TimeResponse:
    """Compute response to standard inputs for control-system analysis."""

    if t_final <= 0:
        raise ValueError("Final time must be positive.")

    t = np.linspace(0.0, t_final, n_points)
    system = signal.TransferFunction(model.numerator, model.denominator)

    if signal_type == "impulse":
        t, y = signal.impulse(system, T=t)
    else:
        u = _input_signal(t, signal_type)
        t, y, _ = signal.lsim(system, U=u, T=t)

    return TimeResponse(t=t, y=y, signal_type=signal_type)


def stability_summary(model: LTIModel) -> str:
    """Return a short textual summary based on pole locations."""

    if model.poles.size == 0:
        return "No dynamic poles detected."

    real_parts = np.real(model.poles)
    if np.all(real_parts < 0):
        return "Asymptotically stable (all poles in left-half plane)."
    if np.any(real_parts > 0):
        return "Unstable (at least one pole in right-half plane)."
    return "Marginally stable (poles on imaginary axis, none in right-half plane)."
