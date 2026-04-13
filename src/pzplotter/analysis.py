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
class PoleMetric:
    """Per-pole modal information for control-system analysis."""

    pole: complex
    damping_ratio: float | None
    natural_frequency: float | None
    damped_frequency: float | None
    time_constant: float | None


@dataclass
class ControlSystemMetrics:
    """Step-response and pole-based performance metrics."""

    initial_value: float
    final_value: float
    peak_value: float
    peak_time: float
    minimum_value: float
    minimum_time: float
    overshoot_percent: float | None
    undershoot_percent: float | None
    rise_time: float | None
    settling_time_2pct: float | None
    settling_time_5pct: float | None
    steady_state_error: float | None
    dominant_pole: complex | None
    damping_ratio: float | None
    natural_frequency: float | None
    damped_frequency: float | None
    time_constant: float | None
    pole_metrics: list[PoleMetric]


@dataclass
class FilterDesignSpecs:
    """User-facing filter design specification for signal-processing mode."""

    family: Literal["Butterworth", "Chebyshev I", "Chebyshev II", "Elliptic", "Bessel"]
    response_type: Literal["lowpass", "highpass", "bandpass", "bandstop"]
    passband_edges: float | tuple[float, float]
    stopband_edges: float | tuple[float, float]
    passband_ripple_db: float
    stopband_attenuation_db: float
    order: int | None = None
    bessel_norm: Literal["phase", "delay"] = "phase"
    gain: float = 1.0


@dataclass
class FilterDesignMetrics:
    """Frequency-domain metrics for signal-processing filter design."""

    family: str
    response_type: str
    order: int
    passband_edges: float | tuple[float, float]
    stopband_edges: float | tuple[float, float]
    transition_band: float | tuple[float, float] | None
    cutoff_frequency_3db: float | tuple[float, float] | None
    peak_gain_db: float
    passband_ripple_db: float | None
    stopband_attenuation_db: float | None
    passband_gain_db: float | None
    stopband_gain_db: float | None


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


def _normalize_edges(edges: float | tuple[float, float], response_type: str) -> float | tuple[float, float]:
    if response_type in {"bandpass", "bandstop"}:
        if isinstance(edges, tuple):
            if len(edges) != 2:
                raise ValueError("Band filters require exactly two edge frequencies.")
            low, high = float(edges[0]), float(edges[1])
        else:
            raise ValueError("Band filters require two edge frequencies.")
        if low <= 0 or high <= 0 or low >= high:
            raise ValueError("Band edge frequencies must be positive and strictly increasing.")
        return (low, high)

    if isinstance(edges, tuple):
        if len(edges) != 1:
            raise ValueError("Low/high-pass filters require a single edge frequency.")
        value = float(edges[0])
    else:
        value = float(edges)
    if value <= 0:
        raise ValueError("Edge frequencies must be positive.")
    return value


def _design_iir_filter(
    order: int,
    wn: float | tuple[float, float],
    family: str,
    response_type: str,
    passband_ripple_db: float,
    stopband_attenuation_db: float,
    bessel_norm: str = "phase",
) -> tuple[np.ndarray, np.ndarray]:
    family_key = family.lower()
    if family_key == "butterworth":
        num, den = signal.iirfilter(order, wn, btype=response_type, analog=True, ftype="butter")
    elif family_key == "chebyshev i":
        num, den = signal.iirfilter(order, wn, rp=passband_ripple_db, btype=response_type, analog=True, ftype="cheby1")
    elif family_key == "chebyshev ii":
        num, den = signal.iirfilter(order, wn, rs=stopband_attenuation_db, btype=response_type, analog=True, ftype="cheby2")
    elif family_key == "elliptic":
        num, den = signal.iirfilter(
            order,
            wn,
            rp=passband_ripple_db,
            rs=stopband_attenuation_db,
            btype=response_type,
            analog=True,
            ftype="ellip",
        )
    elif family_key == "bessel":
        num, den = signal.bessel(order, wn, btype=response_type, analog=True, norm=bessel_norm)
    else:
        raise ValueError(f"Unsupported filter family: {family}")

    return np.asarray(num, dtype=float), np.asarray(den, dtype=float)


def design_filter_from_specs(specs: FilterDesignSpecs) -> LTIModel:
    """Design an analog IIR filter from passband/stopband specifications."""

    response_type = specs.response_type.lower()
    family = specs.family
    passband_edges = _normalize_edges(specs.passband_edges, response_type)
    stopband_edges = _normalize_edges(specs.stopband_edges, response_type)

    if specs.order is not None and specs.order <= 0:
        raise ValueError("Filter order must be positive.")

    if family.lower() == "bessel" and specs.order is None:
        raise ValueError("Bessel filters require an explicit order.")

    if specs.order is None:
        if family.lower() == "butterworth":
            order, wn = signal.buttord(passband_edges, stopband_edges, specs.passband_ripple_db, specs.stopband_attenuation_db, analog=True)
        elif family.lower() == "chebyshev i":
            order, wn = signal.cheb1ord(passband_edges, stopband_edges, specs.passband_ripple_db, specs.stopband_attenuation_db, analog=True)
        elif family.lower() == "chebyshev ii":
            order, wn = signal.cheb2ord(passband_edges, stopband_edges, specs.passband_ripple_db, specs.stopband_attenuation_db, analog=True)
        elif family.lower() == "elliptic":
            order, wn = signal.ellipord(passband_edges, stopband_edges, specs.passband_ripple_db, specs.stopband_attenuation_db, analog=True)
        else:
            raise ValueError("Automatic order selection is supported for Butterworth, Chebyshev I/II, and Elliptic filters.")
    else:
        order = specs.order
        wn = passband_edges

    numerator, denominator = _design_iir_filter(
        order,
        wn,
        family=family,
        response_type=response_type,
        passband_ripple_db=specs.passband_ripple_db,
        stopband_attenuation_db=specs.stopband_attenuation_db,
        bessel_norm=specs.bessel_norm,
    )
    if family.lower() == "bessel":
        numerator = numerator * float(specs.gain)
    else:
        numerator = numerator * float(specs.gain)
    return build_lti_model(numerator, denominator)


def _sample_filter_response(model: LTIModel, specs: FilterDesignSpecs | None = None, n_points: int = 2400) -> FrequencyResponse:
    if specs is None:
        return frequency_response(model, n_points=n_points)

    raw_edges = [specs.passband_edges, specs.stopband_edges]
    frequencies: list[float] = []
    for edge in raw_edges:
        if isinstance(edge, tuple):
            frequencies.extend([float(value) for value in edge])
        else:
            frequencies.append(float(edge))

    minimum = max(min(frequencies) / 20.0, 1e-3)
    maximum = max(frequencies) * 20.0
    omega = np.logspace(np.log10(minimum), np.log10(maximum), n_points)
    system = signal.TransferFunction(model.numerator, model.denominator)
    omega, magnitude_db, phase_deg = signal.bode(system, w=omega)
    return FrequencyResponse(omega=omega, magnitude_db=magnitude_db, phase_deg=phase_deg)


def _frequency_windows(specs: FilterDesignSpecs) -> tuple[tuple[float, float], tuple[float, float]]:
    response_type = specs.response_type.lower()
    passband = _normalize_edges(specs.passband_edges, response_type)
    stopband = _normalize_edges(specs.stopband_edges, response_type)
    if isinstance(passband, tuple):
        passband_window = passband
    else:
        if response_type == "lowpass":
            passband_window = (1e-6, passband)
        else:
            passband_window = (passband, passband * 100.0)
    if isinstance(stopband, tuple):
        stopband_window = stopband
    else:
        if response_type == "lowpass":
            stopband_window = (stopband, stopband * 100.0)
        else:
            stopband_window = (1e-6, stopband)
    return passband_window, stopband_window


def _crossing_frequency(omega: np.ndarray, magnitude_db: np.ndarray, threshold_db: float, rising: bool) -> float | None:
    if rising:
        indices = np.where(magnitude_db >= threshold_db)[0]
    else:
        indices = np.where(magnitude_db <= threshold_db)[0]
    if indices.size == 0:
        return None
    return float(omega[indices[0]])


def filter_design_metrics(model: LTIModel, specs: FilterDesignSpecs, n_points: int = 2400) -> FilterDesignMetrics:
    """Compute response metrics for a designed analog filter."""

    response = _sample_filter_response(model, specs=specs, n_points=n_points)
    omega = response.omega
    magnitude_db = response.magnitude_db

    passband_window, stopband_window = _frequency_windows(specs)
    response_type = specs.response_type.lower()

    if response_type in {"lowpass", "highpass"}:
        p_edge = passband_window[1] if response_type == "lowpass" else passband_window[0]
        s_edge = stopband_window[0] if response_type == "lowpass" else stopband_window[1]
        if response_type == "lowpass":
            passband_mask = omega <= p_edge
            stopband_mask = omega >= s_edge
        else:
            passband_mask = omega >= p_edge
            stopband_mask = omega <= s_edge
    elif response_type == "bandpass":
        passband_mask = (omega >= passband_window[0]) & (omega <= passband_window[1])
        stopband_mask = (omega <= stopband_window[0]) | (omega >= stopband_window[1])
    else:
        passband_mask = (omega <= passband_window[0]) | (omega >= passband_window[1])
        stopband_mask = (omega >= stopband_window[0]) & (omega <= stopband_window[1])

    passband_values = magnitude_db[passband_mask]
    stopband_values = magnitude_db[stopband_mask]

    peak_gain_db = float(np.max(magnitude_db)) if magnitude_db.size else float("nan")
    passband_ripple_db = float(np.max(passband_values) - np.min(passband_values)) if passband_values.size else None
    stopband_attenuation_db = float(-np.max(stopband_values)) if stopband_values.size else None
    passband_gain_db = float(np.max(passband_values)) if passband_values.size else None
    stopband_gain_db = float(np.max(stopband_values)) if stopband_values.size else None

    cutoff_frequency_3db: float | tuple[float, float] | None = None
    threshold_db = peak_gain_db - 3.0
    if response_type == "lowpass":
        cutoff_frequency_3db = _crossing_frequency(omega, magnitude_db, threshold_db, rising=False)
    elif response_type == "highpass":
        cutoff_frequency_3db = _crossing_frequency(omega, magnitude_db, threshold_db, rising=True)
    elif response_type in {"bandpass", "bandstop"}:
        crossings = np.where(np.diff((magnitude_db >= threshold_db).astype(int)) != 0)[0]
        if crossings.size >= 2:
            cutoff_frequency_3db = (float(omega[crossings[0]]), float(omega[crossings[-1] + 1]))

    if response_type in {"lowpass", "highpass"}:
        transition_band: float | tuple[float, float] | None = abs(float(specs.stopband_edges) - float(specs.passband_edges))
    else:
        transition_band = (
            float(abs(specs.passband_edges[0] - specs.stopband_edges[0])),
            float(abs(specs.stopband_edges[1] - specs.passband_edges[1])),
        )

    return FilterDesignMetrics(
        family=specs.family,
        response_type=specs.response_type,
        order=int(len(model.poles)) if model.poles.size else 0,
        passband_edges=specs.passband_edges,
        stopband_edges=specs.stopband_edges,
        transition_band=transition_band,
        cutoff_frequency_3db=cutoff_frequency_3db,
        peak_gain_db=peak_gain_db,
        passband_ripple_db=passband_ripple_db,
        stopband_attenuation_db=stopband_attenuation_db,
        passband_gain_db=passband_gain_db,
        stopband_gain_db=stopband_gain_db,
    )


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


def second_order_model(damping_ratio: float, natural_frequency: float, gain: float = 1.0) -> LTIModel:
    """Build a canonical second-order control-system model from common specs."""

    if natural_frequency <= 0:
        raise ValueError("Natural frequency must be positive.")
    if damping_ratio < 0:
        raise ValueError("Damping ratio must be non-negative.")

    numerator = np.array([gain * natural_frequency**2], dtype=float)
    denominator = np.array([1.0, 2.0 * damping_ratio * natural_frequency, natural_frequency**2], dtype=float)
    return build_lti_model(numerator, denominator)


def _settling_time(t: np.ndarray, y: np.ndarray, target: float, tolerance: float) -> float | None:
    if not len(t):
        return None
    band = max(abs(target) * tolerance, tolerance)
    outside = np.where(np.abs(y - target) > band)[0]
    if outside.size == 0:
        return float(t[0])
    last_outside = int(outside[-1])
    if last_outside >= len(t) - 1:
        return None
    return float(t[last_outside + 1])


def _rise_time(t: np.ndarray, y: np.ndarray, start_value: float, final_value: float) -> float | None:
    delta = final_value - start_value
    if abs(delta) < 1e-12:
        return None

    lower = start_value + 0.1 * delta
    upper = start_value + 0.9 * delta

    if delta > 0:
        lower_hits = np.where(y >= lower)[0]
        upper_hits = np.where(y >= upper)[0]
    else:
        lower_hits = np.where(y <= lower)[0]
        upper_hits = np.where(y <= upper)[0]

    if lower_hits.size == 0 or upper_hits.size == 0:
        return None
    return float(t[upper_hits[0]] - t[lower_hits[0]])


def _modal_metric_for_pole(pole: complex) -> PoleMetric:
    if abs(pole) < 1e-12:
        return PoleMetric(pole=pole, damping_ratio=None, natural_frequency=0.0, damped_frequency=0.0, time_constant=None)

    natural_frequency = float(abs(pole))
    damped_frequency = float(abs(np.imag(pole)))
    if np.isclose(np.imag(pole), 0.0):
        damping_ratio = 1.0
    else:
        damping_ratio = float(-np.real(pole) / natural_frequency)

    time_constant = None
    if np.real(pole) < 0:
        time_constant = float(1.0 / abs(np.real(pole)))

    return PoleMetric(
        pole=pole,
        damping_ratio=damping_ratio,
        natural_frequency=natural_frequency,
        damped_frequency=damped_frequency,
        time_constant=time_constant,
    )


def control_system_metrics(model: LTIModel, t_final: float = 10.0, n_points: int = 2400) -> ControlSystemMetrics:
    """Compute common control-system performance specifications from the model."""

    if t_final <= 0:
        raise ValueError("Final time must be positive.")

    if model.denominator.size == 0:
        raise ValueError("Model denominator is empty.")

    response = time_response(model, signal_type="step", t_final=t_final, n_points=n_points)
    t = np.asarray(response.t, dtype=float)
    y = np.real_if_close(np.asarray(response.y))
    y = np.real(y)

    initial_value = float(y[0])
    peak_index = int(np.argmax(y)) if y.size else 0
    minimum_index = int(np.argmin(y)) if y.size else 0
    peak_value = float(y[peak_index])
    minimum_value = float(y[minimum_index])
    peak_time = float(t[peak_index]) if y.size else 0.0
    minimum_time = float(t[minimum_index]) if y.size else 0.0

    stable = bool(model.poles.size == 0 or np.all(np.real(model.poles) < 0))
    final_value_theoretical: float | None = None
    if stable and abs(model.denominator[-1]) > 1e-12:
        final_value_theoretical = float(np.real_if_close(np.polyval(model.numerator, 0.0) / np.polyval(model.denominator, 0.0)))

    final_value = final_value_theoretical if final_value_theoretical is not None else float(y[-1])

    overshoot_percent = None
    undershoot_percent = None
    if abs(final_value) > 1e-12:
        overshoot_percent = max(0.0, (peak_value - final_value) / abs(final_value) * 100.0)
        undershoot_percent = max(0.0, (final_value - minimum_value) / abs(final_value) * 100.0)

    rise_time = _rise_time(t, y, initial_value, final_value)
    settling_time_2pct = _settling_time(t, y, final_value, 0.02)
    settling_time_5pct = _settling_time(t, y, final_value, 0.05)
    steady_state_error = abs(final_value - final_value_theoretical) if final_value_theoretical is not None else None

    pole_metrics = [_modal_metric_for_pole(pole) for pole in model.poles]
    dominant_pole = None
    damping_ratio = None
    natural_frequency = None
    damped_frequency = None
    time_constant = None

    stable_poles = [pole for pole in model.poles if np.real(pole) < 0]
    if stable_poles:
        dominant_pole = max(stable_poles, key=lambda pole: np.real(pole))
        dominant_metric = _modal_metric_for_pole(dominant_pole)
        damping_ratio = dominant_metric.damping_ratio
        natural_frequency = dominant_metric.natural_frequency
        damped_frequency = dominant_metric.damped_frequency
        time_constant = dominant_metric.time_constant

    return ControlSystemMetrics(
        initial_value=initial_value,
        final_value=final_value,
        peak_value=peak_value,
        peak_time=peak_time,
        minimum_value=minimum_value,
        minimum_time=minimum_time,
        overshoot_percent=overshoot_percent,
        undershoot_percent=undershoot_percent,
        rise_time=rise_time,
        settling_time_2pct=settling_time_2pct,
        settling_time_5pct=settling_time_5pct,
        steady_state_error=steady_state_error,
        dominant_pole=dominant_pole,
        damping_ratio=damping_ratio,
        natural_frequency=natural_frequency,
        damped_frequency=damped_frequency,
        time_constant=time_constant,
        pole_metrics=pole_metrics,
    )


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
