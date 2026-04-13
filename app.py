from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from pzplotter.analysis import (
	LTIModel,
	build_lti_model,
	coefficients_from_roots,
	frequency_response,
	parse_coefficients,
	parse_transfer_function_equation,
	stability_summary,
	time_response,
)


st.set_page_config(page_title="Interactive Pole-Zero Plotter", layout="wide")


def _complex_rows(values: np.ndarray) -> list[dict[str, float]]:
	return [{"Re": float(np.real(v)), "Im": float(np.imag(v))} for v in values]


def _roots_from_rows(rows: list[dict[str, float]]) -> np.ndarray:
	roots: list[complex] = []
	for row in rows:
		re = row.get("Re", 0.0)
		im = row.get("Im", 0.0)
		if re in (None, ""):
			re = 0.0
		if im in (None, ""):
			im = 0.0
		roots.append(complex(float(re), float(im)))
	return np.array(roots, dtype=complex)


def _initialize_editor_state(model: LTIModel) -> None:
	st.session_state["poles_rows"] = _complex_rows(model.poles)
	st.session_state["zeros_rows"] = _complex_rows(model.zeros)
	st.session_state["gain"] = float(model.numerator[0])


def _build_model_from_editor() -> LTIModel:
	poles = _roots_from_rows(st.session_state.get("poles_rows", []))
	zeros = _roots_from_rows(st.session_state.get("zeros_rows", []))
	gain = float(st.session_state.get("gain", 1.0))

	numerator = coefficients_from_roots(zeros, scale=gain)
	denominator = coefficients_from_roots(poles, scale=1.0)
	return build_lti_model(numerator, denominator)


def _pole_zero_figure(model: LTIModel) -> go.Figure:
	fig = go.Figure()

	if model.poles.size:
		fig.add_trace(
			go.Scatter(
				x=np.real(model.poles),
				y=np.imag(model.poles),
				mode="markers",
				marker={"symbol": "x", "size": 12, "color": "#d62828", "line": {"width": 2}},
				name="Poles",
			)
		)

	if model.zeros.size:
		fig.add_trace(
			go.Scatter(
				x=np.real(model.zeros),
				y=np.imag(model.zeros),
				mode="markers",
				marker={"symbol": "circle-open", "size": 13, "color": "#1d3557", "line": {"width": 2}},
				name="Zeros",
			)
		)

	fig.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="#666")
	fig.add_vline(x=0.0, line_width=1, line_dash="dash", line_color="#666")
	fig.update_layout(
		title="Pole-Zero Map",
		xaxis_title="Real axis",
		yaxis_title="Imag axis",
		template="plotly_white",
		height=430,
		legend={"orientation": "h"},
	)
	fig.update_yaxes(scaleanchor="x", scaleratio=1)
	return fig


def _time_response_figure(model: LTIModel, t_final: float) -> go.Figure:
	step = time_response(model, signal_type="step", t_final=t_final)
	impulse = time_response(model, signal_type="impulse", t_final=t_final)
	ramp = time_response(model, signal_type="ramp", t_final=t_final)

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=step.t, y=step.y, mode="lines", name="Step", line={"color": "#264653"}))
	fig.add_trace(
		go.Scatter(x=impulse.t, y=impulse.y, mode="lines", name="Impulse", line={"color": "#e76f51"})
	)
	fig.add_trace(go.Scatter(x=ramp.t, y=ramp.y, mode="lines", name="Ramp", line={"color": "#2a9d8f"}))
	fig.update_layout(
		title="Control Perspective: Time Responses",
		xaxis_title="Time (s)",
		yaxis_title="Output",
		template="plotly_white",
		height=430,
	)
	return fig


def _bode_figures(model: LTIModel) -> tuple[go.Figure, go.Figure]:
	bode = frequency_response(model)

	mag = go.Figure()
	mag.add_trace(
		go.Scatter(x=bode.omega, y=bode.magnitude_db, mode="lines", line={"color": "#1d3557"}, name="|H(jw)|")
	)
	mag.update_layout(
		title="Filter Perspective: Bode Magnitude",
		xaxis_title="Angular frequency (rad/s)",
		yaxis_title="Magnitude (dB)",
		template="plotly_white",
		height=350,
	)
	mag.update_xaxes(type="log")

	phase = go.Figure()
	phase.add_trace(
		go.Scatter(x=bode.omega, y=bode.phase_deg, mode="lines", line={"color": "#e63946"}, name="Phase")
	)
	phase.update_layout(
		title="Filter Perspective: Bode Phase",
		xaxis_title="Angular frequency (rad/s)",
		yaxis_title="Phase (deg)",
		template="plotly_white",
		height=350,
	)
	phase.update_xaxes(type="log")
	return mag, phase


def _default_model() -> LTIModel:
	return build_lti_model(np.array([1.0]), np.array([1.0, 1.4, 1.0]))


st.title("Interactive Pole-Zero Plotter")
st.caption(
	"Control-system and filter-design analysis with editable poles/zeros, equation input, and live response plots."
)

with st.sidebar:
	st.subheader("System definition")
	input_mode = st.radio("Input method", ["Coefficients", "Equation"], horizontal=True)
	t_final = st.slider("Simulation horizon (s)", min_value=1.0, max_value=40.0, value=10.0, step=0.5)

	model: LTIModel | None = None
	error_message: str | None = None

	if input_mode == "Coefficients":
		num_text = st.text_input("Numerator", value="1")
		den_text = st.text_input("Denominator", value="1 1.4 1")
		if st.button("Load from coefficients", width="stretch"):
			try:
				num = parse_coefficients(num_text)
				den = parse_coefficients(den_text)
				model = build_lti_model(num, den)
			except ValueError as exc:
				error_message = str(exc)
	else:
		equation_text = st.text_input("Transfer function equation", value="H(s) = (s + 1) / (s^2 + 1.4*s + 1)")
		variable = st.selectbox("Variable", options=["s", "z"], index=0)
		if st.button("Load from equation", width="stretch"):
			try:
				num, den = parse_transfer_function_equation(equation_text, variable=variable)
				model = build_lti_model(num, den)
			except ValueError as exc:
				error_message = str(exc)

if "editor_initialized" not in st.session_state:
	base_model = _default_model()
	_initialize_editor_state(base_model)
	st.session_state["editor_initialized"] = True

if model is not None:
	_initialize_editor_state(model)

if error_message:
	st.error(error_message)

edit_col1, edit_col2, edit_col3 = st.columns([1.2, 1.2, 0.7])
with edit_col1:
	st.markdown("### Editable poles")
	poles_rows = st.data_editor(
		st.session_state.get("poles_rows", []),
		column_config={"Re": st.column_config.NumberColumn("Re", format="%.4f"), "Im": st.column_config.NumberColumn("Im", format="%.4f")},
		num_rows="dynamic",
		key="poles_editor",
	)
	st.session_state["poles_rows"] = poles_rows

with edit_col2:
	st.markdown("### Editable zeros")
	zeros_rows = st.data_editor(
		st.session_state.get("zeros_rows", []),
		column_config={"Re": st.column_config.NumberColumn("Re", format="%.4f"), "Im": st.column_config.NumberColumn("Im", format="%.4f")},
		num_rows="dynamic",
		key="zeros_editor",
	)
	st.session_state["zeros_rows"] = zeros_rows

with edit_col3:
	st.markdown("### Gain")
	st.session_state["gain"] = st.number_input(
		"K",
		value=float(st.session_state.get("gain", 1.0)),
		step=0.1,
		format="%.4f",
	)
	st.info("Edit Re/Im values to move poles and zeros and immediately update responses.")

analysis_error: str | None = None
active_model: LTIModel | None = None
try:
	active_model = _build_model_from_editor()
except ValueError as exc:
	analysis_error = str(exc)

if analysis_error:
	st.error(analysis_error)
else:
	summary_col, coeff_col = st.columns([1, 1])
	with summary_col:
		st.markdown("### Stability")
		st.write(stability_summary(active_model))
	with coeff_col:
		st.markdown("### Transfer function coefficients")
		st.write(f"Numerator: {np.array2string(active_model.numerator, precision=4)}")
		st.write(f"Denominator: {np.array2string(active_model.denominator, precision=4)}")

	pz_col, tr_col = st.columns(2)
	with pz_col:
		st.plotly_chart(_pole_zero_figure(active_model), width="stretch")
	with tr_col:
		st.plotly_chart(_time_response_figure(active_model, t_final=t_final), width="stretch")
	b1, b2 = st.columns(2)
	magnitude_fig, phase_fig = _bode_figures(active_model)
	with b1:
		st.plotly_chart(magnitude_fig, width="stretch")
	with b2:
		st.plotly_chart(phase_fig, width="stretch")
