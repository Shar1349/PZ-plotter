from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from pzplotter.analysis import (  # noqa: E402
	LTIModel,
	build_lti_model,
	coefficients_from_roots,
	parse_coefficients,
	parse_transfer_function_equation_with_mode,
	stability_summary,
)


def _snap_value(value: float, step: float) -> float:
	if step <= 0:
		return float(value)
	return float(round(value / step) * step)


class ToolTip:
	def __init__(self, widget: tk.Widget, text: str) -> None:
		self.widget = widget
		self.text = text
		self.tip_window: tk.Toplevel | None = None
		widget.bind("<Enter>", self._show)
		widget.bind("<Leave>", self._hide)
		widget.bind("<ButtonPress>", self._hide)

	def _show(self, _event=None) -> None:
		if self.tip_window is not None:
			return
		x = self.widget.winfo_rootx() + 18
		y = self.widget.winfo_rooty() + 18
		self.tip_window = tw = tk.Toplevel(self.widget)
		tw.wm_overrideredirect(True)
		tw.wm_geometry(f"+{x}+{y}")
		label = tk.Label(
			tw,
			text=self.text,
			justify="left",
			relief="solid",
			borderwidth=1,
			background="#fff7cc",
			foreground="#222222",
			wraplength=300,
			font=("TkDefaultFont", 9),
		)
		label.pack(ipadx=6, ipady=4)

	def _hide(self, _event=None) -> None:
		if self.tip_window is not None:
			self.tip_window.destroy()
			self.tip_window = None


class CollapsibleSection:
	def __init__(self, master: tk.Widget, title: str, expanded: bool = True) -> None:
		self.title = title
		self.expanded = expanded
		self.container = ttk.Frame(master)
		self.container.columnconfigure(0, weight=1)
		self._header = ttk.Button(self.container, text=self._label_text(), command=self.toggle, style="Toolbutton")
		self._header.grid(row=0, column=0, sticky="ew")
		self.content = ttk.Frame(self.container, padding=(10, 8, 10, 10))
		if self.expanded:
			self.content.grid(row=1, column=0, sticky="ew")

	def _label_text(self) -> str:
		return f"{'▼' if self.expanded else '►'} {self.title}"

	def toggle(self) -> None:
		self.expanded = not self.expanded
		self._header.configure(text=self._label_text())
		if self.expanded:
			self.content.grid(row=1, column=0, sticky="ew")
		else:
			self.content.grid_remove()


class PoleZeroDesktopApp:
	def __init__(self) -> None:
		self.root = tk.Tk()
		self.root.title("Interactive Pole-Zero Plotter")
		self.root.geometry("1600x980")

		self.allow_complex = tk.BooleanVar(value=False)
		self.input_mode = tk.StringVar(value="Coefficients")
		self.mirror_conjugates = tk.BooleanVar(value=True)
		self.snap_to_grid = tk.BooleanVar(value=True)
		self.grid_step = tk.DoubleVar(value=0.01)
		self.sim_time = tk.DoubleVar(value=10.0)
		self.analysis_mode_text = tk.StringVar(value="Control systems")
		self.show_step_response = tk.BooleanVar(value=True)
		self.show_impulse_response = tk.BooleanVar(value=True)
		self.show_ramp_response = tk.BooleanVar(value=True)
		self.show_bode_magnitude = tk.BooleanVar(value=True)
		self.show_bode_phase = tk.BooleanVar(value=True)
		self.selected_kind: str | None = None
		self.selected_index: int | None = None
		self.selected_pair_index: int | None = None
		self.dragging = False
		self.drag_threshold = 0.35
		self.num_text = tk.StringVar(value="1 1")
		self.den_text = tk.StringVar(value="1 1.4 1")
		self.equation_text = tk.StringVar(value="H(s) = (s + 1) / (s^2 + 1.4*s + 1)")
		self.variable_text = tk.StringVar(value="s")
		self.selected_kind_text = tk.StringVar(value="None")
		self.selected_index_text = tk.StringVar(value="-")
		self.selected_re_text = tk.StringVar(value="0.0")
		self.selected_im_text = tk.StringVar(value="0.0")
		self.status_text = tk.StringVar(value="Ready.")
		self.model_mode_text = tk.StringVar(value="Real coefficients")
		self.snap_info_text = tk.StringVar(value="Grid snap active.")
		self.equation_display_text = tk.StringVar(value="H(s) = ...")

		self.model: LTIModel = build_lti_model(np.array([1.0, 1.0]), np.array([1.0, 1.4, 1.0]))
		self.poles = self.model.poles.astype(complex).copy()
		self.zeros = self.model.zeros.astype(complex).copy()
		self.gain = float(np.real_if_close(self.model.numerator[0]).real)

		self._build_ui()
		self._connect_events()
		self._refresh_from_state()

	def _build_ui(self) -> None:
		self.root.columnconfigure(1, weight=1)
		self.root.rowconfigure(0, weight=1)

		sidebar_container = ttk.Frame(self.root, padding=(12, 12, 0, 12))
		sidebar_container.grid(row=0, column=0, sticky="nsew")
		sidebar_container.rowconfigure(0, weight=1)
		sidebar_container.columnconfigure(0, weight=1)
		self.sidebar_container = sidebar_container

		sidebar_canvas = tk.Canvas(sidebar_container, highlightthickness=0, borderwidth=0)
		sidebar_scrollbar = ttk.Scrollbar(sidebar_container, orient="vertical", command=sidebar_canvas.yview)
		sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)
		sidebar_canvas.grid(row=0, column=0, sticky="nsew")
		sidebar_scrollbar.grid(row=0, column=1, sticky="ns")
		self.sidebar_canvas = sidebar_canvas

		sidebar_content = ttk.Frame(sidebar_canvas)
		sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar_content, anchor="nw")
		sidebar_content.bind(
			"<Configure>",
			lambda _event: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")),
		)
		sidebar_canvas.bind(
			"<Configure>",
			lambda event: sidebar_canvas.itemconfigure(sidebar_window, width=event.width),
		)
		self.root.bind_all("<MouseWheel>", self._on_global_mousewheel)
		self.root.bind_all("<Button-4>", self._on_global_mousewheel)
		self.root.bind_all("<Button-5>", self._on_global_mousewheel)

		controls = ttk.Frame(sidebar_content, padding=0)
		controls.grid(row=0, column=0, sticky="nsew")
		controls.columnconfigure(0, weight=1)

		plot_frame = ttk.Frame(self.root, padding=(0, 12, 12, 12))
		plot_frame.grid(row=0, column=1, sticky="nsew")
		plot_frame.rowconfigure(0, weight=1)
		plot_frame.rowconfigure(1, weight=0)
		plot_frame.columnconfigure(0, weight=1)

		analysis_section = CollapsibleSection(controls, "Analysis mode", expanded=True)
		analysis_section.container.grid(row=0, column=0, sticky="ew", pady=(0, 10))
		analysis_mode_box = ttk.LabelFrame(analysis_section.content, text="", padding=10)
		analysis_mode_box.grid(row=0, column=0, sticky="ew")
		control_radio = ttk.Radiobutton(
			analysis_mode_box,
			text="Control systems",
			value="Control systems",
			variable=self.analysis_mode_text,
			command=self._on_analysis_mode_changed,
		)
		control_radio.grid(sticky="w")
		signal_radio = ttk.Radiobutton(
			analysis_mode_box,
			text="Signal processing",
			value="Signal processing",
			variable=self.analysis_mode_text,
			command=self._on_analysis_mode_changed,
		)
		signal_radio.grid(sticky="w")
		ToolTip(control_radio, "Show time-domain response analysis first.")
		ToolTip(signal_radio, "Show filter/frequency-response analysis first.")

		mode_section = CollapsibleSection(controls, "Coefficient mode", expanded=True)
		mode_section.container.grid(row=1, column=0, sticky="ew", pady=(0, 10))
		mode_box = ttk.LabelFrame(mode_section.content, text="", padding=10)
		mode_box.grid(row=0, column=0, sticky="ew")
		mode_real = ttk.Radiobutton(mode_box, text="Real coefficients", value="Real coefficients", variable=self.model_mode_text, command=self._on_mode_changed)
		mode_real.grid(sticky="w")
		mode_complex = ttk.Radiobutton(mode_box, text="Complex coefficients", value="Complex coefficients", variable=self.model_mode_text, command=self._on_mode_changed)
		mode_complex.grid(sticky="w")

		input_section = CollapsibleSection(controls, "Input", expanded=True)
		input_section.container.grid(row=2, column=0, sticky="ew", pady=(0, 10))
		input_box = ttk.LabelFrame(input_section.content, text="", padding=10)
		input_box.grid(row=0, column=0, sticky="ew")
		input_coeff = ttk.Radiobutton(input_box, text="Coefficients", value="Coefficients", variable=self.input_mode, command=self._on_input_mode_changed)
		input_coeff.grid(sticky="w")
		input_equation = ttk.Radiobutton(input_box, text="Equation", value="Equation", variable=self.input_mode, command=self._on_input_mode_changed)
		input_equation.grid(sticky="w")
		input_pz = ttk.Radiobutton(input_box, text="Pole-zero table", value="Pole-zero table", variable=self.input_mode, command=self._on_input_mode_changed)
		input_pz.grid(sticky="w")

		self.coeff_fields_frame = ttk.Frame(input_box)
		self.coeff_fields_frame.grid(row=3, column=0, sticky="ew")
		self.coeff_fields_frame.columnconfigure(0, weight=1)
		ttk.Label(self.coeff_fields_frame, text="Numerator").grid(row=0, column=0, sticky="w", pady=(8, 2))
		self.num_entry = ttk.Entry(self.coeff_fields_frame, textvariable=self.num_text, width=34)
		self.num_entry.grid(row=1, column=0, sticky="ew")
		ttk.Label(self.coeff_fields_frame, text="Denominator").grid(row=2, column=0, sticky="w", pady=(8, 2))
		self.den_entry = ttk.Entry(self.coeff_fields_frame, textvariable=self.den_text, width=34)
		self.den_entry.grid(row=3, column=0, sticky="ew")

		self.equation_fields_frame = ttk.Frame(input_box)
		self.equation_fields_frame.grid(row=4, column=0, sticky="ew")
		self.equation_fields_frame.columnconfigure(0, weight=1)
		ttk.Label(self.equation_fields_frame, text="Equation").grid(row=0, column=0, sticky="w", pady=(8, 2))
		self.equation_entry = ttk.Entry(self.equation_fields_frame, textvariable=self.equation_text, width=34)
		self.equation_entry.grid(row=1, column=0, sticky="ew")
		ttk.Label(self.equation_fields_frame, text="Variable").grid(row=2, column=0, sticky="w", pady=(8, 2))
		self.variable_combo = ttk.Combobox(self.equation_fields_frame, textvariable=self.variable_text, values=("s", "z"), width=4, state="readonly")
		self.variable_combo.grid(row=3, column=0, sticky="w")

		self.pz_table_frame = ttk.Frame(input_box)
		self.pz_table_frame.grid(row=5, column=0, sticky="ew")
		self.pz_table_frame.columnconfigure(0, weight=1)
		self.pz_tree = ttk.Treeview(self.pz_table_frame, columns=("Type", "Real", "Imag"), show="headings", height=7, selectmode="browse")
		for column, width in (("Type", 80), ("Real", 100), ("Imag", 100)):
			self.pz_tree.heading(column, text=column)
			self.pz_tree.column(column, width=width, anchor="center")
		self.pz_tree.grid(row=0, column=0, sticky="ew")
		pz_scroll = ttk.Scrollbar(self.pz_table_frame, orient="vertical", command=self.pz_tree.yview)
		pz_scroll.grid(row=0, column=1, sticky="ns")
		self.pz_tree.configure(yscrollcommand=pz_scroll.set)
		self.pz_tree.bind("<<TreeviewSelect>>", self._on_pz_tree_select)
		self.pz_table_frame.bind("<Configure>", self._on_pz_table_configure)

		pz_button_row = ttk.Frame(self.pz_table_frame)
		pz_button_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
		add_pole_button = ttk.Button(pz_button_row, text="Add pole", command=lambda: self._add_pz_row("pole"))
		add_pole_button.grid(row=0, column=0, padx=(0, 4))
		add_zero_button = ttk.Button(pz_button_row, text="Add zero", command=lambda: self._add_pz_row("zero"))
		add_zero_button.grid(row=0, column=1, padx=(0, 4))
		remove_row_button = ttk.Button(pz_button_row, text="Remove selected", command=self._remove_selected_pz_row)
		remove_row_button.grid(row=0, column=2, padx=(0, 4))
		ToolTip(add_pole_button, "Insert a new pole row into the table.")
		ToolTip(add_zero_button, "Insert a new zero row into the table.")
		ToolTip(remove_row_button, "Remove the selected table row.")

		pz_edit_row = ttk.LabelFrame(self.pz_table_frame, text="Selected row", padding=8)
		pz_edit_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
		self.pz_type_text = tk.StringVar(value="pole")
		self.pz_real_text = tk.StringVar(value="0.0")
		self.pz_imag_text = tk.StringVar(value="0.0")
		ttk.Label(pz_edit_row, text="Type").grid(row=0, column=0, sticky="w")
		self.pz_type_combo = ttk.Combobox(pz_edit_row, textvariable=self.pz_type_text, values=("pole", "zero"), width=8, state="readonly")
		self.pz_type_combo.grid(row=0, column=1, sticky="w", padx=(8, 12))
		ttk.Label(pz_edit_row, text="Real").grid(row=0, column=2, sticky="w")
		self.pz_real_entry = ttk.Entry(pz_edit_row, textvariable=self.pz_real_text, width=12)
		self.pz_real_entry.grid(row=0, column=3, sticky="w", padx=(8, 12))
		ttk.Label(pz_edit_row, text="Imag").grid(row=0, column=4, sticky="w")
		self.pz_imag_entry = ttk.Entry(pz_edit_row, textvariable=self.pz_imag_text, width=12)
		self.pz_imag_entry.grid(row=0, column=5, sticky="w", padx=(8, 0))
		update_row_button = ttk.Button(pz_edit_row, text="Update row", command=self._update_selected_pz_row)
		update_row_button.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(8, 0))
		ToolTip(update_row_button, "Write the selected type and coordinates back into the table.")

		load_button = ttk.Button(input_box, text="Load system", command=self._load_system)
		load_button.grid(row=6, column=0, sticky="ew", pady=(10, 0))

		edit_section = CollapsibleSection(controls, "Selected point", expanded=True)
		edit_section.container.grid(row=3, column=0, sticky="ew", pady=(0, 10))
		edit_box = ttk.LabelFrame(edit_section.content, text="", padding=10)
		edit_box.grid(row=0, column=0, sticky="ew")
		for row, (label, var) in enumerate(
			(
				("Kind", self.selected_kind_text),
				("Index", self.selected_index_text),
				("Real", self.selected_re_text),
				("Imag", self.selected_im_text),
			)
		):
			ttk.Label(edit_box, text=label).grid(row=row, column=0, sticky="w", pady=2)
			if label in ("Real", "Imag"):
				entry = ttk.Entry(edit_box, textvariable=var, width=14)
				entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=2)
				ToolTip(entry, f"Exact {label.lower()} coordinate for the selected root.")
			else:
				ttk.Label(edit_box, textvariable=var).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=2)
		apply_button = ttk.Button(edit_box, text="Apply exact values", command=self._apply_exact_values)
		apply_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 4))
		ToolTip(apply_button, "Write the current Real and Imag values into the selected root.")
		step_row = ttk.Frame(edit_box)
		step_row.grid(row=5, column=0, columnspan=2, sticky="ew")
		for col, (text, dx, dy) in enumerate((
			("←", -1, 0),
			("→", 1, 0),
			("↓", 0, -1),
			("↑", 0, 1),
		)):
			button = ttk.Button(step_row, text=text, width=4, command=lambda dx=dx, dy=dy: self._nudge_selected(dx, dy))
			button.grid(row=0, column=col, padx=2)
			ToolTip(button, f"Move the selected root {text} by one nudge step.")
		spin_row = ttk.Frame(edit_box)
		spin_row.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(6, 0))
		ttk.Label(spin_row, text="Nudge").grid(row=0, column=0, sticky="w")
		nudge_spin = ttk.Spinbox(spin_row, from_=0.001, to=1.0, increment=0.001, textvariable=self.grid_step, width=10)
		nudge_spin.grid(row=0, column=1, sticky="w", padx=(8, 0))
		ToolTip(nudge_spin, "Set how far each arrow button moves the selected root.")

		equation_section = CollapsibleSection(controls, "Derived equation", expanded=True)
		equation_section.container.grid(row=4, column=0, sticky="ew", pady=(0, 10))
		equation_box = ttk.LabelFrame(equation_section.content, text="", padding=10)
		equation_box.grid(row=0, column=0, sticky="ew")
		self.equation_text_widget = tk.Text(equation_box, height=7, width=38, wrap="word")
		self.equation_text_widget.grid(row=0, column=0, sticky="ew")
		self.equation_text_widget.configure(state="disabled")

		options_section = CollapsibleSection(controls, "Options", expanded=True)
		options_section.container.grid(row=5, column=0, sticky="ew", pady=(0, 10))
		opts_box = ttk.LabelFrame(options_section.content, text="", padding=10)
		opts_box.grid(row=0, column=0, sticky="ew")
		visibility_box = ttk.LabelFrame(opts_box, text="Visibility", padding=8)
		visibility_box.grid(sticky="ew", pady=(0, 6))
		step_check = ttk.Checkbutton(visibility_box, text="Show step response", variable=self.show_step_response)
		step_check.grid(sticky="w")
		impulse_check = ttk.Checkbutton(visibility_box, text="Show impulse response", variable=self.show_impulse_response)
		impulse_check.grid(sticky="w")
		ramp_check = ttk.Checkbutton(visibility_box, text="Show ramp response", variable=self.show_ramp_response)
		ramp_check.grid(sticky="w")
		ToolTip(step_check, "Toggle the step-response line on or off.")
		ToolTip(impulse_check, "Toggle the impulse-response line on or off.")
		ToolTip(ramp_check, "Toggle the ramp-response line on or off.")
		mirror_check = ttk.Checkbutton(opts_box, text="Mirror conjugate pairs", variable=self.mirror_conjugates)
		mirror_check.grid(sticky="w")
		snap_check = ttk.Checkbutton(opts_box, text="Snap to grid", variable=self.snap_to_grid)
		snap_check.grid(sticky="w", pady=(4, 0))
		self.snap_label = ttk.Label(opts_box, textvariable=self.snap_info_text)
		self.snap_label.grid(sticky="w", pady=(4, 0))
		bode_mag_check = ttk.Checkbutton(opts_box, text="Show Bode magnitude", variable=self.show_bode_magnitude)
		bode_mag_check.grid(sticky="w", pady=(4, 0))
		bode_phase_check = ttk.Checkbutton(opts_box, text="Show Bode phase", variable=self.show_bode_phase)
		bode_phase_check.grid(sticky="w", pady=(4, 0))
		ttk.Label(opts_box, text="Time horizon").grid(sticky="w", pady=(8, 2))
		time_spin = ttk.Spinbox(opts_box, from_=1.0, to=40.0, increment=0.5, textvariable=self.sim_time, width=10, command=self._refresh_from_state)
		time_spin.grid(
			sticky="w"
		)
		reset_button = ttk.Button(opts_box, text="Reset defaults", command=self._reset_defaults)
		reset_button.grid(sticky="ew", pady=(8, 0))

		status_section = CollapsibleSection(controls, "Status", expanded=True)
		status_section.container.grid(row=6, column=0, sticky="ew")
		status_box = ttk.LabelFrame(status_section.content, text="", padding=10)
		status_box.grid(row=0, column=0, sticky="ew")
		ttk.Label(status_box, textvariable=self.status_text, wraplength=330, justify="left").grid(sticky="w")

		self.figure = Figure(figsize=(12, 8), dpi=100)
		self.ax_pz = self.figure.add_subplot(2, 2, 1)
		self.ax_time = self.figure.add_subplot(2, 2, 2)
		self.ax_mag = self.figure.add_subplot(2, 2, 3)
		self.ax_phase = self.figure.add_subplot(2, 2, 4)
		self.figure.tight_layout(pad=2.0)

		self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
		self.canvas_widget = self.canvas.get_tk_widget()
		self.canvas_widget.grid(row=0, column=0, sticky="nsew")
		toolbar_frame = ttk.Frame(plot_frame)
		toolbar_frame.grid(row=1, column=0, sticky="ew")
		self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
		self.toolbar.update()

		ToolTip(mode_real, "Constrain roots so the transfer function remains real-valued.")
		ToolTip(mode_complex, "Allow free complex coefficients and unconstrained root movement.")
		ToolTip(control_radio, "Prioritize time-domain response plots for control-system analysis.")
		ToolTip(signal_radio, "Prioritize Bode plots for signal-processing / filter analysis.")
		ToolTip(input_coeff, "Load a system from numerator and denominator coefficients.")
		ToolTip(input_equation, "Load a system from a transfer-function equation string.")
		ToolTip(load_button, "Build the model from the chosen input method.")
		ToolTip(self.equation_text_widget, "Shows the transfer function rebuilt from the current pole and zero locations.")
		ToolTip(mirror_check, "Keep complex poles and zeros in conjugate pairs.")
		ToolTip(visibility_box, "Choose which time-response signals are drawn.")
		ToolTip(snap_check, "Snap movement to the selected grid spacing.")
		ToolTip(bode_mag_check, "Show or hide the Bode magnitude subplot.")
		ToolTip(bode_phase_check, "Show or hide the Bode phase subplot.")
		ToolTip(time_spin, "Set the total simulation time used for time-response plots.")
		ToolTip(reset_button, "Restore the default example system and UI settings.")

		self._populate_pz_table()
		self._set_input_mode_visibility()
		self._update_sidebar_width()

		self.root.bind("<Return>", lambda _event: self._apply_exact_values())

	def _connect_events(self) -> None:
		self.canvas.mpl_connect("button_press_event", self._on_press)
		self.canvas.mpl_connect("motion_notify_event", self._on_motion)
		self.canvas.mpl_connect("button_release_event", self._on_release)

	def _reset_defaults(self) -> None:
		self.analysis_mode_text.set("Control systems")
		self.model_mode_text.set("Real coefficients")
		self.input_mode.set("Coefficients")
		self.num_text.set("1 1")
		self.den_text.set("1 1.4 1")
		self.equation_text.set("H(s) = (s + 1) / (s^2 + 1.4*s + 1)")
		self.variable_text.set("s")
		self.show_step_response.set(True)
		self.show_impulse_response.set(True)
		self.show_ramp_response.set(True)
		self.show_bode_magnitude.set(True)
		self.show_bode_phase.set(True)
		self.mirror_conjugates.set(True)
		self.snap_to_grid.set(True)
		self.grid_step.set(0.01)
		self.sim_time.set(10.0)
		self.selected_kind = None
		self.selected_index = None
		self._set_input_mode_visibility()
		self._load_system()

	def _on_mode_changed(self) -> None:
		if self.model_mode_text.get() == "Complex coefficients":
			self.snap_to_grid.set(False)
			self.mirror_conjugates.set(False)
			self.snap_info_text.set("Complex mode active.")
		else:
			self.snap_to_grid.set(True)
			self.mirror_conjugates.set(True)
			self.snap_info_text.set("Grid snap active.")
		self._refresh_from_state()

	def _on_analysis_mode_changed(self) -> None:
		self._refresh_from_state()

	def _on_input_mode_changed(self) -> None:
		self._set_input_mode_visibility()
		self._update_sidebar_width()
		if self.input_mode.get() == "Pole-zero table" and self.pz_tree.get_children():
			self._select_first_pz_row()

	def _set_input_mode_visibility(self) -> None:
		mode = self.input_mode.get()
		for frame in (self.coeff_fields_frame, self.equation_fields_frame, self.pz_table_frame):
			frame.grid_remove()
		if mode == "Coefficients":
			self.coeff_fields_frame.grid()
		elif mode == "Equation":
			self.equation_fields_frame.grid()
		else:
			self.pz_table_frame.grid()
			self.root.after_idle(self._resize_pz_table_columns)

	def _update_sidebar_width(self) -> None:
		if self.input_mode.get() == "Pole-zero table":
			self.root.columnconfigure(0, weight=0, minsize=560)
		else:
			self.root.columnconfigure(0, weight=0, minsize=380)
		self.root.columnconfigure(1, weight=1)

	def _on_pz_table_configure(self, _event=None) -> None:
		self._resize_pz_table_columns()

	def _resize_pz_table_columns(self) -> None:
		if not hasattr(self, "pz_tree"):
			return
		table_width = self.pz_tree.winfo_width()
		if table_width <= 20:
			return
		usable = max(table_width - 18, 180)
		type_width = max(90, int(usable * 0.28))
		value_width = max(110, int((usable - type_width) / 2))
		self.pz_tree.column("Type", width=type_width)
		self.pz_tree.column("Real", width=value_width)
		self.pz_tree.column("Imag", width=value_width)

	def _populate_pz_table(self) -> None:
		if not hasattr(self, "pz_tree"):
			return
		for item in self.pz_tree.get_children():
			self.pz_tree.delete(item)
		for root in self.poles:
			self.pz_tree.insert("", "end", values=("pole", f"{root.real:.6f}", f"{root.imag:.6f}"))
		for root in self.zeros:
			self.pz_tree.insert("", "end", values=("zero", f"{root.real:.6f}", f"{root.imag:.6f}"))
		self._select_first_pz_row()

	def _select_first_pz_row(self) -> None:
		children = self.pz_tree.get_children()
		if not children:
			self.pz_type_text.set("pole")
			self.pz_real_text.set("0.0")
			self.pz_imag_text.set("0.0")
			return
		self.pz_tree.selection_set(children[0])
		self.pz_tree.focus(children[0])
		self._on_pz_tree_select(None)

	def _add_pz_row(self, row_type: str) -> None:
		if row_type not in {"pole", "zero"}:
			row_type = "pole"
		item_id = self.pz_tree.insert("", "end", values=(row_type, "0.0", "0.0"))
		self.pz_tree.selection_set(item_id)
		self.pz_tree.focus(item_id)
		self._on_pz_tree_select(None)

	def _remove_selected_pz_row(self) -> None:
		selection = self.pz_tree.selection()
		if not selection:
			return
		self.pz_tree.delete(selection[0])
		self._select_first_pz_row()

	def _on_pz_tree_select(self, _event) -> None:
		selection = self.pz_tree.selection()
		if not selection:
			return
		values = self.pz_tree.item(selection[0], "values")
		if len(values) != 3:
			return
		self.pz_type_text.set(values[0])
		self.pz_real_text.set(values[1])
		self.pz_imag_text.set(values[2])

	def _update_selected_pz_row(self) -> None:
		selection = self.pz_tree.selection()
		if not selection:
			messagebox.showinfo("Pole-zero table", "Select a row before updating it.")
			return
		row_type = self.pz_type_text.get().strip().lower()
		if row_type not in {"pole", "zero"}:
			messagebox.showerror("Pole-zero table", "Type must be pole or zero.")
			return
		try:
			re_val = float(self.pz_real_text.get())
			im_val = float(self.pz_imag_text.get())
		except ValueError as exc:
			messagebox.showerror("Pole-zero table", f"Invalid coordinates: {exc}")
			return
		selected_item = selection[0]
		self.pz_tree.item(selected_item, values=(row_type, f"{re_val:.6f}", f"{im_val:.6f}"))
		if not self._allow_complex() and self.mirror_conjugates.get() and abs(im_val) > 1e-10:
			partner_item = self._find_conjugate_tree_partner(selected_item, row_type, re_val, im_val)
			if partner_item is None:
				partner_item = self.pz_tree.insert("", "end", values=(row_type, f"{re_val:.6f}", f"{-im_val:.6f}"))
			else:
				self.pz_tree.item(partner_item, values=(row_type, f"{re_val:.6f}", f"{-im_val:.6f}"))

	def _find_conjugate_tree_partner(self, selected_item: str, row_type: str, re_val: float, im_val: float) -> str | None:
		for item_id in self.pz_tree.get_children():
			if item_id == selected_item:
				continue
			values = self.pz_tree.item(item_id, "values")
			if len(values) != 3:
				continue
			item_type, item_re, item_im = values
			if item_type != row_type:
				continue
			try:
				row_re = float(item_re)
				row_im = float(item_im)
			except ValueError:
				continue
			if abs(row_re - re_val) < 1e-8 and abs(row_im + im_val) < 1e-8:
				return item_id
		return None

	def _roots_from_pz_table(self) -> tuple[np.ndarray, np.ndarray]:
		poles: list[complex] = []
		zeros: list[complex] = []
		for item_id in self.pz_tree.get_children():
			values = self.pz_tree.item(item_id, "values")
			if len(values) != 3:
				continue
			row_type, re_text, im_text = values
			try:
				root = complex(float(re_text), float(im_text))
			except ValueError as exc:
				raise ValueError(f"Invalid pole-zero row: {values}") from exc
			if row_type == "pole":
				poles.append(root)
			elif row_type == "zero":
				zeros.append(root)
			else:
				raise ValueError(f"Row type must be pole or zero, not {row_type!r}.")
		poles_array = np.array(poles, dtype=complex)
		zeros_array = np.array(zeros, dtype=complex)
		if not self._allow_complex():
			poles_array = self._sync_real_constraints_for_roots(poles_array)
			zeros_array = self._sync_real_constraints_for_roots(zeros_array)
		return poles_array, zeros_array

	def _sync_real_constraints_for_roots(self, roots: np.ndarray) -> np.ndarray:
		if self._allow_complex() or not len(roots):
			return roots
		adjusted = roots.astype(complex).copy()
		if self.mirror_conjugates.get():
			for index, root in enumerate(adjusted):
				if abs(root.imag) < 1e-10:
					adjusted[index] = complex(root.real, 0.0)
					continue
				partner = self._find_conjugate_partner(adjusted, index, root.real, root.imag)
				if partner is None:
					raise ValueError("Real coefficient mode requires conjugate pole/zero pairs.")
				adjusted[partner] = complex(root.real, -root.imag)
			return adjusted
		adjusted.imag = 0.0
		return adjusted

	def _analysis_mode(self) -> str:
		return self.analysis_mode_text.get()

	def _pointer_within_widget(self, widget: tk.Widget, x: int, y: int) -> bool:
		left = widget.winfo_rootx()
		top = widget.winfo_rooty()
		right = left + widget.winfo_width()
		bottom = top + widget.winfo_height()
		return left <= x <= right and top <= y <= bottom

	def _on_global_mousewheel(self, event) -> None:
		canvas = getattr(self, "sidebar_canvas", None)
		sidebar_container = getattr(self, "sidebar_container", None)
		if canvas is None or sidebar_container is None:
			return
		x = self.root.winfo_pointerx()
		y = self.root.winfo_pointery()
		if not self._pointer_within_widget(sidebar_container, x, y):
			return
		if getattr(event, "num", None) == 4:
			step = -1
		elif getattr(event, "num", None) == 5:
			step = 1
		else:
			step = -1 if getattr(event, "delta", 0) > 0 else 1
		canvas.yview_scroll(step, "units")

	def _allow_complex(self) -> bool:
		return self.model_mode_text.get() == "Complex coefficients"

	def _parse_coefficients_text(self, raw: str) -> np.ndarray:
		if self._allow_complex():
			tokens = raw.replace(",", " ").split()
			if not tokens:
				raise ValueError("Coefficient field is empty.")
			try:
				values = np.array([complex(token.replace("I", "j").replace("i", "j")) for token in tokens], dtype=complex)
			except ValueError as exc:
				raise ValueError("Coefficients must be valid complex numbers.") from exc
			if np.allclose(values, 0.0):
				raise ValueError("All coefficients cannot be zero.")
			return values
		return parse_coefficients(raw)

	def _build_model(self) -> LTIModel:
		allow_complex = self._allow_complex()
		if self.input_mode.get() == "Coefficients":
			num = self._parse_coefficients_text(self.num_text.get())
			den = self._parse_coefficients_text(self.den_text.get())
		elif self.input_mode.get() == "Equation":
			num, den = parse_transfer_function_equation_with_mode(
				self.equation_text.get(), variable=self.variable_text.get(), allow_complex=allow_complex
			)
		else:
			poles, zeros = self._roots_from_pz_table()
			num = coefficients_from_roots(zeros, scale=self.gain, allow_complex=allow_complex)
			den = coefficients_from_roots(poles, scale=1.0, allow_complex=allow_complex)
		return build_lti_model(num, den, allow_complex=allow_complex)

	def _load_system(self) -> None:
		try:
			self.model = self._build_model()
		except Exception as exc:
			messagebox.showerror("Invalid system", str(exc))
			return

		self.poles = np.array(self.model.poles, dtype=complex)
		self.zeros = np.array(self.model.zeros, dtype=complex)
		self.gain = float(np.real_if_close(self.model.numerator[0]).real)
		self.status_text.set(stability_summary(self.model))
		self._clear_selection()
		if self.input_mode.get() != "Pole-zero table":
			self._populate_pz_table()
		self._refresh_from_state()

	def _clear_selection(self) -> None:
		self.selected_kind = None
		self.selected_index = None
		self.selected_pair_index = None
		self.selected_kind_text.set("None")
		self.selected_index_text.set("-")
		self.selected_re_text.set("0.0")
		self.selected_im_text.set("0.0")

	def _selected_root(self) -> complex | None:
		if self.selected_kind is None or self.selected_index is None:
			return None
		roots = self.poles if self.selected_kind == "pole" else self.zeros
		if self.selected_index < 0 or self.selected_index >= len(roots):
			return None
		return roots[self.selected_index]

	def _current_root_array(self) -> np.ndarray:
		return self.poles if self.selected_kind == "pole" else self.zeros

	def _set_selected_root(self, re_val: float, im_val: float) -> None:
		if self.selected_kind is None or self.selected_index is None:
			return
		if self.snap_to_grid.get():
			re_val = _snap_value(re_val, float(self.grid_step.get()))
			im_val = _snap_value(im_val, float(self.grid_step.get()))
			if abs(im_val) < float(self.grid_step.get()) * 0.5:
				im_val = 0.0

		roots = self._current_root_array()
		idx = self.selected_index
		if idx < 0 or idx >= len(roots):
			return

		if not self._allow_complex():
			if abs(im_val) < 1e-10:
				roots[idx] = complex(re_val, 0.0)
			else:
				if self.mirror_conjugates.get():
					roots[idx] = complex(re_val, im_val)
					partner = self.selected_pair_index
					if partner is None or partner >= len(roots) or partner == idx:
						partner = self._find_conjugate_partner(roots, idx, re_val, im_val)
					if partner is None:
						roots = np.append(roots, complex(re_val, -im_val))
						partner = len(roots) - 1
					else:
						roots[partner] = complex(re_val, -im_val)
					self.selected_pair_index = partner
					if self.selected_kind == "pole":
						self.poles = roots
					else:
						self.zeros = roots
				else:
					roots[idx] = complex(re_val, 0.0)
		else:
			roots[idx] = complex(re_val, im_val)

		self._sync_real_constraints()
		self._refresh_model_from_roots()

	def _find_conjugate_partner(self, roots: np.ndarray, idx: int, re_val: float, im_val: float) -> int | None:
		for i, root in enumerate(roots):
			if i == idx:
				continue
			if abs(root.real - re_val) < 1e-8 and abs(root.imag + im_val) < 1e-8:
				return i
		return None

	def _sync_real_constraints(self) -> None:
		if self._allow_complex():
			return
		if self.mirror_conjugates.get():
			self.poles = self._enforce_conjugates(self.poles)
			self.zeros = self._enforce_conjugates(self.zeros)
		else:
			self.poles = self._snap_complex_roots_to_real_axis(self.poles)
			self.zeros = self._snap_complex_roots_to_real_axis(self.zeros)

	def _enforce_conjugates(self, roots: np.ndarray) -> np.ndarray:
		roots = np.array(roots, dtype=complex)
		result: list[complex] = []
		used = [False] * len(roots)
		for i, root in enumerate(roots):
			if used[i]:
				continue
			if abs(root.imag) < 1e-10:
				result.append(complex(root.real, 0.0))
				used[i] = True
				continue
			partner = None
			for j in range(i + 1, len(roots)):
				if used[j]:
					continue
				if abs(roots[j].real - root.real) < 1e-8 and abs(roots[j].imag + root.imag) < 1e-8:
					partner = j
					break
			if partner is None:
				result.append(complex(root.real, abs(root.imag)))
				result.append(complex(root.real, -abs(root.imag)))
			else:
				result.append(complex(0.5 * (root.real + roots[partner].real), abs(root.imag)))
				result.append(complex(0.5 * (root.real + roots[partner].real), -abs(root.imag)))
				used[partner] = True
			used[i] = True
		return np.array(result, dtype=complex)

	def _snap_complex_roots_to_real_axis(self, roots: np.ndarray) -> np.ndarray:
		roots = np.array(roots, dtype=complex)
		for i, root in enumerate(roots):
			if abs(root.imag) > 1e-10:
				roots[i] = complex(root.real, 0.0)
		return roots

	def _refresh_model_from_roots(self) -> None:
		allow_complex = self._allow_complex()
		if not allow_complex:
			self.poles = self._snap_complex_roots_to_real_axis(self.poles) if not self.mirror_conjugates.get() else self._enforce_conjugates(self.poles)
			self.zeros = self._snap_complex_roots_to_real_axis(self.zeros) if not self.mirror_conjugates.get() else self._enforce_conjugates(self.zeros)

		numerator = coefficients_from_roots(self.zeros, scale=self.gain, allow_complex=allow_complex)
		denominator = coefficients_from_roots(self.poles, scale=1.0, allow_complex=allow_complex)
		self.model = build_lti_model(numerator, denominator, allow_complex=allow_complex)
		self.status_text.set(stability_summary(self.model))
		if self.input_mode.get() == "Pole-zero table":
			self._populate_pz_table()
		self._refresh_from_state()

	def _apply_exact_values(self) -> None:
		root = self._selected_root()
		if root is None:
			messagebox.showinfo("Selection required", "Click a pole or zero first.")
			return
		try:
			re_val = float(self.selected_re_text.get())
			im_val = float(self.selected_im_text.get())
		except ValueError:
			messagebox.showerror("Invalid values", "Real and Imag fields must be numeric.")
			return
		self._set_selected_root(re_val, im_val)

	def _nudge_selected(self, dx: float, dy: float) -> None:
		root = self._selected_root()
		if root is None:
			return
		step = float(self.grid_step.get())
		self._set_selected_root(root.real + dx * step, root.imag + dy * step)

	def _nearest_root(self, x: float, y: float) -> tuple[str, int] | None:
		candidates: list[tuple[float, str, int]] = []
		for i, root in enumerate(self.poles):
			candidates.append((abs(complex(x, y) - root), "pole", i))
		for i, root in enumerate(self.zeros):
			candidates.append((abs(complex(x, y) - root), "zero", i))
		if not candidates:
			return None
		candidates.sort(key=lambda item: item[0])
		dist, kind, idx = candidates[0]
		if dist <= self.drag_threshold:
			return kind, idx
		return None

	def _on_press(self, event) -> None:
		if event.inaxes != self.ax_pz or event.xdata is None or event.ydata is None:
			return
		nearest = self._nearest_root(event.xdata, event.ydata)
		if nearest is None:
			self._clear_selection()
			self._refresh_from_state()
			return
		self.selected_kind, self.selected_index = nearest
		self.selected_pair_index = None
		if not self._allow_complex() and self.mirror_conjugates.get():
			root = self._selected_root()
			if root is not None and abs(root.imag) > 1e-10:
				partner = self._find_conjugate_partner(self._current_root_array(), self.selected_index, root.real, root.imag)
				self.selected_pair_index = partner
		self._update_selection_entries()
		self.dragging = True

	def _on_motion(self, event) -> None:
		if not self.dragging or event.inaxes != self.ax_pz or event.xdata is None or event.ydata is None:
			return
		self._set_selected_root(event.xdata, event.ydata)

	def _on_release(self, _event) -> None:
		self.dragging = False

	def _update_selection_entries(self) -> None:
		root = self._selected_root()
		if root is None:
			self._clear_selection()
			return
		self.selected_kind_text.set(self.selected_kind.capitalize() if self.selected_kind else "None")
		self.selected_index_text.set(str(self.selected_index))
		self.selected_re_text.set(f"{root.real:.6f}")
		self.selected_im_text.set(f"{root.imag:.6f}")
		if self.selected_pair_index is not None:
			self.status_text.set(f"Dragging with conjugate partner at index {self.selected_pair_index}.")

	def _format_root_factor(self, root: complex, variable: str) -> str:
		if abs(root.imag) < 1e-10:
			return f"({variable} - ({root.real:.4f}))"
		sign = "+" if root.imag >= 0 else "-"
		return f"({variable} - ({root.real:.4f} {sign} {abs(root.imag):.4f}j))"

	def _equation_from_roots(self) -> str:
		variable = self.variable_text.get() or "s"
		numerator_terms = [self._format_root_factor(root, variable) for root in self.zeros]
		denominator_terms = [self._format_root_factor(root, variable) for root in self.poles]
		numerator = " * ".join(numerator_terms) if numerator_terms else "1"
		denominator = " * ".join(denominator_terms) if denominator_terms else "1"
		return f"H({variable}) = {self.gain:.4f} * ({numerator}) / ({denominator})"

	def _update_equation_output(self) -> None:
		equation = self._equation_from_roots()
		self.equation_display_text.set(equation)
		self.equation_text_widget.configure(state="normal")
		self.equation_text_widget.delete("1.0", tk.END)
		self.equation_text_widget.insert(tk.END, equation)
		self.equation_text_widget.configure(state="disabled")

	def _response_time_series(self, signal_type: str) -> tuple[np.ndarray, np.ndarray]:
		t_final = float(self.sim_time.get())
		t = np.linspace(0.0, t_final, 1600)
		den = np.array(self.model.denominator, dtype=complex)
		num = np.array(self.model.numerator, dtype=complex)
		if signal_type == "impulse":
			input_num = num
			input_den = den
		elif signal_type == "step":
			input_num = num
			input_den = np.polymul(den, np.array([1.0, 0.0], dtype=complex))
		elif signal_type == "ramp":
			input_num = num
			input_den = np.polymul(den, np.array([1.0, 0.0, 0.0], dtype=complex))
		else:
			input_num = num
			input_den = den
		residues, poles, _ = signal.residue(input_num, input_den)
		y = np.zeros_like(t, dtype=complex)
		for residue_value, pole in zip(residues, poles):
			y += residue_value * np.exp(pole * t)
		return t, np.real_if_close(y)

	def _frequency_response_series(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		omega = np.logspace(-2, 3, 900)
		jw = 1j * omega
		h = np.polyval(self.model.numerator, jw) / np.polyval(self.model.denominator, jw)
		magnitude_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-15))
		phase_deg = np.degrees(np.unwrap(np.angle(h)))
		return omega, magnitude_db, phase_deg

	def _axis_bounds(self) -> tuple[float, float]:
		all_points = np.concatenate([self.poles, self.zeros]) if (len(self.poles) or len(self.zeros)) else np.array([0j])
		max_abs = max(float(np.max(np.abs(np.real(all_points)))), float(np.max(np.abs(np.imag(all_points)))), 2.0)
		bound = max_abs * 1.6
		return -bound, bound

	def _refresh_from_state(self) -> None:
		self._update_selection_entries()
		self._update_equation_output()
		self._update_sidebar_width()
		self._draw_plots()

	def _draw_plots(self) -> None:
		self.figure.clear()
		if self._analysis_mode() == "Control systems":
			self.ax_pz = self.figure.add_subplot(1, 2, 1)
			self.ax_time = self.figure.add_subplot(1, 2, 2)
			self.ax_mag = None
			self.ax_phase = None
		else:
			grid = self.figure.add_gridspec(2, 2, width_ratios=[1.2, 1.0])
			self.ax_pz = self.figure.add_subplot(grid[:, 0])
			self.ax_time = None
			self.ax_mag = self.figure.add_subplot(grid[0, 1])
			self.ax_phase = self.figure.add_subplot(grid[1, 1])

		self._draw_pole_zero_plot()
		self._draw_time_plot()
		self._draw_frequency_plots()
		self.figure.tight_layout(pad=2.0)
		self.canvas.draw_idle()

	def _draw_pole_zero_plot(self) -> None:
		self.ax_pz.axhline(0.0, color="#666666", linestyle="--", linewidth=1)
		self.ax_pz.axvline(0.0, color="#666666", linestyle="--", linewidth=1)
		if len(self.poles):
			self.ax_pz.scatter(np.real(self.poles), np.imag(self.poles), marker="x", s=90, c="#d62828", label="Poles")
		if len(self.zeros):
			self.ax_pz.scatter(
				np.real(self.zeros), np.imag(self.zeros), marker="o", s=85, facecolors="none", edgecolors="#1d3557", label="Zeros"
			)
		if self.selected_kind is not None and self.selected_index is not None:
			root = self._selected_root()
			if root is not None:
				self.ax_pz.scatter([root.real], [root.imag], s=180, facecolors="none", edgecolors="#2a9d8f", linewidths=2)
				self.ax_pz.annotate(
					f"{self.selected_kind} {self.selected_index}\n({root.real:.4f}, {root.imag:.4f})",
					xy=(root.real, root.imag),
					xytext=(8, 8),
					textcoords="offset points",
					fontsize=9,
					color="#2a9d8f",
				)
		self.ax_pz.set_title("Pole-Zero Map")
		self.ax_pz.set_xlabel("Real axis")
		self.ax_pz.set_ylabel("Imag axis")
		self.ax_pz.set_aspect("equal", adjustable="box")
		low, high = self._axis_bounds()
		self.ax_pz.set_xlim(low, high)
		self.ax_pz.set_ylim(low, high)
		self.ax_pz.grid(True, alpha=0.25)
		self.ax_pz.legend(loc="upper right")

	def _draw_time_plot(self) -> None:
		if self.ax_time is None:
			return

		for signal_type, color in (("step", "#264653"), ("impulse", "#e76f51"), ("ramp", "#2a9d8f")):
			visible = (
				(signal_type == "step" and self.show_step_response.get())
				or (signal_type == "impulse" and self.show_impulse_response.get())
				or (signal_type == "ramp" and self.show_ramp_response.get())
			)
			if not visible:
				continue
			t, y = self._response_time_series(signal_type)
			y_plot = np.real_if_close(y)
			self.ax_time.plot(t, np.real(y_plot), color=color, linewidth=1.8, label=signal_type.capitalize())
		self.ax_time.set_title("Time Response")
		self.ax_time.set_xlabel("Time (s)")
		self.ax_time.set_ylabel("Output")
		self.ax_time.grid(True, alpha=0.25)
		if self.ax_time.lines:
			self.ax_time.legend(loc="best")
		else:
			self.ax_time.text(0.5, 0.5, "No signals enabled", ha="center", va="center", transform=self.ax_time.transAxes)
			self.ax_time.set_axis_off()

	def _draw_frequency_plots(self) -> None:
		if self.ax_mag is None or self.ax_phase is None:
			return

		omega, magnitude_db, phase_deg = self._frequency_response_series()
		if self.show_bode_magnitude.get():
			self.ax_mag.plot(omega, magnitude_db, color="#1d3557", linewidth=1.8)
			self.ax_mag.set_xscale("log")
			self.ax_mag.set_title("Bode Magnitude")
			self.ax_mag.set_xlabel("Angular frequency (rad/s)")
			self.ax_mag.set_ylabel("Magnitude (dB)")
			self.ax_mag.grid(True, alpha=0.25)
		else:
			self.ax_mag.set_title("Bode Magnitude")
			self.ax_mag.text(0.5, 0.5, "Magnitude hidden", ha="center", va="center", transform=self.ax_mag.transAxes)
			self.ax_mag.set_axis_off()

		if self.show_bode_phase.get():
			self.ax_phase.plot(omega, phase_deg, color="#e63946", linewidth=1.8)
			self.ax_phase.set_xscale("log")
			self.ax_phase.set_title("Bode Phase")
			self.ax_phase.set_xlabel("Angular frequency (rad/s)")
			self.ax_phase.set_ylabel("Phase (deg)")
			self.ax_phase.grid(True, alpha=0.25)
		else:
			self.ax_phase.set_title("Bode Phase")
			self.ax_phase.text(0.5, 0.5, "Phase hidden", ha="center", va="center", transform=self.ax_phase.transAxes)
			self.ax_phase.set_axis_off()

	def run(self) -> None:
		self.root.mainloop()


if __name__ == "__main__":
	PoleZeroDesktopApp().run()
