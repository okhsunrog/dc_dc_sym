# File: gui.py

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QHBoxLayout,
                               QLabel, QPushButton, QSizePolicy, QVBoxLayout,
                               QWidget)

from controller import SlewRatePI  # Import from controller.py
from spice_simulation import run_boost_sim  # Import from spice_simulation.py

# Constants
DEFAULT_KP_CONTROLLER = 0.02
DEFAULT_KI_CONTROLLER = 20.0
DEFAULT_MAX_DUTY_CONTROLLER = 0.90


class BoostSimWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Boost Converter Simulator with S-Curve Target Ramping")
        self.main_layout = QVBoxLayout(self)

        # --- Parameter Controls ---
        controls_row1 = QHBoxLayout()
        self.vin_spin = self._add_param(controls_row1, "Vin", 5, 0.1, 50, "V")
        self.l_spin = self._add_param(controls_row1, "L", 10, 1, 1000, "uH")
        self.c_spin = self._add_param(controls_row1, "C", 100, 1, 1000, "uF")
        self.r_spin = self._add_param(controls_row1, "R", 20, 1, 200, "Ω")

        controls_row2 = QHBoxLayout()
        self.freq_spin = self._add_param(controls_row2, "F", 200, 1, 1000, "kHz")
        self.duty_spin = self._add_param(controls_row2, "Initial Duty", 1, 1, 90, "%")
        self.vref_spin = self._add_param(
            controls_row2, "Vref (Final)", 12, 0.1, 100, "V"
        )
        self.cycles_spin = self._add_param(
            controls_row2, "PWM Cycles", 500, 50, 5000, ""
        )

        controls_row3 = QHBoxLayout()
        self.kp_spin = self._add_param(
            controls_row3, "Kp", DEFAULT_KP_CONTROLLER, 0.000, 0.5, ""
        )
        self.kp_spin.setDecimals(4)
        self.ki_spin = self._add_param(
            controls_row3, "Ki", DEFAULT_KI_CONTROLLER, 0.0, 200.0, ""
        )
        self.ki_spin.setDecimals(3)
        self.max_duty_spin = self._add_param(
            controls_row3, "Dmax PI (%)", DEFAULT_MAX_DUTY_CONTROLLER * 100, 10, 99, ""
        )
        controls_row3.addStretch()

        controls_row4 = QHBoxLayout()
        self.softstart_cycles_spin = self._add_param(
            controls_row4, "Gain Ramp Cycles", 0, 0, 20000, ""
        )
        self.slew_rate_spin = self._add_param(
            controls_row4, "Vref Slew Rate", 1000000, 10, 1000000, "V/s"
        )
        self.acceleration_spin = self._add_param(
            controls_row4, "Acceleration", 3000000, 10, 50000000, "V/s²"
        )
        self.acceleration_spin.setDecimals(0)
        controls_row4.addStretch()

        self.main_layout.addLayout(controls_row1)
        self.main_layout.addLayout(controls_row2)
        self.main_layout.addLayout(controls_row3)
        self.main_layout.addLayout(controls_row4)

        # --- Simulation Buttons ---
        btn_layout = QHBoxLayout()
        self.sim_btn = QPushButton("Simulate (Reset)")
        self.sim_btn.clicked.connect(self.simulate_reset)
        btn_layout.addWidget(self.sim_btn)
        self.next_btn = QPushButton("Next (Continue)")
        self.next_btn.clicked.connect(self.simulate_next)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addStretch()
        self.main_layout.addLayout(btn_layout)

        # --- Matplotlib Figure and Canvas ---
        self.fig, self.axes = plt.subplots(
            4, 1, figsize=(8, 9), sharex=True
        )  # Vout, IL, Duty, Error
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout.addWidget(self.toolbar)
        self.main_layout.addWidget(self.canvas)

        # --- Controller Initialization ---
        self.controller = SlewRatePI(
            DEFAULT_KP_CONTROLLER,
            DEFAULT_KI_CONTROLLER,
            T_sample=1 / (50e3),  # Placeholder T_sample
            slew_rate_limit=self.slew_rate_spin.value(),
            acceleration=self.acceleration_spin.value(),
            duty_max_limit=self.max_duty_spin.value() / 100.0,
        )

        # --- Simulation State & Data ---
        self.last_Vc0_for_next_sim = 0.0
        self.last_Il0_for_next_sim = 0.0
        self.last_applied_duty_for_next_sim = 0.0

        self.t_all_spice = np.array([])
        self.vout_all = np.array([])
        self.il_all = np.array([])
        self.t_duty_updates = np.array([])
        self.duty_all = np.array([])
        self.target_voltage_all = np.array([])
        self.t_last_plot_point = 0.0

    def _add_param(self, layout, label, value, minv, maxv, suffix):
        lbl = QLabel(f"{label}:")
        box = QDoubleSpinBox()
        box.setRange(minv, maxv)
        box.setValue(value)
        box.setMaximumWidth(130)  # Adjusted width slightly
        box.setSuffix(f" {suffix}")
        if (
            suffix == "%"
            or "cycles" in label.lower()
            or "Dmax" in label
            or "Cycles" in label
        ):
            box.setDecimals(0)
        elif "V/s" in suffix or "V/s²" in suffix:  # Added V/s²
            box.setDecimals(0)
        elif "V" in suffix or "A" in suffix or "Ω" in suffix:
            box.setDecimals(2)
        elif "H" in suffix or "F" in suffix or "kHz" in suffix:
            box.setDecimals(1)
        layout.addWidget(lbl)
        layout.addWidget(box)
        return box

    def get_params(self):
        Vin = self.vin_spin.value()
        L_val = self.l_spin.value() * 1e-6
        C_val = self.c_spin.value() * 1e-6
        Rload = self.r_spin.value()
        freq = self.freq_spin.value() * 1e3
        initial_duty_from_gui = self.duty_spin.value() / 100.0
        Vref_final = self.vref_spin.value()
        Num_PWM_cycles_to_run = int(self.cycles_spin.value())
        T_switching_calc = 1.0 / freq if freq > 0 else 1.0
        step_time_calc = T_switching_calc / 200 if T_switching_calc > 0 else 1e-7
        Kp = self.kp_spin.value()
        Ki = self.ki_spin.value()
        Max_duty_pi = self.max_duty_spin.value() / 100.0
        gain_ramp_cycles = int(self.softstart_cycles_spin.value())
        slew_rate = self.slew_rate_spin.value()
        acceleration = self.acceleration_spin.value()  # Get new acceleration parameter

        return (
            Vin,
            L_val,
            C_val,
            Rload,
            freq,
            initial_duty_from_gui,
            step_time_calc,
            Vref_final,
            Num_PWM_cycles_to_run,
            Kp,
            Ki,
            T_switching_calc,
            Max_duty_pi,
            gain_ramp_cycles,
            slew_rate,
            acceleration,  # Pass acceleration
        )

    def _update_controller_params(self, Kp, Ki, T_switching, slew_rate, acceleration):
        """Helper to update controller instance with current GUI params."""
        self.controller.update_coeffs(Kp, Ki)
        self.controller.T_sample = T_switching
        self.controller.set_slew_parameters(slew_rate, acceleration)  # Use new method

    def _run_simulation_segment(
        self,
        Vin,
        L_val,
        C_val,
        Rload,
        freq,
        initial_duty_for_segment,
        step_time_calc,
        Vref_final,
        Num_PWM_cycles_segment,
        Kp,
        Ki,
        T_switching,
        Max_duty_pi,
        gain_ramp_cycles,
        slew_rate,
        acceleration,  # Add acceleration
        start_Vc0,
        start_Il0,
        start_last_applied_duty,
        use_gain_ramp,
    ):
        # Update controller with potentially changed slew/accel params from GUI FOR THIS SEGMENT
        self._update_controller_params(Kp, Ki, T_switching, slew_rate, acceleration)
        # self.controller.final_Vref = Vref_final # Controller handles this via _start_new_ramp_if_needed

        current_Vc0 = start_Vc0
        current_Il0 = start_Il0
        last_applied_duty = start_last_applied_duty

        Kp_init = 0.05 * Kp
        Ki_init = 0.05 * Ki
        Max_duty_init_pi = 0.05 * Max_duty_pi + 0.01

        for pwm_cycle_num in range(Num_PWM_cycles_segment):
            time_of_current_cycle_start = self.t_last_plot_point
            Vc_feedback = current_Vc0

            # Apply gain scheduling (Kp, Ki, Dmax_PI ramp)
            if use_gain_ramp and pwm_cycle_num < gain_ramp_cycles:
                ramp_factor = pwm_cycle_num / max(gain_ramp_cycles, 1)
                Kp_current = Kp_init + (Kp - Kp_init) * ramp_factor
                Ki_current = Ki_init + (Ki - Ki_init) * ramp_factor
                Max_duty_current_pi = (
                    Max_duty_init_pi + (Max_duty_pi - Max_duty_init_pi) * ramp_factor
                )
            else:
                Kp_current = Kp
                Ki_current = Ki
                Max_duty_current_pi = Max_duty_pi

            self.controller.update_coeffs(Kp_current, Ki_current)
            self.controller.set_duty_limits(0.01, Max_duty_current_pi)

            current_duty_to_apply = 0.0
            target_voltage_for_plot = 0.0

            if self.t_last_plot_point == 0 and pwm_cycle_num == 0:
                current_duty_to_apply = initial_duty_for_segment
                # Controller's current_target was set to Vin by reset(), final_Vref set before this loop
                target_voltage_for_plot = self.controller.current_target
                # Trigger ramp logic for the first step if Vref_final is different from current_target
                _, target_voltage_for_plot = self.controller.calculate_duty(
                    Vref_final, Vc_feedback, Vin, last_applied_duty
                )
                # This is a bit tricky for the very first point. The calculate_duty call above will move it.
                # We should probably record the target *before* calculate_duty for the first point if initial_duty is used
                # Or, ensure controller state (like current_target and final_Vref) is fully set before this loop.
                # For simplicity, let's assume initial_duty_for_segment is just for SPICE, controller starts its ramp.
                # The first recorded target_voltage_for_plot might be slightly off if initial_duty is used and Vref!=Vin.

            else:
                current_duty_to_apply, target_voltage_for_plot = (
                    self.controller.calculate_duty(
                        Vref_final, Vc_feedback, Vin, last_applied_duty
                    )
                )

            # Store data *before* running SPICE for this cycle, as it reflects the controller's state *at the start* of the cycle
            self.duty_all = np.append(self.duty_all, current_duty_to_apply)
            self.t_duty_updates = np.append(
                self.t_duty_updates, time_of_current_cycle_start
            )
            self.target_voltage_all = np.append(
                self.target_voltage_all, target_voltage_for_plot
            )
            last_applied_duty = (
                current_duty_to_apply  # Update for next iteration's anti-windup
            )

            try:
                analysis = run_boost_sim(
                    Vin,
                    L_val,
                    C_val,
                    Rload,
                    freq,
                    current_duty_to_apply,
                    step_time_calc,
                    0,
                    T_switching,
                    current_Il0,
                    current_Vc0,
                )
            except RuntimeError as e:
                print(f"SPICE simulation failed: {e}")
                break

            time_segment_spice = np.array(analysis.time)
            if len(time_segment_spice) < 2:
                print(
                    f"Warning: SPICE sim for cycle {pwm_cycle_num} (t={time_of_current_cycle_start*1e3:.3f}ms) "
                    f"returned too few points ({len(time_segment_spice)}). Stopping segment."
                )
                break

            time_segment_global = (
                time_segment_spice - time_segment_spice[0] + self.t_last_plot_point
            )

            try:
                vout_segment = np.array(analysis.nodes["n2"])
                il_segment = np.array(analysis.branches["vjl_sense"])
            except KeyError as e:
                print(
                    f"Error: SPICE node/branch key not found for cycle {pwm_cycle_num}: {e}. Stopping segment."
                )
                break
            except Exception as e:
                print(
                    f"Error processing SPICE results for cycle {pwm_cycle_num}: {e}. Stopping segment."
                )
                break

            min_len_spice = min(
                len(vout_segment), len(il_segment), len(time_segment_global)
            )
            if min_len_spice <= 1:
                print(
                    f"Warning: SPICE segment too short (len={min_len_spice}) for cycle {pwm_cycle_num}. Stopping."
                )
                break

            time_segment_global = time_segment_global[:min_len_spice]
            vout_segment = vout_segment[:min_len_spice]
            il_segment = il_segment[:min_len_spice]

            self.t_all_spice = np.concatenate((self.t_all_spice, time_segment_global))
            self.vout_all = np.concatenate((self.vout_all, vout_segment))
            self.il_all = np.concatenate((self.il_all, il_segment))

            current_Vc0 = float(vout_segment[-1])
            current_Il0 = float(il_segment[-1])
            self.t_last_plot_point = time_segment_global[-1]

        self.last_Vc0_for_next_sim = current_Vc0
        self.last_Il0_for_next_sim = current_Il0
        self.last_applied_duty_for_next_sim = last_applied_duty

    def simulate_reset(self):
        params = self.get_params()
        (
            Vin,
            L_val,
            C_val,
            Rload,
            freq,
            initial_duty_gui,
            step_time_calc,
            Vref_final,
            Num_PWM_cycles_to_run,
            Kp,
            Ki,
            T_switching,
            Max_duty_pi,
            gain_ramp_cycles,
            slew_rate,
            acceleration,
        ) = params

        self.controller.reset(initial_target_voltage=Vin, initial_last_vout_voltage=Vin)
        self._update_controller_params(Kp, Ki, T_switching, slew_rate, acceleration)
        # self.controller.final_Vref = Vref_final # This is handled by calculate_duty's _start_new_ramp_if_needed

        self.t_all_spice = np.array([])
        self.vout_all = np.array([])
        self.il_all = np.array([])
        self.duty_all = np.array([])
        self.t_duty_updates = np.array([])
        self.target_voltage_all = np.array([])
        self.t_last_plot_point = 0.0

        # For the very first cycle, if initial_duty_gui is used, the first plotted target
        # might be Vin, then it immediately jumps. To make it smoother:
        # Set controller's initial state more explicitly for reset.
        self.controller.current_target = Vin
        self.controller.final_Vref = (
            Vref_final  # Ensure controller knows the goal from the start
        )
        self.controller.is_ramping = False  # Let _start_new_ramp_if_needed trigger

        self._run_simulation_segment(
            Vin,
            L_val,
            C_val,
            Rload,
            freq,
            initial_duty_gui,
            step_time_calc,
            Vref_final,
            Num_PWM_cycles_to_run,
            Kp,
            Ki,
            T_switching,
            Max_duty_pi,
            gain_ramp_cycles,
            slew_rate,
            acceleration,
            start_Vc0=Vin,
            start_Il0=0.0,
            start_last_applied_duty=initial_duty_gui,
            use_gain_ramp=True,
        )

        self.plot_results()

    def simulate_next(self):
        if self.t_last_plot_point == 0 and not self.t_all_spice.size:
            print("Nothing to continue. Please run 'Simulate (Reset)' first.")
            return

        params = self.get_params()
        (
            Vin,
            L_val,
            C_val,
            Rload,
            freq,
            _,
            step_time_calc,
            Vref_final,
            Num_PWM_cycles_to_run,
            Kp,
            Ki,
            T_switching,
            Max_duty_pi,
            gain_ramp_cycles,
            slew_rate,
            acceleration,
        ) = params

        self._update_controller_params(Kp, Ki, T_switching, slew_rate, acceleration)
        # self.controller.final_Vref = Vref_final # Handled by calculate_duty

        self._run_simulation_segment(
            Vin,
            L_val,
            C_val,
            Rload,
            freq,
            self.last_applied_duty_for_next_sim,
            step_time_calc,
            Vref_final,
            Num_PWM_cycles_to_run,
            Kp,
            Ki,
            T_switching,
            Max_duty_pi,
            gain_ramp_cycles,
            slew_rate,
            acceleration,
            start_Vc0=self.last_Vc0_for_next_sim,
            start_Il0=self.last_Il0_for_next_sim,
            start_last_applied_duty=self.last_applied_duty_for_next_sim,
            use_gain_ramp=False,
        )

        self.plot_results()

    def plot_results(self):
        if not self.t_all_spice.size:
            for ax_idx, ax in enumerate(self.axes):
                ax.clear()
                ax.grid(True)
                if ax_idx == len(self.axes) - 1:
                    ax.set_xlabel("Time (ms)")
                ax.text(
                    0.5,
                    0.5,
                    "No simulation data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            self.fig.tight_layout()
            self.canvas.draw()
            return

        Vref_final_val = self.vref_spin.value()
        t_ms_spice = self.t_all_spice * 1e3
        t_ms_duty = self.t_duty_updates * 1e3

        for ax_idx, ax in enumerate(self.axes):
            ax.clear()
            ax.grid(True)
            if ax_idx == len(self.axes) - 1:
                ax.set_xlabel("Time (ms)")

        # Axes 0: Vout, Vref_final, Slew-Limited Target V
        self.axes[0].plot(t_ms_spice, self.vout_all, label="Vout")
        self.axes[0].axhline(
            y=Vref_final_val,
            color="r",
            linestyle="--",
            label=f"Final Vref = {Vref_final_val:.2f}V",
        )
        if self.t_duty_updates.size and self.target_voltage_all.size:
            plot_len = min(len(self.t_duty_updates), len(self.target_voltage_all))
            self.axes[0].plot(
                t_ms_duty[:plot_len],
                self.target_voltage_all[:plot_len],
                "g--",
                label="S-Curve Target V",
            )
        self.axes[0].set_ylabel("Voltage (V)")
        self.axes[0].set_title("Output Voltage and Controller Target")
        self.axes[0].legend(loc="lower right")

        # Axes 1: Inductor Current
        self.axes[1].plot(t_ms_spice, self.il_all)
        self.axes[1].set_ylabel("Inductor Current (A)")
        self.axes[1].set_title("Inductor Current")

        # Axes 2: PWM Duty Cycle
        if self.t_duty_updates.size and self.duty_all.size:
            plot_len = min(len(self.t_duty_updates), len(self.duty_all))
            self.axes[2].plot(
                t_ms_duty[:plot_len],
                self.duty_all[:plot_len] * 100,
                drawstyle="steps-post",
            )
            self.axes[2].set_ylabel("Duty Cycle (%)")
            self.axes[2].set_title("PWM Duty Cycle")
            current_max_duty_from_gui = self.max_duty_spin.value()
            self.axes[2].set_ylim(-5, min(current_max_duty_from_gui + 10, 105))
        else:
            self.axes[2].text(
                0.5,
                0.5,
                "No Duty Data",
                ha="center",
                va="center",
                transform=self.axes[2].transAxes,
            )

        # Axes 3: Control Error
        if (
            self.t_duty_updates.size
            and self.target_voltage_all.size
            and self.vout_all.size
            and self.t_all_spice.size
        ):
            plot_len = min(len(self.t_duty_updates), len(self.target_voltage_all))
            # Interpolate Vout to the times when duty/target_V are updated (start of each cycle)
            vout_at_duty_updates = np.interp(
                self.t_duty_updates[:plot_len],
                self.t_all_spice,
                self.vout_all,
                left=self.vout_all[
                    0
                ],  # Use first actual Vout for points before SPICE data starts
                right=self.vout_all[-1],
            )  # Use last actual Vout for points after SPICE data ends
            control_error = self.target_voltage_all[:plot_len] - vout_at_duty_updates
            self.axes[3].plot(t_ms_duty[:plot_len], control_error)
            self.axes[3].set_ylabel("Control Error (V)")
            self.axes[3].set_title("Control Error (Target V - Vout feedback)")
        else:
            self.axes[3].text(
                0.5,
                0.5,
                "No Error Data",
                ha="center",
                va="center",
                transform=self.axes[3].transAxes,
            )

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = BoostSimWidget()
    w.show()
    sys.exit(app.exec())
