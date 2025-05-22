import sys
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QPushButton,
    QSizePolicy,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from PySpice import Simulator

# Constants
DEFAULT_KP_CONTROLLER = 0.005
DEFAULT_KI_CONTROLLER = 0.05
DEFAULT_MAX_DUTY_CONTROLLER = 0.90


# --- PySpice Simulation Function (largely unchanged) ---
def run_boost_sim(
    Vin_val=5,
    L_val=100e-6,
    C_val=100e-6,
    Rload_val=10,
    freq_val=50e3,
    duty_cycle_val=0.6,
    step_time_val=0.1e-6,
    t_start_val=0,
    t_end_val=2e-3,
    Il0_val=0,
    Vc0_val=0,
):
    circuit = Circuit(
        f"Boost Converter XSPICE (Duty={duty_cycle_val:.2f} Il0={Il0_val:.2e})"
    )

    circuit.V("input", "vin", circuit.gnd, Vin_val @ u_V)
    node_l_sense_out = (
        "n_for_l_sense"  # Node after current sense Vsource, before inductor
    )
    circuit.VoltageSource(
        "Jl_sense", "vin", node_l_sense_out, 0 @ u_V
    )  # For inductor current measurement

    l_spice_float = float(L_val)
    il0_spice_float = float(Il0_val)

    # Using XSPICE inductoric model for initial current
    circuit.raw_spice = f""".model inductor_ic_model inductoric L={l_spice_float:.7e} IC={il0_spice_float:.7e}
A_L1 {node_l_sense_out} n1 inductor_ic_model"""  # A_L1 is the inductor

    circuit.S("1", "n1", circuit.gnd, "gate", circuit.gnd, model="SW")
    circuit.model("SW", "SW", Ron=1 @ u_mOhm, Roff=1 @ u_MOhm, Vt=2.5, Vh=0.1)
    circuit.D("1", "n1", "n2", model="D")
    circuit.model("D", "D", IS=1e-15, N=1)
    circuit.C("1", "n2", circuit.gnd, C_val @ u_F, ic=Vc0_val @ u_V)
    circuit.R("load", "n2", circuit.gnd, Rload_val @ u_Ohm)

    period = 1 / freq_val
    safe_duty_cycle = max(0.001, min(float(duty_cycle_val), 0.999))
    pulse_width = period * safe_duty_cycle
    circuit.PulseVoltageSource(
        "gate_drive",
        "gate",
        circuit.gnd,
        initial_value=0 @ u_V,
        pulsed_value=5 @ u_V,
        pulse_width=pulse_width @ u_s,
        period=period @ u_s,
        delay_time=0 @ u_s,
        rise_time=10 @ u_ns,
        fall_time=10 @ u_ns,
    )

    try:
        actual_simulator_object = Simulator.factory(simulator="ngspice-shared")
    except Exception:
        try:
            actual_simulator_object = Simulator.factory(simulator="ngspice-subprocess")
        except Exception as e_subprocess:
            raise RuntimeError(f"SPICE simulator creation error: {e_subprocess}")

    simulation_instance = actual_simulator_object.simulation(
        circuit, temperature=25, nominal_temperature=25
    )
    analysis = simulation_instance.transient(
        step_time=step_time_val @ u_s,
        end_time=t_end_val @ u_s,
        start_time=t_start_val @ u_s,
        use_initial_condition=True,
    )
    return analysis


# --- Slew Rate PI Controller ---
class SlewRatePI:
    def __init__(
        self,
        Kp,
        Ki,
        T_sample,
        slew_rate_limit=10.0,
        duty_min=0.01,
        duty_max_limit=0.99,
        ff_enabled=True,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.T_sample = T_sample
        self.duty_min_internal = duty_min
        self.duty_max_internal = duty_max_limit  # Overall duty cycle limit from PI
        self.ff_enabled = ff_enabled
        self.integral_error = 0.0
        self.last_calculated_duty = 0.0

        self.slew_rate_limit = slew_rate_limit  # V/s
        self.current_target = 0.0  # Current target voltage after slew rate limiting
        self.last_vout = (
            0.0  # Last measured output voltage (for calculating actual slew, if needed)
        )

    def reset(self, initial_target_voltage=0.0, initial_last_vout_voltage=0.0):
        self.integral_error = 0.0
        self.last_calculated_duty = 0.0
        self.current_target = initial_target_voltage
        self.last_vout = initial_last_vout_voltage

    def update_coeffs(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki

    def set_duty_limits(self, duty_min, duty_max):
        self.duty_min_internal = duty_min
        self.duty_max_internal = duty_max  # This is Dmax_PI from GUI

    def set_slew_rate_limit(self, slew_rate):
        self.slew_rate_limit = slew_rate

    def calculate_duty(self, Vref_final, Vc_feedback, Vin, duty_applied_in_prev_step):
        # Update slew-rate limited internal target (self.current_target)
        max_voltage_change = self.slew_rate_limit * self.T_sample

        if self.current_target < Vref_final:
            self.current_target = min(
                Vref_final, self.current_target + max_voltage_change
            )
        elif self.current_target > Vref_final:
            self.current_target = max(
                Vref_final, self.current_target - max_voltage_change
            )

        # Calculate error based on the internal, slew-limited target
        error = self.current_target - Vc_feedback

        # Actual output voltage slew rate (calculated but not used in this PI version for control action)
        # voltage_slew_actual = (Vc_feedback - self.last_vout) / self.T_sample
        self.last_vout = Vc_feedback  # Update for next cycle's observation

        # Anti-windup logic
        can_integrate = True
        if (duty_applied_in_prev_step >= self.duty_max_internal and error > 0) or (
            duty_applied_in_prev_step <= self.duty_min_internal and error < 0
        ):
            can_integrate = False

        if can_integrate:
            self.integral_error += error * self.T_sample

        pi_correction = self.Kp * error + self.Ki * self.integral_error

        calculated_duty = 0.0
        if self.ff_enabled:
            duty_ff = 0.0
            if (
                self.current_target > Vin and self.current_target > 0
            ):  # Use current_target for FF
                duty_ff = (self.current_target - Vin) / self.current_target
            # Clamp FF component: min_duty <= duty_ff <= min(0.90, Dmax_PI)
            # 0.90 is an arbitrary cap for FF to leave room for PI.
            duty_ff = max(
                self.duty_min_internal, min(duty_ff, min(0.90, self.duty_max_internal))
            )
            calculated_duty = duty_ff + pi_correction
        else:
            calculated_duty = pi_correction

        # Final clamping of total calculated duty
        self.last_calculated_duty = max(
            self.duty_min_internal, min(calculated_duty, self.duty_max_internal)
        )

        return (
            self.last_calculated_duty,
            self.current_target,
        )  # Return duty and the target used


# --- Main Application Widget ---
class BoostSimWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Boost Converter Simulator with Slew Rate & Gain Ramp Control"
        )
        self.main_layout = QVBoxLayout(self)

        # Parameter Controls - Row 1
        controls_row1 = QHBoxLayout()
        self.vin_spin = self._add_param(controls_row1, "Vin", 5, 0.1, 50, "V")
        self.l_spin = self._add_param(controls_row1, "L", 100, 1, 1000, "uH")
        self.c_spin = self._add_param(controls_row1, "C", 100, 1, 1000, "uF")
        self.r_spin = self._add_param(controls_row1, "R", 20, 1, 200, "Ω")

        # Parameter Controls - Row 2
        controls_row2 = QHBoxLayout()
        self.freq_spin = self._add_param(controls_row2, "F", 50, 1, 1000, "kHz")
        self.duty_spin = self._add_param(controls_row2, "Initial Duty", 5, 1, 90, "%")
        self.vref_spin = self._add_param(
            controls_row2, "Vref (Final)", 10, 0.1, 100, "V"
        )
        self.cycles_spin = self._add_param(
            controls_row2, "PWM Cycles", 500, 50, 5000, ""
        )

        # Parameter Controls - Row 3 (PI Controller Gains)
        controls_row3 = QHBoxLayout()
        self.kp_spin = self._add_param(
            controls_row3, "Kp", DEFAULT_KP_CONTROLLER, 0.0000, 0.5, ""
        )
        self.kp_spin.setDecimals(4)
        self.ki_spin = self._add_param(
            controls_row3, "Ki", DEFAULT_KI_CONTROLLER, 0.0, 20.0, ""
        )
        self.ki_spin.setDecimals(3)
        self.max_duty_spin = self._add_param(
            controls_row3, "Dmax PI (%)", DEFAULT_MAX_DUTY_CONTROLLER * 100, 10, 99, ""
        )
        controls_row3.addStretch()

        # Parameter Controls - Row 4 (Soft Start: Gain Ramp & Slew Rate)
        controls_row4 = QHBoxLayout()
        self.softstart_cycles_spin = self._add_param(
            controls_row4, "Gain Ramp Cycles", 50, 0, 20000, ""
        )
        self.slew_rate_spin = self._add_param(
            controls_row4, "Vref Slew Rate", 1000, 10, 100000, "V/s"
        )
        controls_row4.addStretch()

        self.main_layout.addLayout(controls_row1)
        self.main_layout.addLayout(controls_row2)
        self.main_layout.addLayout(controls_row3)
        self.main_layout.addLayout(controls_row4)

        # Simulation Buttons
        btn_layout = QHBoxLayout()
        self.sim_btn = QPushButton("Simulate (Reset)")
        self.sim_btn.clicked.connect(self.simulate_reset)
        btn_layout.addWidget(self.sim_btn)
        self.next_btn = QPushButton("Next (Continue)")
        self.next_btn.clicked.connect(self.simulate_next)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addStretch()
        self.main_layout.addLayout(btn_layout)

        # Matplotlib Figure and Canvas (4 subplots)
        self.fig, self.axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout.addWidget(self.toolbar)
        self.main_layout.addWidget(self.canvas)

        # Initialize SlewRatePI controller
        self.controller = SlewRatePI(
            DEFAULT_KP_CONTROLLER,
            DEFAULT_KI_CONTROLLER,
            T_sample=0,  # T_sample set dynamically
            slew_rate_limit=self.slew_rate_spin.value(),
            duty_max_limit=self.max_duty_spin.value() / 100.0,
        )

        # Simulation state variables
        self.last_Vc0_for_next_sim = 0.0
        self.last_Il0_for_next_sim = 0.0
        self.last_applied_duty_for_next_sim = 0.0

        # Data storage for plots
        self.t_all_spice = np.array([])
        self.vout_all = np.array([])
        self.il_all = np.array([])
        self.t_duty_updates = np.array([])  # Times at which duty/target_V are updated
        self.duty_all = np.array([])
        self.target_voltage_all = np.array([])  # Store slew-rate limited target voltage
        self.t_last_plot_point = 0.0

    def _add_param(self, layout, label, value, minv, maxv, suffix):
        lbl = QLabel(f"{label}:")
        box = QDoubleSpinBox()
        box.setRange(minv, maxv)
        box.setValue(value)
        box.setMaximumWidth(120)
        box.setSuffix(f" {suffix}")
        if (
            suffix == "%"
            or "cycles" in label.lower()
            or "Dmax" in label
            or "Cycles" in label
        ):
            box.setDecimals(0)
        elif "V/s" in suffix:
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
        initial_duty_from_gui = self.duty_spin.value() / 100.0  # For 1st cycle on reset
        Vref_final = self.vref_spin.value()  # The ultimate target Vref
        Num_PWM_cycles_to_run = int(self.cycles_spin.value())
        T_switching_calc = 1 / freq
        step_time_calc = T_switching_calc / 200
        Kp = self.kp_spin.value()
        Ki = self.ki_spin.value()
        Max_duty_pi = (
            self.max_duty_spin.value() / 100.0
        )  # Dmax for PI controller output
        gain_ramp_cycles = int(
            self.softstart_cycles_spin.value()
        )  # For Kp/Ki/Dmax ramp
        slew_rate = self.slew_rate_spin.value()  # V/s for Vref ramp

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
        )

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
        start_Vc0,
        start_Il0,
        start_last_applied_duty,
        use_gain_ramp,  # Boolean flag for Kp/Ki/Dmax ramping
    ):
        self.controller.T_sample = T_switching
        self.controller.set_slew_rate_limit(slew_rate)

        current_Vc0 = start_Vc0
        current_Il0 = start_Il0
        last_applied_duty = start_last_applied_duty

        # Initial gains for gain scheduling ramp
        Kp_init = 0.05 * Kp
        Ki_init = 0.05 * Ki
        Max_duty_init_pi = (
            0.05 * Max_duty_pi + 0.01
        )  # Avoid zero, ensure small initial Dmax for PI

        for pwm_cycle_num in range(Num_PWM_cycles_segment):
            time_of_current_cycle_start = self.t_last_plot_point
            Vc_feedback = current_Vc0  # Vout at start of this cycle

            # Gain Scheduling (Kp, Ki, Dmax_PI ramp)
            if use_gain_ramp and pwm_cycle_num < gain_ramp_cycles:
                ramp_factor = pwm_cycle_num / max(
                    gain_ramp_cycles, 1
                )  # Avoid div by zero
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

            target_voltage_for_plot = 0.0
            # For the very first cycle after a reset (t_last_plot_point is 0):
            if self.t_last_plot_point == 0 and pwm_cycle_num == 0:
                current_duty_to_apply = (
                    initial_duty_for_segment  # Use GUI's initial duty
                )
                # self.controller.current_target was set to Vin by reset()
                target_voltage_for_plot = self.controller.current_target
            else:
                # Controller calculates duty based on its internally ramped current_target.
                # Vref_final is the ultimate goal.
                current_duty_to_apply, target_voltage_for_plot = (
                    self.controller.calculate_duty(
                        Vref_final, Vc_feedback, Vin, last_applied_duty
                    )
                )

            self.duty_all = np.append(self.duty_all, current_duty_to_apply)
            self.t_duty_updates = np.append(
                self.t_duty_updates, time_of_current_cycle_start
            )
            self.target_voltage_all = np.append(
                self.target_voltage_all, target_voltage_for_plot
            )
            last_applied_duty = current_duty_to_apply

            # Run SPICE simulation for this single PWM cycle
            analysis = run_boost_sim(
                Vin,
                L_val,
                C_val,
                Rload,
                freq,
                current_duty_to_apply,
                step_time_calc,
                0,
                T_switching,  # t_start=0, t_end=T_switching
                current_Il0,
                current_Vc0,
            )

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
                # Current through the 0V source 'Jl_sense' is inductor current
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

            # Ensure all segments have the same length for concatenation
            min_len_spice = min(
                len(vout_segment), len(il_segment), len(time_segment_global)
            )
            if len(time_segment_global) != min_len_spice:
                time_segment_global = time_segment_global[:min_len_spice]
            if len(vout_segment) != min_len_spice:
                vout_segment = vout_segment[:min_len_spice]
            if len(il_segment) != min_len_spice:
                il_segment = il_segment[:min_len_spice]

            self.t_all_spice = np.concatenate((self.t_all_spice, time_segment_global))
            self.vout_all = np.concatenate((self.vout_all, vout_segment))
            self.il_all = np.concatenate((self.il_all, il_segment))

            if len(vout_segment) > 0:  # Should always be true if min_len_spice > 0
                current_Vc0 = float(vout_segment[-1])
                current_Il0 = float(il_segment[-1])
                self.t_last_plot_point = time_segment_global[-1]
            else:  # Should be caught by len(time_segment_spice) < 2
                print(
                    f"Warning: Empty Vout segment for cycle {pwm_cycle_num}. Stopping segment."
                )
                break

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
        ) = params

        # Reset controller: initial target and last_vout start at current Vin
        self.controller.reset(initial_target_voltage=Vin, initial_last_vout_voltage=Vin)

        # Data arrays reset
        self.t_all_spice = np.array([])
        self.vout_all = np.array([])
        self.il_all = np.array([])
        self.duty_all = np.array([])
        self.t_duty_updates = np.array([])
        self.target_voltage_all = np.array([])
        self.t_last_plot_point = 0.0  # Reset simulation time

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
            start_Vc0=Vin,
            start_Il0=0.0,  # Initial conditions for SPICE
            start_last_applied_duty=initial_duty_gui,  # For anti-windup in 1st PI calc
            use_gain_ramp=True,  # Enable Kp/Ki/Dmax ramp for reset
        )

        if len(self.t_all_spice) > 0:
            self.plot_results()

    def simulate_next(self):
        if self.t_last_plot_point == 0 and len(self.t_all_spice) == 0:
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
            step_time_calc,  # initial_duty_gui not used
            Vref_final,
            Num_PWM_cycles_to_run,
            Kp,
            Ki,
            T_switching,
            Max_duty_pi,
            gain_ramp_cycles,
            slew_rate,
        ) = params

        # Vin might have changed in GUI. Controller's FF term will use new Vin.
        # Controller's internal target_v continues from where it left off.
        self._run_simulation_segment(
            Vin,
            L_val,
            C_val,
            Rload,
            freq,
            self.last_applied_duty_for_next_sim,  # "Initial" duty for this segment
            step_time_calc,
            Vref_final,
            Num_PWM_cycles_to_run,
            Kp,
            Ki,
            T_switching,
            Max_duty_pi,
            gain_ramp_cycles,
            slew_rate,
            start_Vc0=self.last_Vc0_for_next_sim,  # Continue from last Vc
            start_Il0=self.last_Il0_for_next_sim,  # Continue from last Il
            start_last_applied_duty=self.last_applied_duty_for_next_sim,
            use_gain_ramp=False,  # Gain ramp typically only for initial soft-start
        )

        if len(self.t_all_spice) > 0:
            self.plot_results()

    def plot_results(self):
        if not self.t_all_spice.size:  # Check if empty
            print("No simulation data to plot.")
            # Clear axes if needed, or show a message on plots
            for ax in self.axes:
                ax.clear()
                ax.grid(True)
                ax.text(
                    0.5,
                    0.5,
                    "No simulation data",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
            self.axes[-1].set_xlabel("Time (ms)")  # Keep label on bottom plot
            self.fig.tight_layout()
            self.canvas.draw()
            return

        Vref_final_val = self.vref_spin.value()
        t_ms_spice = self.t_all_spice * 1e3
        t_ms_duty = self.t_duty_updates * 1e3

        for ax_idx, ax in enumerate(self.axes):
            ax.clear()
            ax.grid(True)
            if ax_idx == len(self.axes) - 1:  # X-label only on the bottom-most plot
                ax.set_xlabel("Time (ms)")

        # Axes 0: Output Voltage, Final Vref, and Slew-Limited Target Voltage
        self.axes[0].plot(t_ms_spice, self.vout_all, label="Vout")
        self.axes[0].axhline(
            y=Vref_final_val,
            color="r",
            linestyle="--",
            label=f"Final Vref = {Vref_final_val:.2f}V",
        )
        if self.t_duty_updates.size and self.target_voltage_all.size:
            self.axes[0].plot(
                t_ms_duty, self.target_voltage_all, "g--", label="Slew-Lim Target V"
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
            self.axes[2].plot(t_ms_duty, self.duty_all * 100, drawstyle="steps-post")
            self.axes[2].set_ylabel("Duty Cycle (%)")
            self.axes[2].set_title("PWM Duty Cycle")
            current_max_duty_from_gui = self.max_duty_spin.value()  # Dmax PI from GUI
            self.axes[2].set_ylim(-5, min(current_max_duty_from_gui + 10, 105))
        else:
            self.axes[2].text(
                0.5,
                0.5,
                "No Duty Data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=self.axes[2].transAxes,
            )

        # Axes 3: Control Error (Slew-Limited Target V - Vout_at_feedback_instant)
        if (
            self.t_duty_updates.size
            and self.target_voltage_all.size
            and self.vout_all.size
        ):
            # Interpolate Vout to the times when duty/target_V are updated (start of each cycle)
            # This represents the Vc_feedback that was used for that cycle's PI calculation.
            vout_at_duty_updates = np.interp(
                self.t_duty_updates,
                self.t_all_spice,
                self.vout_all,
                left=self.vout_all[0] if self.vout_all.size else 0,
                right=self.vout_all[-1] if self.vout_all.size else 0,
            )
            control_error = self.target_voltage_all - vout_at_duty_updates
            self.axes[3].plot(t_ms_duty, control_error)
            self.axes[3].set_ylabel("Control Error (V)")
            self.axes[3].set_title("Control Error (Target V - Vout feedback)")
        else:
            self.axes[3].text(
                0.5,
                0.5,
                "No Error Data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=self.axes[3].transAxes,
            )

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = BoostSimWidget()
    w.show()
    sys.exit(app.exec())
