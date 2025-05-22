import sys
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from PySpice import Simulator

class PIController:
    def __init__(self, Kp, Ki, T_sample, duty_min=0.01, duty_max_limit=0.99, ff_enabled=True):
        self.Kp = Kp
        self.Ki = Ki
        self.T_sample = T_sample
        self.duty_min_internal = duty_min
        self.duty_max_internal = duty_max_limit
        self.ff_enabled = ff_enabled
        self.integral_error = 0.0
        self.last_calculated_duty = 0.0

    def reset(self):
        self.integral_error = 0.0
        self.last_calculated_duty = 0.0

    def update_coeffs(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki

    def set_duty_limits(self, duty_min, duty_max):
        self.duty_min_internal = duty_min
        self.duty_max_internal = duty_max

    def calculate_duty(self, Vref, Vc_feedback, Vin, duty_applied_in_prev_step):
        error = Vref - Vc_feedback
        can_integrate = True
        if (duty_applied_in_prev_step >= self.duty_max_internal and error > 0) or \
           (duty_applied_in_prev_step <= self.duty_min_internal and error < 0):
            can_integrate = False

        if can_integrate:
            self.integral_error += error * self.T_sample

        pi_correction = self.Kp * error + self.Ki * self.integral_error

        calculated_duty = 0.0
        if self.ff_enabled:
            duty_ff = 0.0
            if Vref > Vin and Vref > 0:
                duty_ff = (Vref - Vin) / Vref
            duty_ff = max(self.duty_min_internal, min(duty_ff, min(0.90, self.duty_max_internal)))
            calculated_duty = duty_ff + pi_correction
        else:
            calculated_duty = pi_correction
        self.last_calculated_duty = max(self.duty_min_internal, min(calculated_duty, self.duty_max_internal))
        return self.last_calculated_duty

DEFAULT_KP_CONTROLLER = 0.005
DEFAULT_KI_CONTROLLER = 0.05
DEFAULT_MAX_DUTY_CONTROLLER = 0.90

def run_boost_sim(
    Vin_val=5, L_val=100e-6, C_val=100e-6, Rload_val=10, freq_val=50e3, duty_cycle_val=0.6,
    step_time_val=0.1e-6, t_start_val=0, t_end_val=2e-3, Il0_val=0, Vc0_val=0
):
    circuit = Circuit(f'Boost Converter XSPICE (Duty={duty_cycle_val:.2f} Il0={Il0_val:.2e})')

    circuit.V('input', 'vin', circuit.gnd, Vin_val@u_V)
    node_l_sense_out = 'n_for_l_sense'
    circuit.VoltageSource('Jl_sense', 'vin', node_l_sense_out, 0@u_V)

    l_spice_float = float(L_val)
    il0_spice_float = float(Il0_val)

    circuit.raw_spice = f""".model inductor_ic_model inductoric L={l_spice_float:.7e} IC={il0_spice_float:.7e}
A_L1 {node_l_sense_out} n1 inductor_ic_model"""

    circuit.S('1', 'n1', circuit.gnd, 'gate', circuit.gnd, model='SW')
    circuit.model('SW', 'SW', Ron=1@u_mOhm, Roff=1@u_MOhm, Vt=2.5, Vh=0.1)
    circuit.D('1', 'n1', 'n2', model='D')
    circuit.model('D', 'D', IS=1e-15, N=1)
    circuit.C('1', 'n2', circuit.gnd, C_val@u_F, ic=Vc0_val@u_V)
    circuit.R('load', 'n2', circuit.gnd, Rload_val@u_Ohm)

    period = 1 / freq_val
    safe_duty_cycle = max(0.001, min(float(duty_cycle_val), 0.999))
    pulse_width = period * safe_duty_cycle
    circuit.PulseVoltageSource(
        'gate_drive', 'gate', circuit.gnd,
        initial_value=0@u_V, pulsed_value=5@u_V,
        pulse_width=pulse_width@u_s,
        period=period@u_s,
        delay_time=0@u_s,
        rise_time=10@u_ns,
        fall_time=10@u_ns
    )

    try:
        actual_simulator_object = Simulator.factory(simulator='ngspice-shared')
    except Exception:
        try:
            actual_simulator_object = Simulator.factory(simulator='ngspice-subprocess')
        except Exception as e_subprocess:
            raise RuntimeError(f"SPICE simulator creation error: {e_subprocess}")

    simulation_instance = actual_simulator_object.simulation(
        circuit,
        temperature=25,
        nominal_temperature=25
    )

    analysis = simulation_instance.transient(
        step_time=step_time_val@u_s,
        end_time=t_end_val@u_s,
        start_time=t_start_val@u_s,
        use_initial_condition=True
    )

    return analysis

class BoostSimWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Boost Converter Simulator (XSPICE Sense, cycle-by-cycle)")
        self.main_layout = QVBoxLayout(self)

        controls_row1 = QHBoxLayout()
        self.vin_spin = self._add_param(controls_row1, "Vin", 5, 0.1, 50, "V")
        self.l_spin = self._add_param(controls_row1, "L", 100, 1, 1000, "uH")
        self.c_spin = self._add_param(controls_row1, "C", 100, 1, 1000, "uF")
        self.r_spin = self._add_param(controls_row1, "R", 20, 1, 200, "Ω")

        controls_row2 = QHBoxLayout()
        self.freq_spin = self._add_param(controls_row2, "F", 50, 1, 1000, "kHz")
        self.duty_spin = self._add_param(controls_row2, "Initial Duty", 5, 1, 90, "%")
        self.vref_spin = self._add_param(controls_row2, "Vref", 10, 0.1, 100, "V")
        self.cycles_spin = self._add_param(controls_row2, "PWM Cycles", 500, 50, 5000, "")

        controls_row3 = QHBoxLayout()
        self.kp_spin = self._add_param(controls_row3, "Kp", DEFAULT_KP_CONTROLLER, 0.0000, 0.5, "")
        self.kp_spin.setDecimals(4)
        self.ki_spin = self._add_param(controls_row3, "Ki", DEFAULT_KI_CONTROLLER, 0.0, 20.0, "")
        self.ki_spin.setDecimals(3)
        self.max_duty_spin = self._add_param(controls_row3, "Dmax PI (%)", DEFAULT_MAX_DUTY_CONTROLLER*100, 10, 99, "")
        controls_row3.addStretch()

        controls_row4 = QHBoxLayout()
        self.softstart_cycles_spin = self._add_param(controls_row4, "Soft-start cycles", 50, 0, 20000, "")
        controls_row4.addStretch()

        self.main_layout.addLayout(controls_row1)
        self.main_layout.addLayout(controls_row2)
        self.main_layout.addLayout(controls_row3)
        self.main_layout.addLayout(controls_row4)

        btn_layout = QHBoxLayout()
        self.sim_btn = QPushButton("Simulate (Reset)")
        self.sim_btn.clicked.connect(self.simulate_reset)
        btn_layout.addWidget(self.sim_btn)
        self.next_btn = QPushButton("Next (Continue)")
        self.next_btn.clicked.connect(self.simulate_next)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addStretch()
        self.main_layout.addLayout(btn_layout)

        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout.addWidget(self.toolbar)
        self.main_layout.addWidget(self.canvas)

        self.controller = PIController(
            DEFAULT_KP_CONTROLLER, DEFAULT_KI_CONTROLLER, T_sample=0,
            duty_max_limit=self.max_duty_spin.value()/100.0
        )

        self.last_Vc0_for_next_sim = 0.0
        self.last_Il0_for_next_sim = 0.0
        self.last_applied_duty_for_next_sim = 0.0
        self.t_all_spice = np.array([])
        self.vout_all = np.array([])
        self.il_all = np.array([])
        self.t_duty_updates = np.array([])
        self.duty_all = np.array([])
        self.t_last_plot_point = 0.0

    def _add_param(self, layout, label, value, minv, maxv, suffix):
        lbl = QLabel(f"{label}:")
        box = QDoubleSpinBox()
        box.setRange(minv, maxv)
        box.setValue(value)
        box.setMaximumWidth(110)
        box.setSuffix(f" {suffix}")
        if suffix == "%" or "cycles" in label.lower() or "Dmax" in label:
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
        Vref = self.vref_spin.value()
        Num_PWM_cycles_to_run = int(self.cycles_spin.value())
        T_switching_calc = 1 / freq
        step_time_calc = T_switching_calc / 200
        Kp = self.kp_spin.value()
        Ki = self.ki_spin.value()
        Max_duty_pi = self.max_duty_spin.value() / 100.0
        softstart_cycles = int(self.softstart_cycles_spin.value())
        return (
            Vin, L_val, C_val, Rload, freq, initial_duty_from_gui,
            step_time_calc, Vref, Num_PWM_cycles_to_run, Kp, Ki, T_switching_calc, Max_duty_pi,
            softstart_cycles
        )

    def _run_simulation_segment(
        self, Vin, L_val, C_val, Rload, freq,
        initial_duty_for_segment, step_time_calc, Vref,
        Num_PWM_cycles_segment, Kp, Ki, T_switching, Max_duty_pi,
        softstart_cycles,
        start_Vc0, start_Il0, start_last_applied_duty,
        use_softstart
    ):
        self.controller.T_sample = T_switching
        current_Vc0 = start_Vc0
        current_Il0 = start_Il0
        last_applied_duty = start_last_applied_duty

        # Define initial (low) gains for soft-start
        Kp_init = 0.05 * Kp
        Ki_init = 0.05 * Ki
        Max_duty_init = 0.05 * Max_duty_pi + 0.01  # avoid zero

        for pwm_cycle_num in range(Num_PWM_cycles_segment):
            time_of_current_cycle_start = self.t_last_plot_point
            Vc_feedback = current_Vc0

            # Gain scheduling: ramp Kp/Ki and max duty from initial to nominal over softstart_cycles
            if use_softstart and pwm_cycle_num < softstart_cycles:
                ramp = pwm_cycle_num / max(softstart_cycles, 1)
                Kp_current = Kp_init + (Kp - Kp_init) * ramp
                Ki_current = Ki_init + (Ki - Ki_init) * ramp
                Max_duty_current = Max_duty_init + (Max_duty_pi - Max_duty_init) * ramp
            else:
                Kp_current = Kp
                Ki_current = Ki
                Max_duty_current = Max_duty_pi

            self.controller.update_coeffs(Kp_current, Ki_current)
            self.controller.set_duty_limits(0.01, Max_duty_current)

            # For the very first cycle, use initial duty
            if pwm_cycle_num == 0 and self.t_last_plot_point == 0:
                current_duty_to_apply = initial_duty_for_segment
            else:
                current_duty_to_apply = self.controller.calculate_duty(
                    Vref, Vc_feedback, Vin, last_applied_duty
                )

            self.duty_all = np.append(self.duty_all, current_duty_to_apply)
            self.t_duty_updates = np.append(self.t_duty_updates, time_of_current_cycle_start)
            last_applied_duty = current_duty_to_apply

            analysis = run_boost_sim(
                Vin, L_val, C_val, Rload, freq, current_duty_to_apply,
                step_time_calc, 0, T_switching,
                current_Il0, current_Vc0
            )

            time_segment_spice = np.array(analysis.time)
            if len(time_segment_spice) < 2:
                break

            time_segment_global = time_segment_spice - time_segment_spice[0] + self.t_last_plot_point

            try:
                vout_segment = np.array(analysis.nodes['n2'])
                inductor_current_branch_key = 'vjl_sense'
                if inductor_current_branch_key in analysis.branches:
                    il_segment = np.array(analysis.branches[inductor_current_branch_key])
                else:
                    il_segment = np.zeros_like(vout_segment)
            except Exception:
                break

            min_len_spice = min(len(vout_segment), len(il_segment), len(time_segment_global))
            if len(time_segment_global) != min_len_spice:
                time_segment_global = time_segment_global[:min_len_spice]
                vout_segment = vout_segment[:min_len_spice]
                il_segment = il_segment[:min_len_spice]

            self.t_all_spice = np.concatenate((self.t_all_spice, time_segment_global))
            self.vout_all = np.concatenate((self.vout_all, vout_segment))
            self.il_all = np.concatenate((self.il_all, il_segment))

            if len(vout_segment) > 0:
                current_Vc0 = float(vout_segment[-1])
                current_Il0 = float(il_segment[-1])
                self.t_last_plot_point = time_segment_global[-1]
            else:
                break

        self.last_Vc0_for_next_sim = current_Vc0
        self.last_Il0_for_next_sim = current_Il0
        self.last_applied_duty_for_next_sim = last_applied_duty

    def simulate_reset(self):
        params = self.get_params()
        (
            Vin, L_val, C_val, Rload, freq, initial_duty_gui, step_time_calc,
            Vref, Num_PWM_cycles_to_run, Kp, Ki, T_switching, Max_duty_pi,
            softstart_cycles
        ) = params
        self.controller.reset()
        self.controller.set_duty_limits(0.01, Max_duty_pi)
        self.t_all_spice = np.array([])
        self.vout_all = np.array([])
        self.il_all = np.array([])
        self.duty_all = np.array([])
        self.t_duty_updates = np.array([])
        self.t_last_plot_point = 0.0

        self._run_simulation_segment(
            Vin, L_val, C_val, Rload, freq,
            initial_duty_gui, step_time_calc, Vref,
            Num_PWM_cycles_to_run, Kp, Ki, T_switching, Max_duty_pi,
            softstart_cycles,
            start_Vc0=Vin, start_Il0=0.0, start_last_applied_duty=initial_duty_gui,
            use_softstart=True
        )

        if len(self.t_all_spice) > 0:
            self.plot_results()

    def simulate_next(self):
        if self.t_last_plot_point == 0 and len(self.t_all_spice) == 0:
            return
        params = self.get_params()
        (
            Vin, L_val, C_val, Rload, freq, _, step_time_calc,
            Vref, Num_PWM_cycles_to_run, Kp, Ki, T_switching, Max_duty_pi,
            softstart_cycles
        ) = params

        self._run_simulation_segment(
            Vin, L_val, C_val, Rload, freq,
            self.last_applied_duty_for_next_sim, step_time_calc, Vref,
            Num_PWM_cycles_to_run, Kp, Ki, T_switching, Max_duty_pi,
            softstart_cycles,
            start_Vc0=self.last_Vc0_for_next_sim,
            start_Il0=self.last_Il0_for_next_sim,
            start_last_applied_duty=self.last_applied_duty_for_next_sim,
            use_softstart=False
        )

        if len(self.t_all_spice) > 0:
            self.plot_results()

    def plot_results(self):
        if len(self.t_all_spice) == 0:
            return
        Vref_val = self.vref_spin.value()
        t_ms_spice = self.t_all_spice * 1e3
        t_ms_duty = self.t_duty_updates * 1e3
        for ax in self.axes:
            ax.clear()
            ax.grid(True)
        self.axes[0].plot(t_ms_spice, self.vout_all, label="Vout")
        self.axes[0].axhline(y=Vref_val, color='r', linestyle='--', label=f"Vref = {Vref_val:.2f}V")
        self.axes[0].set_ylabel("Vout (V)")
        self.axes[0].set_title("Output Voltage")
        self.axes[0].legend(loc='lower right')
        self.axes[1].plot(t_ms_spice, self.il_all)
        self.axes[1].set_ylabel("Inductor Current (A)")
        self.axes[1].set_title("Inductor Current")
        if len(self.t_duty_updates) > 0 and len(self.duty_all) > 0:
            self.axes[2].plot(t_ms_duty, self.duty_all * 100, drawstyle='steps-post')
            self.axes[2].set_ylabel("Duty Cycle (%)")
            self.axes[2].set_xlabel("Time (ms)")
            self.axes[2].set_title("PWM Duty Cycle")
            current_max_duty_from_gui = self.max_duty_spin.value()
            self.axes[2].set_ylim(-5, min(current_max_duty_from_gui + 10, 105))
        else:
            self.axes[2].text(0.5, 0.5, 'No Duty Data', horizontalalignment='center', verticalalignment='center')
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = BoostSimWidget()
    w.show()
    sys.exit(app.exec())

