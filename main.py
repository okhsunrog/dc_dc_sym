import sys
import numpy as np
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

def run_boost_sim(Vin=5, L=100e-6, C=100e-6, Rload=10, freq=50e3, duty_cycle=0.6,
                  step_time=0.1e-6, t_start=0, t_end=2e-3, Il0=0, Vc0=0):
    """
    Run a boost converter simulation for a specified time window with initial conditions.
    
    Args:
        Vin: Input voltage (V)
        L: Inductance (H)
        C: Capacitance (F)
        Rload: Load resistance (Ohm)
        freq: Switching frequency (Hz)
        duty_cycle: Duty cycle (0 to 1)
        step_time: Simulation time step (s)
        t_start: Start time (s)
        t_end: End time (s)
        Il0: Initial inductor current (A)
        Vc0: Initial capacitor voltage (V)
    Returns:
        PySpice analysis object
    """
    circuit = Circuit('Boost Converter')
    circuit.V('input', 'vin', circuit.gnd, Vin@u_V)
    circuit.L('1', 'vin', 'n1', L@u_H, ic=Il0@u_A)  # Set initial inductor current
    circuit.S('1', 'n1', circuit.gnd, 'gate', circuit.gnd, model='SW')
    circuit.model('SW', 'SW', Ron=1@u_mOhm, Roff=1@u_MOhm, Vt=2.5, Vh=0.1)
    circuit.D('1', 'n1', 'n2', model='D')
    circuit.model('D', 'D', IS=1e-15, N=1)
    circuit.C('1', 'n2', circuit.gnd, C@u_F, ic=Vc0@u_V)  # Set initial capacitor voltage
    circuit.R('load', 'n2', circuit.gnd, Rload@u_Ohm)
    period = 1 / freq
    pulse_width = period * duty_cycle
    circuit.PulseVoltageSource('gate', 'gate', circuit.gnd,
                               initial_value=0@u_V, pulsed_value=5@u_V,
                               pulse_width=pulse_width,
                               period=period)
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=step_time@u_s, start_time=t_start@u_s, end_time=t_end@u_s)
    return analysis

class BoostSimWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Boost Converter Simulator with Digital Control")
        layout = QVBoxLayout(self)

        # Controls layout
        controls = QHBoxLayout()
        self.vin_spin = self._add_param(controls, "Vin", 5, 0.1, 20, "V")
        self.l_spin = self._add_param(controls, "L", 100, 1, 1000, "uH")
        self.c_spin = self._add_param(controls, "C", 100, 1, 1000, "uF")
        self.r_spin = self._add_param(controls, "R", 10, 1, 100, "Ω")
        self.freq_spin = self._add_param(controls, "F", 50, 1, 500, "kHz")
        self.duty_spin = self._add_param(controls, "D", 60, 1, 99, "%")
        self.step_time_spin = self._add_param(controls, "dt", 0.1, 0.01, 10, "us")
        self.vref_spin = self._add_param(controls, "Vref", 10, 0.1, 20, "V")  # New: Reference voltage
        self.cycles_spin = self._add_param(controls, "Cycles", 50, 1, 1000, "")  # New: Number of cycles
        controls.addStretch()
        layout.addLayout(controls)

        # Buttons
        btn_layout = QHBoxLayout()
        self.sim_btn = QPushButton("Simulate")
        self.sim_btn.clicked.connect(self.simulate)
        btn_layout.addWidget(self.sim_btn)
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.simulate_next)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Matplotlib figure
        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        # Simulation state variables
        self.Il = 0.0          # Current inductor current (A)
        self.Vc = 0.0          # Current capacitor voltage (V)
        self.integral_e = 0.0  # Integral of error for PI controller
        self.duty = 0.0        # Current duty cycle
        self.t_all = np.array([])    # Accumulated time
        self.vout_all = np.array([]) # Accumulated output voltage
        self.il_all = np.array([])   # Accumulated inductor current
        self.gate_all = np.array([]) # Accumulated gate signal
        self.t_last = 0.0      # Last time point for continuity

    def _add_param(self, layout, label, value, minv, maxv, suffix):
        """Helper to add a labeled spin box to a layout."""
        lbl = QLabel(f"{label}:")
        box = QDoubleSpinBox()
        box.setRange(minv, maxv)
        box.setValue(value)
        box.setMaximumWidth(70)
        box.setSuffix(f" {suffix}")
        layout.addWidget(lbl)
        layout.addWidget(box)
        return box

    def get_params(self):
        """Retrieve parameters from GUI controls."""
        Vin = self.vin_spin.value()
        L = self.l_spin.value() * 1e-6       # Convert uH to H
        C = self.c_spin.value() * 1e-6       # Convert uF to F
        Rload = self.r_spin.value()
        freq = self.freq_spin.value() * 1e3  # Convert kHz to Hz
        initial_duty = self.duty_spin.value() / 100.0  # Convert % to fraction
        step_time = self.step_time_spin.value() * 1e-6  # Convert us to s
        Vref = self.vref_spin.value()
        N_cycles = int(self.cycles_spin.value())  # Number of cycles
        return Vin, L, C, Rload, freq, initial_duty, step_time, Vref, N_cycles

    def simulate(self):
        """Simulate N switching cycles with digital control, resetting state."""
        Vin, L, C, Rload, freq, initial_duty, step_time, Vref, N_cycles = self.get_params()
        T = 1 / freq  # Switching period

        # Reset state
        self.Il = 0.0
        self.Vc = 0.0
        self.integral_e = 0.0
        self.duty = initial_duty
        self.t_all = np.array([])
        self.vout_all = np.array([])
        self.il_all = np.array([])
        self.gate_all = np.array([])
        self.t_last = 0.0

        # PI controller gains (can be tuned)
        Kp = 0.01
        Ki = 10

        # Simulate N cycles
        for _ in range(N_cycles):
            # Run simulation for one switching period
            analysis = run_boost_sim(Vin, L, C, Rload, freq, self.duty, step_time, 
                                   t_start=0, t_end=T, Il0=self.Il, Vc0=self.Vc)
            time = np.array(analysis.time) + self.t_last
            vout = np.array(analysis['n2'])
            il = np.array(analysis['L1'])
            gate = np.array(analysis['gate'])

            # Accumulate results
            self.t_all = np.concatenate((self.t_all, time))
            self.vout_all = np.concatenate((self.vout_all, vout))
            self.il_all = np.concatenate((self.il_all, il))
            self.gate_all = np.concatenate((self.gate_all, gate))

            # Update state for next cycle
            self.Vc = float(vout[-1])  # Final output voltage
            self.Il = float(il[-1])    # Final inductor current
            error = Vref - self.Vc
            self.integral_e += error * T
            self.duty = self.duty + Kp * error + Ki * self.integral_e
            self.duty = max(0, min(self.duty, 0.99))  # Clamp duty cycle
            self.t_last += T

        self.plot_results(self.t_all, self.vout_all, self.il_all, self.gate_all)

    def simulate_next(self):
        """Simulate the next N cycles, continuing from current state."""
        Vin, L, C, Rload, freq, _, step_time, Vref, N_cycles = self.get_params()
        T = 1 / freq

        # PI controller gains (same as in simulate)
        Kp = 0.01
        Ki = 10

        # Simulate N more cycles
        for _ in range(N_cycles):
            analysis = run_boost_sim(Vin, L, C, Rload, freq, self.duty, step_time, 
                                   t_start=0, t_end=T, Il0=self.Il, Vc0=self.Vc)
            time = np.array(analysis.time) + self.t_last
            vout = np.array(analysis['n2'])
            il = np.array(analysis['L1'])
            gate = np.array(analysis['gate'])

            # Accumulate results
            self.t_all = np.concatenate((self.t_all, time))
            self.vout_all = np.concatenate((self.vout_all, vout))
            self.il_all = np.concatenate((self.il_all, il))
            self.gate_all = np.concatenate((self.gate_all, gate))

            # Update state
            self.Vc = float(vout[-1])
            self.Il = float(il[-1])
            error = Vref - self.Vc
            self.integral_e += error * T
            self.duty = self.duty + Kp * error + Ki * self.integral_e
            self.duty = max(0, min(self.duty, 0.99))
            self.t_last += T

        self.plot_results(self.t_all, self.vout_all, self.il_all, self.gate_all)

    def plot_results(self, t, vout, il, gate):
        """Plot accumulated simulation results."""
        t_ms = t * 1e3  # Convert to ms
        for ax in self.axes:
            ax.clear()
        self.axes[0].plot(t_ms, vout)
        self.axes[0].set_ylabel("Vout (V)")
        self.axes[0].set_title("Output Voltage")
        self.axes[1].plot(t_ms, il)
        self.axes[1].set_ylabel("Inductor I (A)")
        self.axes[1].set_title("Inductor Current")
        self.axes[2].plot(t_ms, gate)
        self.axes[2].set_ylabel("Gate (V)")
        self.axes[2].set_xlabel("Time (ms)")
        self.axes[2].set_title("Switch Gate PWM")
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = BoostSimWidget()
    w.show()
    sys.exit(app.exec())
