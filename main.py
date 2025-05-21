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
                  step_time=0.1e-6, t_start=0, t_end=2e-3):
    circuit = Circuit('Boost Converter')
    circuit.V('input', 'vin', circuit.gnd, Vin@u_V)
    circuit.L('1', 'vin', 'n1', L@u_H)
    circuit.S('1', 'n1', 'gnd', 'gate', circuit.gnd, model='SW')
    circuit.model('SW', 'SW', Ron=1@u_mOhm, Roff=1@u_MOhm, Vt=2.5, Vh=0.1)
    circuit.D('1', 'n1', 'n2', model='D')
    circuit.model('D', 'D', IS=1e-15, N=1)
    circuit.C('1', 'n2', circuit.gnd, C@u_F)
    circuit.R('load', 'n2', circuit.gnd, Rload@u_Ohm)
    period = 1/freq
    pulse_width = period * duty_cycle
    circuit.PulseVoltageSource('gate', 'gate', circuit.gnd,
                               initial_value=0@u_V, pulsed_value=5@u_V,
                               pulse_width=pulse_width,
                               period=period)
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=step_time@u_s, end_time=t_end@u_s, start_time=t_start@u_s)
    return analysis

class BoostSimWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Boost Converter Simulator")
        layout = QVBoxLayout(self)

        # Controls
        controls = QHBoxLayout()
        self.vin_spin = self._add_param(controls, "Vin", 5, 0.1, 20, "V")
        self.l_spin = self._add_param(controls, "L", 100, 1, 1000, "uH")
        self.c_spin = self._add_param(controls, "C", 100, 1, 1000, "uF")
        self.r_spin = self._add_param(controls, "R", 10, 1, 100, "Ω")
        self.freq_spin = self._add_param(controls, "F", 50, 1, 500, "kHz")
        self.duty_spin = self._add_param(controls, "D", 60, 1, 99, "%")
        self.step_time_spin = self._add_param(controls, "dt", 0.1, 0.01, 10, "us")
        self.window_spin = self._add_param(controls, "Period", 2, 0.1, 1000, "ms")
        controls.addStretch()
        layout.addLayout(controls)

        # Simulate and Next buttons
        btn_layout = QHBoxLayout()
        self.sim_btn = QPushButton("Simulate")
        self.sim_btn.clicked.connect(self.simulate)
        btn_layout.addWidget(self.sim_btn)
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.simulate_next)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Matplotlib Figure
        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        # Simulation state
        self.t_last = 0  # last simulation end time, in seconds
        self.last_results = None

    def _add_param(self, layout, label, value, minv, maxv, suffix):
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
        Vin = self.vin_spin.value()
        L = self.l_spin.value() * 1e-6
        C = self.c_spin.value() * 1e-6
        Rload = self.r_spin.value()
        freq = self.freq_spin.value() * 1e3
        duty = self.duty_spin.value() / 100.0
        step_time = self.step_time_spin.value() * 1e-6
        period = self.window_spin.value() * 1e-3
        return Vin, L, C, Rload, freq, duty, step_time, period

    def simulate(self):
        Vin, L, C, Rload, freq, duty, step_time, period = self.get_params()
        t_start = 0
        t_end = period
        self.t_last = t_end
        analysis = run_boost_sim(Vin, L, C, Rload, freq, duty, step_time, t_start, t_end)
        self.last_results = (analysis, t_start, t_end)
        self.plot_results(analysis)

    def simulate_next(self):
        Vin, L, C, Rload, freq, duty, step_time, period = self.get_params()
        t_start = self.t_last
        t_end = t_start + period
        self.t_last = t_end
        analysis = run_boost_sim(Vin, L, C, Rload, freq, duty, step_time, t_start, t_end)
        self.last_results = (analysis, t_start, t_end)
        self.plot_results(analysis)

    def plot_results(self, analysis):
        t = np.array(analysis.time) * 1e3  # ms
        vout = np.array(analysis['n2'])
        il = np.array(analysis['L1'])
        gate = np.array(analysis['gate'])
        for ax in self.axes:
            ax.clear()
        self.axes[0].plot(t, vout)
        self.axes[0].set_ylabel("Vout (V)")
        self.axes[0].set_title("Output Voltage")
        self.axes[1].plot(t, il)
        self.axes[1].set_ylabel("Inductor I (A)")
        self.axes[1].set_title("Inductor Current")
        self.axes[2].plot(t, gate)
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

