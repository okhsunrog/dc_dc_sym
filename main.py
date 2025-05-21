import sys
import numpy as np
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

def run_boost_sim(Vin=5, L=100e-6, C=100e-6, Rload=10, freq=50e3, duty_cycle=0.6):
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
    analysis = simulator.transient(step_time=0.1@u_us, end_time=2@u_ms)
    return analysis

class BoostSimWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Boost Converter Simulator")
        layout = QVBoxLayout(self)

        # Parameter controls
        param_layout = QHBoxLayout()
        self.vin_spin = self._add_param(param_layout, "Vin (V)", 5, 0.1, 20)
        self.l_spin = self._add_param(param_layout, "L (uH)", 100, 1, 1000)
        self.c_spin = self._add_param(param_layout, "C (uF)", 100, 1, 1000)
        self.r_spin = self._add_param(param_layout, "Rload (Ohm)", 10, 1, 100)
        self.freq_spin = self._add_param(param_layout, "Freq (kHz)", 50, 1, 500)
        self.duty_spin = self._add_param(param_layout, "Duty (%)", 60, 1, 99)
        layout.addLayout(param_layout)

        # Simulate button
        self.sim_btn = QPushButton("Simulate")
        self.sim_btn.clicked.connect(self.simulate)
        layout.addWidget(self.sim_btn)

        # Matplotlib Figure
        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def _add_param(self, layout, label, value, minv, maxv):
        box = QDoubleSpinBox()
        box.setRange(minv, maxv)
        box.setValue(value)
        if "Duty" in label:
            box.setSuffix(" %")
        elif "Freq" in label:
            box.setSuffix(" kHz")
        elif "L" in label or "C" in label:
            box.setSuffix(" u")
        layout.addWidget(QLabel(label))
        layout.addWidget(box)
        return box

    def simulate(self):
        Vin = self.vin_spin.value()
        L = self.l_spin.value() * 1e-6
        C = self.c_spin.value() * 1e-6
        Rload = self.r_spin.value()
        freq = self.freq_spin.value() * 1e3
        duty = self.duty_spin.value() / 100.0

        analysis = run_boost_sim(Vin, L, C, Rload, freq, duty)
        t = np.array(analysis.time) * 1e3  # ms

        # Clear axes
        for ax in self.axes:
            ax.clear()

        # Output voltage
        self.axes[0].plot(t, np.array(analysis['n2']))
        self.axes[0].set_ylabel("Vout (V)")
        self.axes[0].set_title("Output Voltage")

        # Inductor current
        self.axes[1].plot(t, np.array(analysis['L1']))
        self.axes[1].set_ylabel("Inductor I (A)")
        self.axes[1].set_title("Inductor Current")

        # Gate PWM
        self.axes[2].plot(t, np.array(analysis['gate']))
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

