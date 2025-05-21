import sys
import numpy as np
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QPushButton, QSlider, QGroupBox, QGridLayout
)
from PySide6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

def run_boost_sim(Vin=5, L=100e-6, C=100e-6, Rload=10, freq=50e3, duty_cycle=0.6, step_time=0.1e-6, end_time=2e-3):
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
    analysis = simulator.transient(step_time=step_time@u_s, end_time=end_time@u_s)
    return analysis

class BoostSimWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Boost Converter Simulator")
        layout = QVBoxLayout(self)

        # Parameter controls
        param_group = QGroupBox("Simulation Parameters")
        param_layout = QGridLayout()
        self.vin_spin = self._add_param(param_layout, "Vin (V)", 5, 0.1, 20, 0)
        self.l_spin = self._add_param(param_layout, "L (uH)", 100, 1, 1000, 1)
        self.c_spin = self._add_param(param_layout, "C (uF)", 100, 1, 1000, 2)
        self.r_spin = self._add_param(param_layout, "Rload (Ohm)", 10, 1, 100, 3)
        self.freq_spin = self._add_param(param_layout, "Freq (kHz)", 50, 1, 500, 4)
        self.duty_spin = self._add_param(param_layout, "Duty (%)", 60, 1, 99, 5)
        self.end_time_spin = self._add_param(param_layout, "End Time (ms)", 2, 0.1, 20, 6)
        self.step_time_spin = self._add_param(param_layout, "Step Time (us)", 0.1, 0.01, 10, 7)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Simulate button
        self.sim_btn = QPushButton("Simulate")
        self.sim_btn.clicked.connect(self.simulate)
        layout.addWidget(self.sim_btn)

        # Matplotlib Figure
        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Playback controls
        playback_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(self.pause_btn)

        playback_layout.addWidget(QLabel("Speed:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 10.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        playback_layout.addWidget(self.speed_spin)

        playback_layout.addWidget(QLabel("Navigate:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.slider_moved)
        playback_layout.addWidget(self.slider)

        layout.addLayout(playback_layout)

        # Animation state
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.anim_idx = 0
        self.anim_max = 0
        self.t = None
        self.vout = None
        self.il = None
        self.gate = None

    def _add_param(self, layout, label, value, minv, maxv, row):
        box = QDoubleSpinBox()
        box.setRange(minv, maxv)
        box.setValue(value)
        if "Duty" in label:
            box.setSuffix(" %")
        elif "Freq" in label:
            box.setSuffix(" kHz")
        elif "L" in label or "C" in label:
            box.setSuffix(" u")
        elif "End Time" in label:
            box.setSuffix(" ms")
        elif "Step Time" in label:
            box.setSuffix(" us")
        layout.addWidget(QLabel(label), row, 0)
        layout.addWidget(box, row, 1)
        return box

    def simulate(self):
        Vin = self.vin_spin.value()
        L = self.l_spin.value() * 1e-6
        C = self.c_spin.value() * 1e-6
        Rload = self.r_spin.value()
        freq = self.freq_spin.value() * 1e3
        duty = self.duty_spin.value() / 100.0
        end_time = self.end_time_spin.value() * 1e-3
        step_time = self.step_time_spin.value() * 1e-6

        analysis = run_boost_sim(Vin, L, C, Rload, freq, duty, step_time, end_time)
        self.t = np.array(analysis.time) * 1e3  # ms
        self.vout = np.array(analysis['n2'])
        self.il = np.array(analysis['L1'])
        self.gate = np.array(analysis['gate'])

        self.anim_idx = 0
        self.anim_max = len(self.t)
        self.slider.setEnabled(True)
        self.slider.setMaximum(self.anim_max - 1)
        self.slider.setValue(0)

        self.plot_at(self.anim_idx)

    def plot_at(self, idx):
        # Plot up to idx
        for ax in self.axes:
            ax.clear()
        if self.t is not None:
            self.axes[0].plot(self.t[:idx+1], self.vout[:idx+1])
            self.axes[0].set_ylabel("Vout (V)")
            self.axes[0].set_title("Output Voltage")
            self.axes[1].plot(self.t[:idx+1], self.il[:idx+1])
            self.axes[1].set_ylabel("Inductor I (A)")
            self.axes[1].set_title("Inductor Current")
            self.axes[2].plot(self.t[:idx+1], self.gate[:idx+1])
            self.axes[2].set_ylabel("Gate (V)")
            self.axes[2].set_xlabel("Time (ms)")
            self.axes[2].set_title("Switch Gate PWM")
        self.fig.tight_layout()
        self.canvas.draw()

    def play(self):
        if self.t is None:
            return
        self.timer.start(int(20 / self.speed_spin.value()))  # ms per frame

    def pause(self):
        self.timer.stop()

    def update_animation(self):
        if self.anim_idx < self.anim_max - 1:
            self.anim_idx += 1
            self.slider.blockSignals(True)
            self.slider.setValue(self.anim_idx)
            self.slider.blockSignals(False)
            self.plot_at(self.anim_idx)
        else:
            self.timer.stop()

    def slider_moved(self, value):
        self.anim_idx = value
        self.plot_at(self.anim_idx)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = BoostSimWidget()
    w.show()
    sys.exit(app.exec())

