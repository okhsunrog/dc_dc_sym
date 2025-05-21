import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
# from PySpice.Spice.BasicElement import Inductor # Не используется напрямую

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Глобальные (или классовые) коэффициенты PI, чтобы их можно было легче настраивать
DEFAULT_KP = 0.02
DEFAULT_KI = 0.5

def run_boost_sim(Vin=5, L=100e-6, C=100e-6, Rload=10, freq=50e3, duty_cycle=0.6,
                  step_time=0.1e-6, t_start=0, t_end=2e-3, Il0=0, Vc0=0):
    """
    Симуляция повышающего преобразователя для заданного временного окна с начальными условиями.
    """
    circuit = Circuit('Boost Converter')
    circuit.V('input', 'vin', circuit.gnd, Vin@u_V)
    circuit.L('1', 'vin', 'n1', L@u_H, ic=Il0@u_A) # Метка индуктора '1'
    circuit.S('1', 'n1', circuit.gnd, 'gate', circuit.gnd, model='SW')
    circuit.model('SW', 'SW', Ron=1@u_mOhm, Roff=1@u_MOhm, Vt=2.5, Vh=0.1)
    circuit.D('1', 'n1', 'n2', model='D')
    circuit.model('D', 'D', IS=1e-15, N=1)
    circuit.C('1', 'n2', circuit.gnd, C@u_F, ic=Vc0@u_V)
    circuit.R('load', 'n2', circuit.gnd, Rload@u_Ohm)

    period = 1 / freq
    safe_duty_cycle = max(0.001, min(float(duty_cycle), 0.999))
    pulse_width = period * safe_duty_cycle

    circuit.PulseVoltageSource('gate_drive', 'gate', circuit.gnd,
                               initial_value=0@u_V, pulsed_value=5@u_V,
                               pulse_width=pulse_width@u_s,
                               period=period@u_s,
                               delay_time=0@u_s,
                               rise_time=10@u_ns,
                               fall_time=10@u_ns)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=step_time@u_s, start_time=t_start@u_s, end_time=t_end@u_s, use_initial_condition=True)
    return analysis

class BoostSimWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Симулятор повышающего преобразователя с цифровым управлением")
        layout = QVBoxLayout(self)

        controls_row1 = QHBoxLayout()
        self.vin_spin = self._add_param(controls_row1, "Vin", 5, 0.1, 20, "V")
        self.l_spin = self._add_param(controls_row1, "L", 100, 1, 1000, "uH")
        self.c_spin = self._add_param(controls_row1, "C", 100, 1, 1000, "uF")
        self.r_spin = self._add_param(controls_row1, "R", 10, 1, 100, "Ω")

        controls_row2 = QHBoxLayout()
        self.freq_spin = self._add_param(controls_row2, "F", 50, 1, 500, "kHz")
        self.duty_spin = self._add_param(controls_row2, "D (начальный)", 60, 1, 99, "%")
        self.vref_spin = self._add_param(controls_row2, "Vref", 10, 0.1, 30, "V")
        self.cycles_spin = self._add_param(controls_row2, "Циклы", 50, 1, 1000, "")

        controls_row3 = QHBoxLayout()
        self.kp_spin = self._add_param(controls_row3, "Kp", DEFAULT_KP, 0.0001, 1.0, "")
        self.kp_spin.setDecimals(4)
        self.ki_spin = self._add_param(controls_row3, "Ki", DEFAULT_KI, 0.0, 100.0, "")
        self.ki_spin.setDecimals(3)
        controls_row3.addStretch()

        layout.addLayout(controls_row1)
        layout.addLayout(controls_row2)
        layout.addLayout(controls_row3)

        btn_layout = QHBoxLayout()
        self.sim_btn = QPushButton("Симулировать (Сброс)")
        self.sim_btn.clicked.connect(self.simulate)
        btn_layout.addWidget(self.sim_btn)
        self.next_btn = QPushButton("Далее (Продолжить)")
        self.next_btn.clicked.connect(self.simulate_next)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        self.Il = 0.0
        self.Vc = self.vin_spin.value()
        self.integral_e = 0.0
        self.duty = self.duty_spin.value() / 100.0
        self.t_all = np.array([])
        self.vout_all = np.array([])
        self.il_all = np.array([])
        self.gate_all = np.array([])
        self.t_last = 0.0

    def _add_param(self, layout, label, value, minv, maxv, suffix):
        lbl = QLabel(f"{label}:")
        box = QDoubleSpinBox()
        box.setRange(minv, maxv)
        box.setValue(value)
        box.setMaximumWidth(90)
        box.setSuffix(f" {suffix}")
        if suffix == "%" or label == "Циклы":
             box.setDecimals(0)
        elif "V" in suffix or "A" in suffix or "Ω" in suffix:
             box.setDecimals(2)
        elif "H" in suffix or "F" in suffix or "kHz" in suffix :
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
        N_cycles = int(self.cycles_spin.value())
        period = 1 / freq
        step_time = period / 500

        Kp = self.kp_spin.value()
        Ki = self.ki_spin.value()

        return Vin, L_val, C_val, Rload, freq, initial_duty_from_gui, step_time, Vref, N_cycles, Kp, Ki

    def _perform_simulation_step(self, Vin, L_val, C_val, Rload, freq, step_time, Vref, Kp, Ki):
        T = 1 / freq
        analysis = run_boost_sim(Vin, L_val, C_val, Rload, freq, self.duty, step_time,
                                   t_start=0, t_end=T, Il0=self.Il, Vc0=self.Vc)

        time_segment = np.array(analysis.time)
        if len(time_segment) > 0:
            time_segment = time_segment - time_segment[0] + self.t_last


        try:
            vout_segment = np.array(analysis.nodes['n2'])
            # Пытаемся получить ток индуктора из analysis.branches
            # Ключ 'l1' должен быть в нижнем регистре, как указано в логах
            il_segment = np.array(analysis.branches['l1'])
            gate_segment = np.array(analysis.nodes['gate'])

        except KeyError as e:
            logging.error(f"Ошибка доступа к данным анализа (KeyError): {e}. Проверьте имена узлов/элементов.")
            logging.info(f"Доступные узлы (analysis.nodes): {list(analysis.nodes.keys())}")
            logging.info(f"Доступные ветви/токи (analysis.branches): {list(analysis.branches.keys())}")
            return False
        except Exception as e:
            logging.error(f"Общая ошибка доступа к данным анализа: {e}")
            if hasattr(analysis, 'branches'):
                 logging.info(f"Содержимое analysis.branches: {analysis.branches}")
            else:
                 logging.info("analysis.branches отсутствует.")
            return False

        if not (len(time_segment) == len(vout_segment) == len(il_segment) == len(gate_segment)):
            min_len = min(len(time_segment), len(vout_segment), len(il_segment), len(gate_segment))
            if min_len == 0:
                 logging.warning("Один из массивов данных симуляции пуст. Пропуск шага.")
                 return False
            time_segment = time_segment[:min_len]
            vout_segment = vout_segment[:min_len]
            il_segment = il_segment[:min_len]
            gate_segment = gate_segment[:min_len]
            logging.warning("Длины массивов данных не совпадают, произведено усечение.")

        if len(vout_segment) == 0 or len(il_segment) == 0:
             logging.warning("Симуляция вернула пустые данные для Vout или IL. Пропускаем обновление состояния.")
             return False

        self.t_all = np.concatenate((self.t_all, time_segment))
        self.vout_all = np.concatenate((self.vout_all, vout_segment))
        self.il_all = np.concatenate((self.il_all, il_segment))
        self.gate_all = np.concatenate((self.gate_all, gate_segment))

        self.Vc = float(vout_segment[-1])
        self.Il = float(il_segment[-1])

        error = Vref - self.Vc
        self.integral_e += error * T

        self.duty = Kp * error + Ki * self.integral_e
        self.duty = max(0.01, min(self.duty, 0.99))

        if len(time_segment) > 0:
            self.t_last = time_segment[-1]
        else:
            logging.warning("time_segment пуст после конкатенации, t_last не обновлен.")
        return True

    def simulate(self):
        Vin, L_val, C_val, Rload, freq, initial_duty, step_time, Vref, N_cycles, Kp, Ki = self.get_params()

        self.Il = 0.0
        self.Vc = Vin
        self.integral_e = 0.0
        self.duty = initial_duty
        self.t_all = np.array([])
        self.vout_all = np.array([])
        self.il_all = np.array([])
        self.gate_all = np.array([])
        self.t_last = 0.0

        logging.info(f"Начало симуляции: Vin={Vin}V, L={L_val*1e6:.1f}uH, C={C_val*1e6:.1f}uF, R={Rload}Ohm, Freq={freq/1e3:.1f}kHz, Vref={Vref}V")
        logging.info(f"PI: Kp={Kp}, Ki={Ki}. Начальный Duty: {self.duty:.2%}")

        for cycle in range(N_cycles):
            success = self._perform_simulation_step(Vin, L_val, C_val, Rload, freq, step_time, Vref, Kp, Ki)
            if not success:
                logging.error(f"Ошибка на цикле {cycle+1}. Симуляция прервана.")
                break
            logging.info(f"Цикл {cycle+1}/{N_cycles}: Vout = {self.Vc:.2f} В, Duty = {self.duty:.2%}, Il = {self.Il:.2f} А, Err = {Vref - self.Vc:.2f}V, I_e = {self.integral_e:.4e}")

        if len(self.t_all) > 0:
            self.plot_results(self.t_all, self.vout_all, self.il_all, self.gate_all, Vref)
        else:
            logging.warning("Нет данных для отображения.")

    def simulate_next(self):
        Vin, L_val, C_val, Rload, freq, _, step_time, Vref, N_cycles, Kp, Ki = self.get_params()

        if self.t_last == 0 and len(self.t_all) == 0:
             logging.warning("Состояние не инициализировано. Нажмите 'Симулировать (Сброс)' сначала.")
             return

        logging.info(f"Продолжение симуляции: Vref={Vref}V, Kp={Kp}, Ki={Ki}, текущий Duty: {self.duty:.2%}")
        for cycle in range(N_cycles):
            success = self._perform_simulation_step(Vin, L_val, C_val, Rload, freq, step_time, Vref, Kp, Ki)
            if not success:
                logging.error(f"Ошибка на цикле (+{cycle+1}). Симуляция прервана.")
                break
            logging.info(f"Цикл (+{cycle+1}): Vout = {self.Vc:.2f} В, Duty = {self.duty:.2%}, Il = {self.Il:.2f} А, Err = {Vref - self.Vc:.2f}V, I_e = {self.integral_e:.4e}")

        if len(self.t_all) > 0:
            self.plot_results(self.t_all, self.vout_all, self.il_all, self.gate_all, Vref)
        else:
            logging.warning("Нет данных для отображения после продолжения.")

    def plot_results(self, t, vout, il, gate, vref):
        if len(t) == 0:
            logging.warning("Попытка построить пустые графики.")
            return

        t_ms = t * 1e3
        for ax in self.axes:
            ax.clear()
            ax.grid(True)

        self.axes[0].plot(t_ms, vout, label="Vout")
        self.axes[0].axhline(y=vref, color='r', linestyle='--', label=f"Vref = {vref:.2f}V")
        self.axes[0].set_ylabel("Vout (В)")
        self.axes[0].set_title("Выходное напряжение")
        self.axes[0].legend(loc='lower right')

        self.axes[1].plot(t_ms, il)
        self.axes[1].set_ylabel("I индуктора (А)")
        self.axes[1].set_title("Ток индуктора")

        self.axes[2].plot(t_ms, gate)
        self.axes[2].set_ylabel("Затвор (В)")
        self.axes[2].set_xlabel("Время (мс)")
        self.axes[2].set_title("ШИМ затвора")

        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = BoostSimWidget()
    w.show()
    sys.exit(app.exec())
