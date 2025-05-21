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

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# РЕКОМЕНДУЕТСЯ НАЧАТЬ С МАЛЕНЬКИХ ЗНАЧЕНИЙ Kp, Ki ДЛЯ НОВОГО РЕЖИМА
DEFAULT_KP = 0.01  # Уменьшено для начала
DEFAULT_KI = 0.1   # Уменьшено для начала (ранее было 0.01, потом Ki рос до 15-25)
                   # Ki=15 или 25 будет СЛИШКОМ большим для обновления каждый цикл ШИМ

def run_boost_sim(Vin=5, L=100e-6, C=100e-6, Rload=10, freq=50e3, duty_cycle=0.6,
                  step_time=0.1e-6, t_start=0, t_end=2e-3, Il0=0, Vc0=0):
    # Эта функция остается без изменений
    circuit = Circuit('Boost Converter')
    circuit.V('input', 'vin', circuit.gnd, Vin@u_V)
    circuit.L('1', 'vin', 'n1', L@u_H, ic=Il0@u_A)
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
        self.setWindowTitle("Симулятор повышающего преобразователя (управление каждый цикл ШИМ)")
        layout = QVBoxLayout(self)

        controls_row1 = QHBoxLayout()
        self.vin_spin = self._add_param(controls_row1, "Vin", 5, 0.1, 20, "V")
        self.l_spin = self._add_param(controls_row1, "L", 100, 1, 1000, "uH") # Попробуйте L=100uH
        self.c_spin = self._add_param(controls_row1, "C", 100, 1, 1000, "uF")
        self.r_spin = self._add_param(controls_row1, "R", 80, 1, 200, "Ω") # Попробуйте Rload=80-100 Ом

        controls_row2 = QHBoxLayout()
        self.freq_spin = self._add_param(controls_row2, "F", 50, 1, 500, "kHz")
        self.duty_spin = self._add_param(controls_row2, "D (начальный)", 50, 1, 99, "%") # Начальный D около 50%
        self.vref_spin = self._add_param(controls_row2, "Vref", 10, 0.1, 30, "V")
        self.cycles_spin = self._add_param(controls_row2, "Всего ШИМ циклов", 200, 10, 5000, "") # Теперь это общее число циклов ШИМ

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
        self.next_btn = QPushButton("Далее (Повтор)") 
        self.next_btn.clicked.connect(self.simulate_next) # simulate_next будет просто перезапускать
        btn_layout.addWidget(self.next_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        # Переменные состояния PI-регулятора (остаются)
        self.integral_e = 0.0
        # self.duty, self.Il, self.Vc будут локальными для цикла симуляции или передаваться как IC
        # Глобальные массивы для накопления всех данных для графика
        self.t_all = np.array([])
        self.vout_all = np.array([])
        self.il_all = np.array([])
        self.gate_all = np.array([])
        self.t_last_plot = 0.0 # Время окончания последнего отображенного сегмента

    def _add_param(self, layout, label, value, minv, maxv, suffix):
        # Эта функция остается без изменений
        lbl = QLabel(f"{label}:")
        box = QDoubleSpinBox()
        box.setRange(minv, maxv)
        box.setValue(value)
        box.setMaximumWidth(110) # Немного увеличено для "Всего ШИМ циклов"
        box.setSuffix(f" {suffix}")
        if suffix == "%" or "циклов" in label.lower() : 
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
        Total_PWM_cycles = int(self.cycles_spin.value())
        
        T_switching_calc = 1 / freq
        step_time_calc = T_switching_calc / 500 

        Kp = self.kp_spin.value()
        Ki = self.ki_spin.value()

        return Vin, L_val, C_val, Rload, freq, initial_duty_from_gui, \
               step_time_calc, Vref, Total_PWM_cycles, Kp, Ki, T_switching_calc

    def simulate(self):
        Vin, L_val, C_val, Rload, freq, initial_duty_gui, step_time_calc, \
        Vref, Total_PWM_cycles, Kp, Ki, T_switching = self.get_params()

        # Сброс состояния для новой серии симуляций
        current_Il0 = 0.0
        current_Vc0 = Vin 
        current_duty = initial_duty_gui
        self.integral_e = 0.0 # Сбрасываем интегратор
        
        # Сброс массивов для графиков
        self.t_all = np.array([])
        self.vout_all = np.array([])
        self.il_all = np.array([])
        self.gate_all = np.array([])
        self.t_last_plot = 0.0
        
        logging.info(f"Начало симуляции: Vin={Vin}V, L={L_val*1e6:.1f}uH, C={C_val*1e6:.1f}uF, R={Rload}Ohm, Freq={freq/1e3:.1f}kHz, Vref={Vref}V")
        logging.info(f"Всего ШИМ циклов для симуляции: {Total_PWM_cycles}")
        logging.info(f"PI: Kp={Kp}, Ki={Ki}. Начальный Duty для 1-го цикла: {current_duty:.2%}")

        for pwm_cycle_num in range(Total_PWM_cycles):
            # Vc_feedback для текущего расчета Duty - это напряжение в начале текущего цикла ШИМ
            Vc_feedback = current_Vc0 

            # Рассчитываем Duty для текущего цикла (кроме самого первого)
            if pwm_cycle_num > 0: 
                error = Vref - Vc_feedback
                
                # Anti-windup и накопление интеграла
                # pi_step_duration для интеграла теперь T_switching
                can_integrate = True
                # Используем current_duty, который будет применен в этом цикле
                if (current_duty >= 0.99 and error > 0) or \
                   (current_duty <= 0.01 and error < 0):
                    can_integrate = False
                
                if can_integrate:
                    self.integral_e += error * T_switching # pi_step_duration is T_switching

                # Расчет Feed-Forward Duty
                duty_ff = 0.0
                if Vref > Vin and Vref > 0:
                    duty_ff = (Vref - Vin) / Vref
                duty_ff = max(0.01, min(duty_ff, 0.90))

                # PI корректирует значение Feed-Forward
                pi_correction = Kp * error + Ki * self.integral_e
                current_duty = duty_ff + pi_correction
                current_duty = max(0.01, min(current_duty, 0.99))
            
            # Симуляция ОДНОГО цикла ШИМ
            analysis = run_boost_sim(Vin, L_val, C_val, Rload, freq, current_duty, 
                                       step_time_calc, 0, T_switching, 
                                       current_Il0, current_Vc0)

            # Обработка результатов
            time_segment_spice = np.array(analysis.time)
            if len(time_segment_spice) < 2:
                logging.error(f"ШИМ Цикл {pwm_cycle_num+1}: Симуляция не вернула достаточно точек.")
                break
            
            time_segment_global = time_segment_spice - time_segment_spice[0] + self.t_last_plot 

            try:
                vout_segment = np.array(analysis.nodes['n2'])
                il_segment = np.array(analysis.branches['l1'])
                gate_segment = np.array(analysis.nodes['gate'])
            except KeyError as e:
                logging.error(f"ШИМ Цикл {pwm_cycle_num+1}: Ошибка доступа к данным (KeyError): {e}.")
                # (можно добавить логирование доступных узлов/ветвей)
                break
            except Exception as e:
                logging.error(f"ШИМ Цикл {pwm_cycle_num+1}: Общая ошибка доступа: {e}")
                break
            
            # Проверка длин и конкатенация (упрощенная, предполагаем, что длины совпадут)
            if not (len(vout_segment) == len(time_segment_global) and \
                    len(il_segment) == len(time_segment_global) and \
                    len(gate_segment) == len(time_segment_global)):
                logging.warning(f"ШИМ Цикл {pwm_cycle_num+1}: Несовпадение длин массивов. Пропуск цикла для графика.")
            else:
                self.t_all = np.concatenate((self.t_all, time_segment_global))
                self.vout_all = np.concatenate((self.vout_all, vout_segment))
                self.il_all = np.concatenate((self.il_all, il_segment))
                self.gate_all = np.concatenate((self.gate_all, gate_segment))

            # Обновление начальных условий для СЛЕДУЮЩЕГО цикла ШИМ
            if len(vout_segment) > 0: # Убедимся, что есть данные
                current_Vc0 = float(vout_segment[-1]) 
                current_Il0 = float(il_segment[-1])
                self.t_last_plot = time_segment_global[-1]
            else: # Если данных нет, прерываем, чтобы избежать ошибок
                logging.error(f"ШИМ Цикл {pwm_cycle_num+1}: Нет данных vout для обновления состояния.")
                break

            if (pwm_cycle_num + 1) % 10 == 0 or pwm_cycle_num == Total_PWM_cycles -1 : # Логирование каждые 10 циклов
                 logging.info(f"ШИМ Цикл {pwm_cycle_num+1} завершен: Vc_end={current_Vc0:.2f}В, Il_end={current_Il0:.2f}A, Duty={current_duty:.2%}, Vc_fdbk={Vc_feedback:.2f}V, Err={Vref-Vc_feedback:.2f}V, I_e={self.integral_e:.4e}")

        if len(self.t_all) > 0:
            self.plot_results(self.t_all, self.vout_all, self.il_all, self.gate_all, Vref)
        else:
            logging.warning("Нет данных для отображения после серии симуляций.")

    def simulate_next(self):
        logging.info("'Далее (Повтор)' вызовет новую серию симуляций с текущими параметрами GUI.")
        self.simulate() 

    def plot_results(self, t, vout, il, gate, vref):
        # Эта функция остается без изменений
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
