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
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar # <--- Для зума

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_KP = 0.01 
DEFAULT_KI = 0.1   

def run_boost_sim(Vin=5, L=100e-6, C=100e-6, Rload=10, freq=50e3, duty_cycle=0.6,
                  step_time=0.1e-6, t_start=0, t_end=2e-3, Il0=0, Vc0=0):
    # ... (эта функция без изменений) ...
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
        self.main_layout = QVBoxLayout(self) # Используем self.main_layout

        # ... (Панели управления controls_row1, controls_row2, controls_row3 без изменений) ...
        controls_row1 = QHBoxLayout()
        self.vin_spin = self._add_param(controls_row1, "Vin", 5, 0.1, 20, "V")
        self.l_spin = self._add_param(controls_row1, "L", 100, 1, 1000, "uH") 
        self.c_spin = self._add_param(controls_row1, "C", 100, 1, 1000, "uF")
        self.r_spin = self._add_param(controls_row1, "R", 80, 1, 200, "Ω") 

        controls_row2 = QHBoxLayout()
        self.freq_spin = self._add_param(controls_row2, "F", 50, 1, 500, "kHz")
        self.duty_spin = self._add_param(controls_row2, "D (начальный)", 50, 1, 99, "%") 
        self.vref_spin = self._add_param(controls_row2, "Vref", 10, 0.1, 30, "V")
        self.cycles_spin = self._add_param(controls_row2, "Всего ШИМ циклов", 200, 10, 5000, "") 

        controls_row3 = QHBoxLayout()
        self.kp_spin = self._add_param(controls_row3, "Kp", DEFAULT_KP, 0.0001, 1.0, "")
        self.kp_spin.setDecimals(4)
        self.ki_spin = self._add_param(controls_row3, "Ki", DEFAULT_KI, 0.0, 100.0, "")
        self.ki_spin.setDecimals(3)
        controls_row3.addStretch()

        self.main_layout.addLayout(controls_row1)
        self.main_layout.addLayout(controls_row2)
        self.main_layout.addLayout(controls_row3)

        # ... (Кнопки btn_layout без изменений) ...
        btn_layout = QHBoxLayout()
        self.sim_btn = QPushButton("Симулировать (Сброс)")
        self.sim_btn.clicked.connect(self.simulate)
        btn_layout.addWidget(self.sim_btn)
        self.next_btn = QPushButton("Далее (Повтор)") 
        self.next_btn.clicked.connect(self.simulate_next)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addStretch()
        self.main_layout.addLayout(btn_layout)


        # График
        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True) # Увеличим немного высоту для панели
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Добавляем NavigationToolbar для зума и панорамирования
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.main_layout.addWidget(self.toolbar) # Добавляем панель инструментов
        self.main_layout.addWidget(self.canvas)  # Добавляем сам холст графика

        # Переменные состояния PI-регулятора
        self.integral_e = 0.0
        
        # Глобальные массивы для накопления всех данных для графика
        self.t_all_spice = np.array([]) # Время из SPICE симуляций (детальное)
        self.vout_all = np.array([])
        self.il_all = np.array([])
        # self.gate_all = np.array([]) # Больше не нужен
        
        self.t_duty_updates = np.array([]) # Время, когда Duty обновлялся (начало каждого цикла)
        self.duty_all = np.array([])       # Значения Duty
        
        self.t_last_plot_point = 0.0 # Время окончания последнего отображенного *SPICE* сегмента

    def _add_param(self, layout, label, value, minv, maxv, suffix):
        # ... (без изменений) ...
        lbl = QLabel(f"{label}:")
        box = QDoubleSpinBox()
        box.setRange(minv, maxv)
        box.setValue(value)
        box.setMaximumWidth(110) 
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
        # ... (без изменений) ...
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

        current_Il0 = 0.0
        current_Vc0 = Vin 
        current_duty = initial_duty_gui
        self.integral_e = 0.0 
        
        self.t_all_spice = np.array([])
        self.vout_all = np.array([])
        self.il_all = np.array([])
        self.duty_all = np.array([])
        self.t_duty_updates = np.array([])
        self.t_last_plot_point = 0.0
        
        logging.info(f"Начало симуляции: Vin={Vin}V, L={L_val*1e6:.1f}uH, C={C_val*1e6:.1f}uF, R={Rload}Ohm, Freq={freq/1e3:.1f}kHz, Vref={Vref}V")
        logging.info(f"Всего ШИМ циклов для симуляции: {Total_PWM_cycles}")
        logging.info(f"PI: Kp={Kp}, Ki={Ki}. Начальный Duty для 1-го цикла: {current_duty:.2%}")

        for pwm_cycle_num in range(Total_PWM_cycles):
            time_of_current_cycle_start = self.t_last_plot_point
            Vc_feedback = current_Vc0 

            if pwm_cycle_num == 0: # Для первого цикла используем initial_duty_gui
                # current_duty уже установлен в initial_duty_gui
                pass
            else: 
                error = Vref - Vc_feedback
                can_integrate = True
                if (current_duty >= 0.99 and error > 0) or \
                   (current_duty <= 0.01 and error < 0):
                    can_integrate = False
                
                if can_integrate:
                    self.integral_e += error * T_switching

                duty_ff = 0.0
                if Vref > Vin and Vref > 0:
                    duty_ff = (Vref - Vin) / Vref
                duty_ff = max(0.01, min(duty_ff, 0.90))

                pi_correction = Kp * error + Ki * self.integral_e
                current_duty = duty_ff + pi_correction
                current_duty = max(0.01, min(current_duty, 0.99))
            
            # Сохраняем Duty и время его применения
            self.duty_all = np.append(self.duty_all, current_duty)
            self.t_duty_updates = np.append(self.t_duty_updates, time_of_current_cycle_start)

            analysis = run_boost_sim(Vin, L_val, C_val, Rload, freq, current_duty, 
                                       step_time_calc, 0, T_switching, 
                                       current_Il0, current_Vc0)

            time_segment_spice = np.array(analysis.time)
            if len(time_segment_spice) < 2:
                logging.error(f"ШИМ Цикл {pwm_cycle_num+1}: Симуляция не вернула достаточно точек.")
                break
            
            time_segment_global = time_segment_spice - time_segment_spice[0] + self.t_last_plot_point 

            try:
                vout_segment = np.array(analysis.nodes['n2'])
                il_segment = np.array(analysis.branches['l1'])
                # gate_segment больше не нужен для графика
            except KeyError as e:
                logging.error(f"ШИМ Цикл {pwm_cycle_num+1}: Ошибка доступа к данным (KeyError): {e}.")
                break
            except Exception as e:
                logging.error(f"ШИМ Цикл {pwm_cycle_num+1}: Общая ошибка доступа: {e}")
                break
            
            min_len_spice = min(len(vout_segment), len(il_segment), len(time_segment_global))
            if len(time_segment_global) != min_len_spice : # Если какая-то из выборок короче
                 logging.warning(f"ШИМ Цикл {pwm_cycle_num+1}: Несовпадение длин массивов SPICE. Усечение.")
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
                logging.error(f"ШИМ Цикл {pwm_cycle_num+1}: Нет данных vout для обновления состояния.")
                break

            if (pwm_cycle_num + 1) % 20 == 0 or pwm_cycle_num == Total_PWM_cycles -1 : 
                 logging.info(f"ШИМ Цикл {pwm_cycle_num+1} завершен: Vc_end={current_Vc0:.2f}В, Il_end={current_Il0:.2f}A, Duty={current_duty:.2%}, Vc_fdbk={Vc_feedback:.2f}V, Err={Vref-Vc_feedback:.2f}V, I_e={self.integral_e:.4e}")

        if len(self.t_all_spice) > 0:
            self.plot_results() # Больше не передаем аргументы, т.к. они члены класса
        else:
            logging.warning("Нет данных для отображения после серии симуляций.")

    def simulate_next(self):
        logging.info("'Далее (Повтор)' вызовет новую серию симуляций с текущими параметрами GUI.")
        self.simulate() 

    def plot_results(self): # Убраны аргументы
        if len(self.t_all_spice) == 0:
            logging.warning("Попытка построить пустые графики.")
            return

        Vref_val = self.vref_spin.value() # Получаем Vref для графика

        t_ms_spice = self.t_all_spice * 1e3
        t_ms_duty = self.t_duty_updates * 1e3

        for ax in self.axes:
            ax.clear()
            ax.grid(True)

        self.axes[0].plot(t_ms_spice, self.vout_all, label="Vout")
        self.axes[0].axhline(y=Vref_val, color='r', linestyle='--', label=f"Vref = {Vref_val:.2f}V")
        self.axes[0].set_ylabel("Vout (В)")
        self.axes[0].set_title("Выходное напряжение")
        self.axes[0].legend(loc='lower right')

        self.axes[1].plot(t_ms_spice, self.il_all)
        self.axes[1].set_ylabel("I индуктора (А)")
        self.axes[1].set_title("Ток индуктора")

        # График Duty Cycle
        # Используем 'steps-post' для ступенчатого графика, т.к. Duty меняется в начале цикла
        self.axes[2].plot(t_ms_duty, self.duty_all * 100, drawstyle='steps-post') # Умножаем на 100 для %
        self.axes[2].set_ylabel("Скважность Duty (%)")
        self.axes[2].set_xlabel("Время (мс)")
        self.axes[2].set_title("Скважность ШИМ")
        self.axes[2].set_ylim(0, 100) # Ограничиваем ось Y для Duty от 0 до 100%

        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = BoostSimWidget()
    w.show()
    sys.exit(app.exec())
