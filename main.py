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
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Начало: Класс PIController ---
class PIController:
    def __init__(self, Kp, Ki, T_sample, duty_min=0.01, duty_max=0.99, ff_enabled=True):
        self.Kp = Kp
        self.Ki = Ki
        self.T_sample = T_sample  # Период дискретизации (равен T_switching)
        self.duty_min = duty_min
        self.duty_max = duty_max
        self.ff_enabled = ff_enabled # Включен ли Feed-Forward

        self.integral_error = 0.0
        self.last_duty = 0.0 # Можно хранить последнее значение duty для информации или anti-windup

    def reset(self):
        self.integral_error = 0.0
        self.last_duty = 0.0

    def update_coeffs(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        # Можно также сбросить интегратор при смене коэффициентов, если это нужно
        # self.reset() 

    def calculate_duty(self, Vref, Vc_feedback, Vin, current_applied_duty):
        error = Vref - Vc_feedback
        
        # Anti-windup для интегратора
        can_integrate = True
        # Проверяем current_applied_duty (который был применен в прошлом цикле и привел к Vc_feedback)
        # или self.last_duty (если мы хотим проверять то, что мы сами посчитали в прошлый раз)
        # Используем current_applied_duty, так как это то, что реально было в системе.
        if (current_applied_duty >= self.duty_max and error > 0) or \
           (current_applied_duty <= self.duty_min and error < 0):
            can_integrate = False
        
        if can_integrate:
            self.integral_error += error * self.T_sample

        pi_correction = self.Kp * error + self.Ki * self.integral_error
        
        calculated_duty = 0.0
        if self.ff_enabled:
            duty_ff = 0.0
            if Vref > Vin and Vref > 0:
                duty_ff = (Vref - Vin) / Vref
            duty_ff = max(self.duty_min, min(duty_ff, 0.90)) # FF ограничен до 90%
            calculated_duty = duty_ff + pi_correction
        else: # Если Feed-Forward отключен, PI работает от "нуля"
            calculated_duty = pi_correction

        self.last_duty = max(self.duty_min, min(calculated_duty, self.duty_max))
        return self.last_duty
# --- Конец: Класс PIController ---


DEFAULT_KP_CONTROLLER = 0.01 
DEFAULT_KI_CONTROLLER = 0.1   

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
        self.main_layout = QVBoxLayout(self) 

        controls_row1 = QHBoxLayout()
        self.vin_spin = self._add_param(controls_row1, "Vin", 5, 0.1, 20, "V")
        self.l_spin = self._add_param(controls_row1, "L", 100, 1, 1000, "uH") 
        self.c_spin = self._add_param(controls_row1, "C", 100, 1, 1000, "uF")
        self.r_spin = self._add_param(controls_row1, "R", 80, 1, 200, "Ω") 

        controls_row2 = QHBoxLayout()
        self.freq_spin = self._add_param(controls_row2, "F", 50, 1, 500, "kHz")
        self.duty_spin = self._add_param(controls_row2, "D (начальный)", 10, 1, 99, "%") # <--- Изменено начальное D для теста
        self.vref_spin = self._add_param(controls_row2, "Vref", 10, 0.1, 30, "V")
        self.cycles_spin = self._add_param(controls_row2, "Всего ШИМ циклов", 200, 10, 5000, "") 

        controls_row3 = QHBoxLayout()
        self.kp_spin = self._add_param(controls_row3, "Kp", DEFAULT_KP_CONTROLLER, 0.0000, 1.0, "") # Min Kp может быть 0
        self.kp_spin.setDecimals(4)
        self.ki_spin = self._add_param(controls_row3, "Ki", DEFAULT_KI_CONTROLLER, 0.0, 100.0, "")
        self.ki_spin.setDecimals(3)
        controls_row3.addStretch()

        self.main_layout.addLayout(controls_row1)
        self.main_layout.addLayout(controls_row2)
        self.main_layout.addLayout(controls_row3)

        btn_layout = QHBoxLayout()
        self.sim_btn = QPushButton("Симулировать (Сброс)")
        self.sim_btn.clicked.connect(self.simulate)
        btn_layout.addWidget(self.sim_btn)
        self.next_btn = QPushButton("Далее (Повтор)") 
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

        # Создаем экземпляр контроллера
        # T_sample будет установлен в simulate() на основе частоты
        self.controller = PIController(DEFAULT_KP_CONTROLLER, DEFAULT_KI_CONTROLLER, T_sample=0) 
        
        self.t_all_spice = np.array([]) 
        self.vout_all = np.array([])
        self.il_all = np.array([])
        self.t_duty_updates = np.array([]) 
        self.duty_all = np.array([])       
        self.t_last_plot_point = 0.0       

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

        # Инициализация или обновление контроллера
        self.controller.T_sample = T_switching
        self.controller.update_coeffs(Kp, Ki)
        self.controller.reset()
        # self.controller.ff_enabled = True # Убедитесь, что FF включен, если хотите

        current_Il0 = 0.0
        current_Vc0 = Vin 
        # current_duty будет устанавливаться ПЕРЕД симуляцией каждого цикла
        
        self.t_all_spice = np.array([])
        self.vout_all = np.array([])
        self.il_all = np.array([])
        self.duty_all = np.array([])
        self.t_duty_updates = np.array([])
        self.t_last_plot_point = 0.0
        
        logging.info(f"Начало симуляции: Vin={Vin}V, L={L_val*1e6:.1f}uH, C={C_val*1e6:.1f}uF, R={Rload}Ohm, Freq={freq/1e3:.1f}kHz, Vref={Vref}V")
        logging.info(f"Всего ШИМ циклов для симуляции: {Total_PWM_cycles}")
        logging.info(f"PI Controller: Kp={self.controller.Kp}, Ki={self.controller.Ki}, T_sample={self.controller.T_sample:.2e}s")

        # current_duty_to_apply - это скважность, которая будет применена в ТЕКУЩЕМ цикле симуляции
        current_duty_to_apply = initial_duty_gui 
        # last_applied_duty - скважность, которая была применена в ПРЕДЫДУЩЕМ цикле (для anti-windup)
        last_applied_duty = initial_duty_gui 


        for pwm_cycle_num in range(Total_PWM_cycles):
            time_of_current_cycle_start = self.t_last_plot_point
            Vc_feedback = current_Vc0 

            if pwm_cycle_num == 0:
                # Для первого цикла используем initial_duty_gui, он уже в current_duty_to_apply
                logging.info(f"ШИМ Цикл 1: Применение начального Duty = {current_duty_to_apply:.2%}")
            else: 
                # Рассчитываем Duty для текущего цикла, используя состояние Vc0 из предыдущего
                current_duty_to_apply = self.controller.calculate_duty(Vref, Vc_feedback, Vin, last_applied_duty)
            
            self.duty_all = np.append(self.duty_all, current_duty_to_apply)
            self.t_duty_updates = np.append(self.t_duty_updates, time_of_current_cycle_start)
            
            # Запоминаем, какой duty был применен, для следующей итерации anti-windup
            last_applied_duty = current_duty_to_apply 

            analysis = run_boost_sim(Vin, L_val, C_val, Rload, freq, current_duty_to_apply, 
                                       step_time_calc, 0, T_switching, 
                                       current_Il0, current_Vc0)
            # ... (остальная часть обработки результатов и логирования как раньше) ...
            time_segment_spice = np.array(analysis.time)
            if len(time_segment_spice) < 2:
                logging.error(f"ШИМ Цикл {pwm_cycle_num+1}: Симуляция не вернула достаточно точек.")
                break
            
            time_segment_global = time_segment_spice - time_segment_spice[0] + self.t_last_plot_point 

            try:
                vout_segment = np.array(analysis.nodes['n2'])
                il_segment = np.array(analysis.branches['l1'])
            except KeyError as e:
                logging.error(f"ШИМ Цикл {pwm_cycle_num+1}: Ошибка доступа к данным (KeyError): {e}.")
                break
            except Exception as e:
                logging.error(f"ШИМ Цикл {pwm_cycle_num+1}: Общая ошибка доступа: {e}")
                break
            
            min_len_spice = min(len(vout_segment), len(il_segment), len(time_segment_global))
            if len(time_segment_global) != min_len_spice : 
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
                 logging.info(f"ШИМ Цикл {pwm_cycle_num+1} завершен: Vc_end={current_Vc0:.2f}В, Il_end={current_Il0:.2f}A, Duty_applied={current_duty_to_apply:.2%}, Vc_fdbk={Vc_feedback:.2f}V, Err_used={Vref-Vc_feedback:.2f}V, I_e={self.controller.integral_error:.4e}")


        if len(self.t_all_spice) > 0:
            self.plot_results()
        else:
            logging.warning("Нет данных для отображения после серии симуляций.")

    def simulate_next(self):
        logging.info("'Далее (Повтор)' вызовет новую серию симуляций с текущими параметрами GUI.")
        self.simulate() 

    def plot_results(self):
        # ... (без изменений) ...
        if len(self.t_all_spice) == 0:
            logging.warning("Попытка построить пустые графики.")
            return

        Vref_val = self.vref_spin.value() 

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
        
        if len(self.t_duty_updates) > 0 and len(self.duty_all) > 0: # Проверка, что есть данные для Duty
            self.axes[2].plot(t_ms_duty, self.duty_all * 100, drawstyle='steps-post') 
            self.axes[2].set_ylabel("Скважность Duty (%)")
            self.axes[2].set_xlabel("Время (мс)")
            self.axes[2].set_title("Скважность ШИМ")
            self.axes[2].set_ylim(0, 100) 
        else:
            self.axes[2].text(0.5, 0.5, 'Нет данных для Duty', horizontalalignment='center', verticalalignment='center')


        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = BoostSimWidget()
    w.show()
    sys.exit(app.exec())
