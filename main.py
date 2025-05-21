import numpy as np
import matplotlib.pyplot as plt
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

circuit = Circuit('Boost Converter')

# Parameters
Vin = 5@u_V
L = 100@u_uH
C = 100@u_uF
Rload = 10@u_Ohm
freq = 50@u_kHz
duty_cycle = 0.6

# Sources
circuit.V('input', 'vin', circuit.gnd, Vin)

# Inductor
circuit.L('1', 'vin', 'n1', L)

# Switch (Voltage controlled switch, controlled by Vgate)
circuit.S('1', 'n1', 'gnd', 'gate', circuit.gnd, model='SW')
circuit.model('SW', 'SW', Ron=1@u_mOhm, Roff=1@u_MOhm, Vt=2.5, Vh=0.1)

# Diode
circuit.D('1', 'n1', 'n2', model='D')
circuit.model('D', 'D', IS=1e-15, N=1)

# Capacitor and Load
circuit.C('1', 'n2', circuit.gnd, C)
circuit.R('load', 'n2', circuit.gnd, Rload)

# Gate drive (PWM)
period = float(1/freq)
pulse_width = period * duty_cycle
circuit.PulseVoltageSource('gate', 'gate', circuit.gnd,
                           initial_value=0@u_V, pulsed_value=5@u_V,
                           pulse_width=pulse_width,
                           period=period)

# Simulation
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.transient(step_time=0.1@u_us, end_time=2@u_ms)

# Plot output voltage
plt.figure(figsize=(10, 5))
plt.plot(analysis.time * 1e3, analysis['n2'])
plt.title('Boost Converter Output Voltage')
plt.xlabel('Time [ms]')
plt.ylabel('Vout [V]')
plt.grid(True)
plt.show()

