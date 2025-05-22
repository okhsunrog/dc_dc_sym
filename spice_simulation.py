# File: spice_simulation.py

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from PySpice import Simulator


def run_boost_sim(
    Vin_val=5,
    L_val=100e-6,
    C_val=100e-6,
    Rload_val=10,
    freq_val=50e3,
    duty_cycle_val=0.6,
    step_time_val=0.1e-6,
    t_start_val=0,
    t_end_val=2e-3,
    Il0_val=0,
    Vc0_val=0,
):
    circuit = Circuit(
        f"Boost Converter XSPICE (Duty={duty_cycle_val:.2f} Il0={Il0_val:.2e} Vc0={Vc0_val:.2e})"
    )

    circuit.V("input", "vin", circuit.gnd, Vin_val @ u_V)
    node_l_sense_out = "n_for_l_sense"
    circuit.VoltageSource("Jl_sense", "vin", node_l_sense_out, 0 @ u_V)

    l_spice_float = float(L_val)
    il0_spice_float = float(Il0_val)

    circuit.raw_spice = f""".model inductor_ic_model inductoric L={l_spice_float:.7e} IC={il0_spice_float:.7e}
A_L1 {node_l_sense_out} n1 inductor_ic_model"""

    circuit.S("1", "n1", circuit.gnd, "gate", circuit.gnd, model="SW")
    circuit.model("SW", "SW", Ron=1 @ u_mOhm, Roff=1 @ u_MOhm, Vt=2.5, Vh=0.1)
    circuit.D("1", "n1", "n2", model="D")
    circuit.model("D", "D", IS=1e-15, N=1)  # Using a more typical Is value
    circuit.C("1", "n2", circuit.gnd, C_val @ u_F, ic=Vc0_val @ u_V)
    circuit.R("load", "n2", circuit.gnd, Rload_val @ u_Ohm)

    period = 1 / freq_val
    # Ensure duty cycle is within SPICE-safe limits (e.g., not exactly 0 or 1)
    safe_duty_cycle = max(0.001, min(float(duty_cycle_val), 0.999))
    pulse_width = period * safe_duty_cycle
    # Rise/fall times for the pulse
    tr_tf = (
        period / 1000 if period / 1000 > 1e-9 else 1e-9
    )  # e.g. 0.1% of period, min 1ns

    circuit.PulseVoltageSource(
        "gate_drive",
        "gate",
        circuit.gnd,
        initial_value=0 @ u_V,
        pulsed_value=5 @ u_V,  # Gate drive voltage
        pulse_width=pulse_width @ u_s,
        period=period @ u_s,
        delay_time=0 @ u_s,  # No delay for this cycle-by-cycle sim
        rise_time=tr_tf @ u_s,
        fall_time=tr_tf @ u_s,
    )

    try:
        # Prefer ngspice-shared for performance
        actual_simulator_object = Simulator.factory(simulator="ngspice-shared")
    except Exception:
        try:
            actual_simulator_object = Simulator.factory(simulator="ngspice-subprocess")
        except Exception as e_subprocess:
            # Handle case where no ngspice is found or other error
            print(f"SPICE simulator creation error: {e_subprocess}")
            raise RuntimeError(
                f"SPICE simulator creation error: {e_subprocess}"
            ) from e_subprocess

    simulation_instance = actual_simulator_object.simulation(
        circuit, temperature=25, nominal_temperature=25  # Celsius  # Celsius
    )

    # Set SPICE transient analysis options for potentially faster simulation
    # gmin and rshunt can help with convergence for some circuits.
    # These are just examples; tuning them requires care.
    # simulation_instance.options(gmin=1e-12, reltol=1e-3, abstol=1e-6, vntol=1e-6)
    # Or add to raw_spice: .options gmin=1e-12 reltol=1e-3 abstol=1e-6 vntol=1e-6 method=gear

    analysis = simulation_instance.transient(
        step_time=step_time_val @ u_s,  # Max output time step
        end_time=t_end_val @ u_s,
        start_time=t_start_val @ u_s,
        use_initial_condition=True,  # Use UIC keyword for .tran
    )
    return analysis
