# File: benchmark_startup.py

import argparse
import time

import numpy as np
from alive_progress import alive_bar

from controller import (
    SlewRatePI,
)  # Assuming controller.py is in the same directory or PYTHONPATH
from spice_simulation import (
    run_boost_sim,
)  # Assuming spice_simulation.py is in the same directory

# --- Default Simulation Parameters ---
DEFAULT_VIN = 5.0  # V
DEFAULT_VREF_FINAL = 12.0  # V
DEFAULT_L_VAL = 10e-6  # H
DEFAULT_C_VAL = 100e-6  # F
DEFAULT_R_LOAD = 20.0  # Ohms
DEFAULT_FREQ = 200 * 1000  # Hz (Note: 500kHz will be much slower)

# DEFAULT_KP = 0.005
DEFAULT_KP = 0.02
# DEFAULT_KI = 0.05
DEFAULT_KI = 20.0
DEFAULT_MAX_DUTY_PI = (
    0.90  # Not directly used by controller's core logic but by gain ramp limits
)
DEFAULT_SLEW_RATE = 1000000  # V/s
DEFAULT_ACCELERATION = 3000000  # V/s^2
DEFAULT_GAIN_RAMP = 0

DEFAULT_DELTA_V_THRESHOLD = (
    0.1  # V (e.g., considered settled if within +/- 0.1V of Vref)
)
DEFAULT_SETTLING_CYCLES = 50  # Number of consecutive cycles Vout must be within delta_V
MAX_SIM_CYCLES = 200000  # Safety break for very long simulations
N_POINTS_PER_CYCLE = 10


def run_startup_benchmark(
    vin,
    vref_final,
    l_val,
    c_val,
    r_load,
    freq,
    kp,
    ki,
    slew_rate,
    acceleration,
    delta_v_threshold,
    settling_cycles_required,
    max_total_cycles=MAX_SIM_CYCLES,
    initial_duty=0.01,  # Small initial duty for the very first SPICE cycle
):
    T_switching = 1.0 / freq
    step_time_calc = T_switching / N_POINTS_PER_CYCLE
    print(f"\n--- Running Startup Benchmark ---")
    print(f"  Vin: {vin:.2f}V, Vref: {vref_final:.2f}V, Freq: {freq/1e3:.1f}kHz")
    print(f"  L: {l_val*1e6:.1f}uH, C: {c_val*1e6:.1f}uF, R: {r_load:.1f}Ohm")
    print(
        f"  Kp: {kp:.4f}, Ki: {ki:.3f}, Slew: {slew_rate:.0f}V/s, Accel: {acceleration:.0f}V/s^2"
    )
    print(
        f"  Settle Condition: Vout within +/-{delta_v_threshold:.2f}V of Vref for {settling_cycles_required} cycles."
    )
    print(f"  Max simulation cycles: {max_total_cycles}")
    print(
        f"  SPICE step time: {step_time_calc*1e9:.0f} ns ({int(T_switching/step_time_calc)} points/cycle)\n"
    )

    # Initialize Controller
    controller = SlewRatePI(
        Kp=kp,
        Ki=ki,
        T_sample=T_switching,
        slew_rate_limit=slew_rate,
        acceleration=acceleration,
        duty_max_limit=DEFAULT_MAX_DUTY_PI,  # Controller's internal Dmax limit
    )
    controller.reset(initial_target_voltage=vin, initial_last_vout_voltage=vin)
    # controller.final_Vref = vref_final # Will be set by calculate_duty

    # Simulation State
    current_Vc0 = vin
    current_Il0 = 0.0  # Assume starting from no load current
    last_applied_duty = initial_duty

    time_elapsed_sim = 0.0
    cycles_in_settled_state = 0

    start_real_time = time.perf_counter()

    # Gain ramp parameters (optional, could be arguments if needed)
    # For a pure startup test, gain ramp might be disabled or very short.
    gain_ramp_cycles = DEFAULT_GAIN_RAMP  # e.g., 50 if you want to include it
    Kp_init = 0.05 * kp
    Ki_init = 0.05 * ki
    Max_duty_init_pi = 0.05 * DEFAULT_MAX_DUTY_PI + 0.01

    with alive_bar(max_total_cycles, title="Simulating Startup") as bar:
        for cycle_num in range(max_total_cycles):
            Vc_feedback = current_Vc0

            # Apply gain scheduling (Kp, Ki, Dmax_PI ramp) if enabled
            if gain_ramp_cycles > 0 and cycle_num < gain_ramp_cycles:
                ramp_factor = cycle_num / max(gain_ramp_cycles, 1)
                Kp_current = Kp_init + (kp - Kp_init) * ramp_factor
                Ki_current = Ki_init + (ki - Ki_init) * ramp_factor
                Max_duty_current_pi = (
                    Max_duty_init_pi
                    + (DEFAULT_MAX_DUTY_PI - Max_duty_init_pi) * ramp_factor
                )
            else:
                Kp_current = kp
                Ki_current = ki
                Max_duty_current_pi = DEFAULT_MAX_DUTY_PI

            controller.update_coeffs(Kp_current, Ki_current)
            controller.set_duty_limits(0.01, Max_duty_current_pi)

            # Calculate duty cycle
            # The first cycle might use a pre-defined initial_duty for SPICE,
            # but controller starts its logic immediately.
            if cycle_num == 0:
                current_duty_to_apply = initial_duty
                # Call calculate_duty to update controller's internal target for the first step
                _, _ = controller.calculate_duty(
                    vref_final, Vc_feedback, vin, last_applied_duty
                )
            else:
                current_duty_to_apply, _ = controller.calculate_duty(
                    vref_final, Vc_feedback, vin, last_applied_duty
                )

            last_applied_duty = current_duty_to_apply

            # Run SPICE for one cycle
            try:
                analysis = run_boost_sim(
                    vin,
                    l_val,
                    c_val,
                    r_load,
                    freq,
                    current_duty_to_apply,
                    step_time_calc,
                    0,
                    T_switching,  # t_start=0, t_end=T_switching
                    current_Il0,
                    current_Vc0,
                )
            except RuntimeError as e:
                print(f"\nSPICE simulation failed at cycle {cycle_num}: {e}")
                return None, None  # Indicate failure

            # Extract results
            if len(analysis.time) < 2:
                print(
                    f"\nSPICE simulation returned too few points at cycle {cycle_num}."
                )
                return None, None  # Indicate failure

            current_Vc0 = float(analysis.nodes["n2"][-1])
            current_Il0 = float(analysis.branches["vjl_sense"][-1])
            time_elapsed_sim += T_switching

            # Check settling condition
            if abs(current_Vc0 - vref_final) <= delta_v_threshold:
                cycles_in_settled_state += 1
            else:
                cycles_in_settled_state = 0  # Reset counter if out of bounds

            bar.text(
                f"Vout: {current_Vc0:.3f}V, Target: {controller.current_target:.3f}V, Duty: {current_duty_to_apply*100:.1f}%"
            )
            bar()  # Increment progress bar

            if cycles_in_settled_state >= settling_cycles_required:
                end_real_time = time.perf_counter()
                print(f"\n--- Benchmark Complete ---")
                print(
                    f"  Settled to Vref ({vref_final:.2f}V +/- {delta_v_threshold:.2f}V) after {cycle_num + 1} cycles."
                )
                print(f"  Simulated time to settle: {time_elapsed_sim * 1000:.3f} ms.")
                print(
                    f"  Actual wall clock time: {end_real_time - start_real_time:.3f} seconds."
                )
                return time_elapsed_sim, cycle_num + 1

        # If loop finishes, max_total_cycles was reached before settling
        end_real_time = time.perf_counter()
        print(f"\n--- Benchmark Incomplete ---")
        print(
            f"  Max cycles ({max_total_cycles}) reached before settling condition met."
        )
        print(
            f"  Last Vout: {current_Vc0:.3f}V at simulated time {time_elapsed_sim * 1000:.3f} ms."
        )
        print(
            f"  Actual wall clock time: {end_real_time - start_real_time:.3f} seconds."
        )
        return None, max_total_cycles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark DC-DC converter startup time."
    )
    parser.add_argument(
        "--vin", type=float, default=DEFAULT_VIN, help="Input voltage (V)"
    )
    parser.add_argument(
        "--vref",
        type=float,
        default=DEFAULT_VREF_FINAL,
        help="Target output voltage (V)",
    )
    parser.add_argument("--l", type=float, default=DEFAULT_L_VAL, help="Inductance (H)")
    parser.add_argument(
        "--c", type=float, default=DEFAULT_C_VAL, help="Capacitance (F)"
    )
    parser.add_argument(
        "--r", type=float, default=DEFAULT_R_LOAD, help="Load resistance (Ohms)"
    )
    parser.add_argument(
        "--freq", type=float, default=DEFAULT_FREQ, help="Switching frequency (Hz)"
    )

    parser.add_argument(
        "--kp", type=float, default=DEFAULT_KP, help="Proportional gain"
    )
    parser.add_argument("--ki", type=float, default=DEFAULT_KI, help="Integral gain")
    parser.add_argument(
        "--slew", type=float, default=DEFAULT_SLEW_RATE, help="Vref slew rate (V/s)"
    )
    parser.add_argument(
        "--accel",
        type=float,
        default=DEFAULT_ACCELERATION,
        help="Vref acceleration (V/s^2)",
    )

    parser.add_argument(
        "--delta_v",
        type=float,
        default=DEFAULT_DELTA_V_THRESHOLD,
        help="Voltage tolerance for settling (V)",
    )
    parser.add_argument(
        "--settle_cycles",
        type=int,
        default=DEFAULT_SETTLING_CYCLES,
        help="Consecutive cycles within delta_v to be considered settled",
    )
    parser.add_argument(
        "--max_cycles",
        type=int,
        default=MAX_SIM_CYCLES,
        help="Maximum simulation cycles to run",
    )
    parser.add_argument(
        "--initial_duty",
        type=float,
        default=0.01,
        help="Initial duty cycle for the first SPICE run",
    )

    args = parser.parse_args()

    run_startup_benchmark(
        vin=args.vin,
        vref_final=args.vref,
        l_val=args.l,
        c_val=args.c,
        r_load=args.r,
        freq=args.freq,
        kp=args.kp,
        ki=args.ki,
        slew_rate=args.slew,
        acceleration=args.accel,
        delta_v_threshold=args.delta_v,
        settling_cycles_required=args.settle_cycles,
        max_total_cycles=args.max_cycles,
        initial_duty=args.initial_duty,
    )
