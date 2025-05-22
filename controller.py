# File: controller.py
import numpy as np


class SlewRatePI:
    def __init__(
        self,
        Kp,
        Ki,
        T_sample,
        slew_rate_limit=1000.0,
        acceleration=50000.0,  # V/s^2, how quickly we ramp up/down the slew rate
        duty_min=0.01,
        duty_max_limit=0.99,
        ff_enabled=True,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.T_sample = T_sample  # Controller sampling time
        self.duty_min_internal = duty_min
        self.duty_max_internal = duty_max_limit
        self.ff_enabled = ff_enabled
        self.integral_error = 0.0
        self.last_calculated_duty = 0.0

        self.max_slew_rate = abs(
            slew_rate_limit
        )  # V/s, maximum rate of change of current_target
        self.acceleration = abs(acceleration)  # V/s^2

        self.current_target = 0.0  # The PI controller's immediate target voltage
        self.current_slew = 0.0  # The current rate of change of current_target (V/s)
        self.last_vout = 0.0

        self.final_Vref = 0.0  # Stores the ultimate Vref from the GUI
        self.is_ramping = (
            False  # Flag to indicate if we are actively ramping current_target
        )

    def reset(self, initial_target_voltage=0.0, initial_last_vout_voltage=0.0):
        self.integral_error = 0.0
        self.last_calculated_duty = 0.0
        self.current_target = initial_target_voltage
        self.current_slew = 0.0  # Start with zero slew
        self.last_vout = initial_last_vout_voltage
        self.is_ramping = False
        # self.final_Vref should be set externally before first calculate_duty if different from initial_target

    def update_coeffs(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki

    def set_duty_limits(self, duty_min, duty_max):
        self.duty_min_internal = duty_min
        self.duty_max_internal = duty_max

    def set_slew_parameters(self, slew_rate_limit, acceleration_limit):
        self.max_slew_rate = abs(slew_rate_limit)
        self.acceleration = abs(acceleration_limit)
        # If parameters change, we might need to re-evaluate the ramp if one is active
        # For simplicity, assume this is called when not actively in a complex ramp or on reset.

    def _start_new_ramp_if_needed(self, Vref_final_gui):
        # If the final Vref from GUI changes, or if we are not ramping and not at target
        if abs(Vref_final_gui - self.final_Vref) > 1e-6 or (
            not self.is_ramping and abs(self.current_target - Vref_final_gui) > 1e-6
        ):
            self.final_Vref = Vref_final_gui
            self.is_ramping = True  # Start a new ramp sequence

    def calculate_duty(
        self, Vref_final_gui, Vc_feedback, Vin, duty_applied_in_prev_step
    ):
        self._start_new_ramp_if_needed(Vref_final_gui)

        if (
            not self.is_ramping
        ):  # If not ramping, current_target should be stable at final_Vref
            self.current_target = self.final_Vref
            self.current_slew = 0.0
        else:
            # S-Curve Motion Profile Logic
            # Direction: 1 for increasing, -1 for decreasing
            direction = np.sign(self.final_Vref - self.current_target)
            if direction == 0:  # Already at target
                self.is_ramping = False
                self.current_slew = 0.0
                self.current_target = self.final_Vref
            else:
                # Calculate distance to stop from current_slew at current_acceleration
                # v_final^2 = v_initial^2 + 2*a*d  => d = (v_final^2 - v_initial^2) / (2*a)
                # Here v_final = 0 (stopping)
                dist_to_stop = (self.current_slew**2) / (2 * self.acceleration)

                remaining_dist = abs(self.final_Vref - self.current_target)

                if remaining_dist <= dist_to_stop + 1e-6:  # Added tolerance
                    # ---- Deceleration Phase ----
                    # Reduce slew rate
                    self.current_slew -= direction * self.acceleration * self.T_sample
                    # Ensure slew doesn't overshoot zero or reverse if we are very close
                    if (
                        np.sign(self.current_slew) != direction
                        and abs(self.current_slew) > 1e-3
                    ):  # Overshot zero slew
                        self.current_slew = 0  # Force stop if very close
                else:
                    # ---- Acceleration or Constant Slew Phase ----
                    self.current_slew += direction * self.acceleration * self.T_sample
                    # Clamp slew to max_slew_rate
                    if abs(self.current_slew) > self.max_slew_rate:
                        self.current_slew = direction * self.max_slew_rate

                # Update current_target based on current_slew
                # Clamp slew in case of very small T_sample leading to overcorrection near target
                if (
                    abs(self.current_slew * self.T_sample) > remaining_dist
                    and remaining_dist < 0.1
                ):  # If step would overshoot and close
                    self.current_target = self.final_Vref
                    self.current_slew = 0.0
                    self.is_ramping = False
                else:
                    self.current_target += self.current_slew * self.T_sample

                # Check if we've reached/passed the target (especially after deceleration)
                new_direction = np.sign(self.final_Vref - self.current_target)
                if (
                    new_direction != direction and direction != 0
                ):  # Passed target or reached it
                    self.current_target = self.final_Vref
                    self.current_slew = 0.0
                    self.is_ramping = False
                elif abs(self.final_Vref - self.current_target) < 1e-5:  # Close enough
                    self.current_target = self.final_Vref
                    self.current_slew = 0.0
                    self.is_ramping = False

        # --- Error calculation and PI (same as before) ---
        error = self.current_target - Vc_feedback
        self.last_vout = Vc_feedback

        can_integrate = True
        if (duty_applied_in_prev_step >= self.duty_max_internal and error > 0) or (
            duty_applied_in_prev_step <= self.duty_min_internal and error < 0
        ):
            can_integrate = False

        if can_integrate:
            self.integral_error += error * self.T_sample

        pi_correction = self.Kp * error + self.Ki * self.integral_error

        calculated_duty = 0.0
        if self.ff_enabled:
            duty_ff = 0.0
            if self.current_target > Vin and self.current_target > 0:
                duty_ff = (self.current_target - Vin) / self.current_target
            duty_ff = max(
                self.duty_min_internal, min(duty_ff, min(0.90, self.duty_max_internal))
            )
            calculated_duty = duty_ff + pi_correction
        else:
            calculated_duty = pi_correction

        self.last_calculated_duty = max(
            self.duty_min_internal, min(calculated_duty, self.duty_max_internal)
        )

        return self.last_calculated_duty, self.current_target
