from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from serial.serialutil import SerialException

import numpy as np

from prodreqs_base_class import *
from config import *

# config.py
TEST_VOLTAGES = [65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
MAX_PERCENT_VOLTAGE_DEVIATION = 2.0
SEQUENCE_DURATION_SECONDS = 60

class VoltageAccuracyTest(TestSonicationDurationBase):
    """Data class to hold voltage accuracy test results."""
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.max_voltage_deviation_absolute: float | None = None
        self.max_voltage_deviation_percentage: float | None = None
        self.test_time_elapsed: float | None = None
        self.sequence_duration: float = SEQUENCE_DURATION_SECONDS
        self.bypass_transmitter = True  # For voltage accuracy test, we can bypass TX communication
        self.max_allowed_voltage_deviation_percentage = MAX_PERCENT_VOLTAGE_DEVIATION

    def _select_starting_test_case(self) -> None:
        valid_test_nums = list(range(1, len(TEST_VOLTAGES) + 1))

        # Test case selection
        if self.args.test_case is not None:
            self.starting_test_case = int(self.args.test_case)
        else:
            self.logger.info("\nAvailable Test Cases:")
            for test_id, voltage in enumerate(TEST_VOLTAGES, start=1):
                self.logger.info(f"Test Case {test_id:>2}. {voltage:>2}V")

            while True:
                choice = input(
                    "Press enter to run through all test cases or "
                    "select a test case by number to start at: "
                )
                if choice == "":
                    self.starting_test_case = 1  # Start from beginning
                    break
                if choice.isdigit() and int(choice) in valid_test_nums:
                    self.starting_test_case = int(choice)
                    break
                self.logger.info("Invalid selection. Please try again.")

    def run(self):
        try:
            self._select_starting_test_case()
            self._attach_file_handler()
            self.print_banner()
        except Exception as e:
            self.logger.error("Error during initial selection: %s", e)
            sys.exit(1)

        self.logger.info("Starting automated test sequence from test case %d out of %d total test cases. " % (self.starting_test_case, len(TEST_VOLTAGES)))
        self.start_time = time.time()

        for test_case, test_case_parameters in enumerate((TEST_VOLTAGES)[self.starting_test_case-1:], start=self.starting_test_case):
            self.test_case_num = test_case
            self.test_results[self.test_case_num] = TestCaseResult()
            self.voltage = test_case_parameters
            # self.interval_msec = int(test_case_parameters["PRI_ms"])
            # self.duration_msec = int(test_case_parameters["duty_cycle"] / 100 * self.interval_msec)
            
            self.logger.info(f"Starting test case {self.test_case_num} out of {len(TEST_VOLTAGES)}")
            self.logger.info("Test Case %d: %dV",
                             self.test_case_num, 
                             self.voltage)

            test_case_start_time = 0

            try:
                if not self.hw_simulate:
                    self.connect_device()
                    self.verify_communication()

                    # if test has already run at least once, skip
                    if test_case == self.starting_test_case: 
                        self.get_firmware_versions()
                        # self.enumerate_devices()
                        
                    # self._verify_start_conditions(test_case, test_case_parameters["max_starting_temperature"])
                else:
                    self.logger.info("Hardware simulation enabled; skipping device configuration.")

                # if self.voltage_accuracy_test:
                #     self.sequence_duration = VOLTAGE_ACCURACY_NO_LOAD_TEST_DURATION_SECONDS
                # elif self.test_runthrough:
                #     self.sequence_duration = SHORT_TEST_DURATION_SECONDS
                # elif self.voltage is not None and self.voltage <= LOW_VOLTAGE_VALUE:
                #     self.sequence_duration = LOW_VOLTAGE_VALUE_TEST_DURATION_SECONDS
                # else:
                #     self.sequence_duration = TEST_CASE_DURATION_SECONDS

            
                self.logger.info("Running console voltage accuracy test under no-load conditions...")
                self.interface.hvcontroller.set_voltage(self.voltage)
                self.logger.info("Turning on HV for voltage accuracy test...")
                self.interface.hvcontroller.turn_hv_on()


                # Start monitoring threads
                self.shutdown_event.clear()
                self.sequence_complete_event.clear()
                self.temperature_shutdown_event.clear()
                self.voltage_shutdown_event.clear()

                temp_thread = threading.Thread(
                    target=self.monitor_temperature,
                    name="TemperatureMonitorThread",
                    # daemon=True,
                )
                completion_thread = threading.Thread(
                    target=self.exit_on_time_complete,
                    name="SequenceCompletionThread",
                    # daemon=True,
                )
                voltage_thread = threading.Thread(
                    target=self.monitor_console_voltage,
                    name="ConsoleVoltageMonitorThread",
                    # daemon=True,
                )
                
                voltage_thread.start()
                # temp_thread.start()
                completion_thread.start()

                # Wait for threads or user interrupt
                try:
                    while not self.shutdown_event.is_set():
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    self.logger.warning("Test aborted by user KeyboardInterrupt.")
                    self.test_status = "aborted by user"
                    self.shutdown_event.set()

                # Ensure shutdown event set
                if not self.shutdown_event.is_set():
                    self.logger.warning("A thread exited without setting shutdown event; forcing shutdown.")
                    self.shutdown_event.set()

                # Stop sonication
                if not self.hw_simulate and self.interface is not None:
                    try:
                        if self.interface.hvcontroller.turn_hv_off():
                            self.logger.info("HV stopped successfully.")
                        else:
                            self.logger.error("Failed to stop HV.")
                    except Exception as e:
                        self.logger.error("Error stopping HV: %s", e)

                # Wait for threads to exit gracefully
                # temp_thread.join(timeout=5.0)
                voltage_thread.join(timeout=5.0)
                completion_thread.join(timeout=5.0)

                # Determine final status
                if self.test_status not in ("aborted by user", "error"):
                    if self.sequence_complete_event.is_set():
                        self.test_status = "passed"
                    elif self.temperature_shutdown_event.is_set():
                        self.test_status = "temperature shutdown"
                    elif self.voltage_shutdown_event.is_set():
                        self.test_status = "voltage deviation"
                    else:
                        self.test_status = "error"

            finally:
                # Record test time
                # self.test_results[self.test_case_num].test_time_elapsed = time.time() - test_case_start_time if test_case_start_time else 0
                duration = time.time() - test_case_start_time if test_case_start_time else 0.0
                self.test_results[self.test_case_num].test_time_elapsed = duration

                # Power down and cleanup
                if not self.hw_simulate:
                    with contextlib.suppress(Exception):
                        self.turn_off_console_and_tx()
                    self.cleanup_interface()

                # Final status log
                if self.test_status == "passed":
                    self.logger.info("TEST CASE %d PASSED.", self.test_case_num)
                    self.test_results[self.test_case_num].status = "PASSED"
                elif self.test_status == "temperature shutdown":
                    self.logger.info("TEST CASE %d FAILED.", self.test_case_num)
                    self.test_results[self.test_case_num].status = "FAILED (temperature shutdown)"
                elif self.test_status == "aborted by user":
                    self.logger.info("TEST CASE %d ABORTED by user.", self.test_case_num)
                    self.test_results[self.test_case_num].status = "ABORTED"
                elif self.test_status == "voltage deviation":
                    self.logger.info("TEST CASE %d FAILED.", self.test_case_num)
                    self.test_results[self.test_case_num].status = "FAILED (voltage deviation)"
                elif self.test_status == "error":
                    self.logger.info("TEST CASE %d FAILED due to error.", self.test_case_num)
                    self.test_results[self.test_case_num].status = "FAILED (error)"
                elif self.test_status == "not started":
                    self.logger.info("TEST CASE %d NOT RUN.", self.test_case_num)
                    self.test_results[self.test_case_num].status = "NOT RUN"
                else:
                    self.logger.info(
                        "TEST CASE %d FAILED due to unexpected error.",
                        self.test_case_num,
                    )
                    self.test_results[self.test_case_num] = "FAILED (unexpected error)"
                
                self.logger.info("TEST CASE %d ran for a total of %s.", self.test_case_num, format_duration(duration))
                # self.test_results[self.test_case_num].cooldown_time_elapsed = 0.0

        self.print_test_summary()

    def print_banner(self) -> None:

        self.logger.info("--------------------------------------------------------------------------------")
        self.logger.info(
            "\n\nThis script will automatically cycle through all of the following test cases:\n\n"
            + "\n".join(
                f"Test Case {i:>2}: {tc}V"
                # f"{tc['voltage']:>3}V, "
                # f"Max Starting Temperature: {tc:>3}C"
                for i, tc in enumerate(TEST_VOLTAGES[self.starting_test_case-1:], start=self.starting_test_case)
            )
            + "\n\nThe script will account for cooldown periods as needed between test cases. \n" \
            f"Each test case will run for {self.sequence_duration/60:.2f} minute. \n"
            f"Maximum allowed console voltage deviation is {self.max_allowed_voltage_deviation_percentage:.2f}% or {VOLTAGE_DEVIATION_ABSOLUTE_VALUE_LIMIT:.2f}V, whichever is greater.\n"
            f"Approximate test duration is {self.sequence_duration*len(TEST_VOLTAGES)/60:.2f} minutes.\n"
        )
        self.logger.info("--------------------------------------------------------------------------------\n\n\n")

    def print_test_summary(self) -> None:
        self.logger.info("--------------------------------------------------------------------------------")
        self.logger.info("\n\nTest Case Summary:\n")

        for test_case, test_case_parameters in enumerate(TEST_VOLTAGES[self.starting_test_case - 1:], start=self.starting_test_case):
            r = self.test_results.get(test_case)
            # act_start  = f"{r.starting_temperature:.2f}C" if r and r.starting_temperature is not None else " -   "
            # final      = f"{r.final_temperature:.2f}C" if r and r.final_temperature is not None else " - "
            max_dv     = f"{r.max_voltage_deviation_absolute:.2f}V" if r and r.max_voltage_deviation_absolute is not None else " - "
            max_dv_pct = f"{r.max_voltage_deviation_percentage:.2f}%" if r and r.max_voltage_deviation_percentage is not None else " - "
            # cooldown   = f"~{r.cooldown_time_elapsed}m" if r and r.cooldown_time_elapsed is not None else " - "
            dur        = format_hhmmss(r.test_time_elapsed) if r and r.test_time_elapsed is not None else " - "
            status     = r.status if r and getattr(r, "status", None) else "NOT RUN"
            
            
            self.logger.info(
                f"Test Case {test_case:>2}: "
                f"{test_case_parameters:>2}V, "
                f"Max Allowed Voltage Deviation: {VOLTAGE_DEVIATION_ABSOLUTE_VALUE_LIMIT:>3}V ({VOLTAGE_DEVIATION_PERCENTAGE_LIMIT:>3}%), "
                f"Actual Voltage Deviation: {max_dv:>5} ({max_dv_pct:>5}) "
                f"Duration Run: {dur:>5}  --> "
                f"{status}" + ("\n" if test_case == len(TEST_VOLTAGES) / 2 else "")
            )

        passed_count = sum(1 for r in self.test_results.values() if getattr(r, 'status', None) == "PASSED")

        self.logger.info(
            f"{passed_count} out of {len(TEST_VOLTAGES)-self.starting_test_case+1} test cases passed."
        )

        self.logger.info(f"Script ran for a total of {format_duration(time.time() - self.start_time)}.")

        self.logger.info(
            "\n\nOVERALL RESULT: %s\n",
            "PASSED" if passed_count == len(TEST_VOLTAGES)-self.starting_test_case+1 else "FAILED",
        )

def main() -> None:
    """Main entry point for the script."""
    args = parse_arguments()
    voltage_test = VoltageAccuracyTest(args)

    try:
        voltage_test.run()
    except KeyboardInterrupt:
        voltage_test.logger.warning("Test aborted by user KeyboardInterrupt. Shutting down...")
        voltage_test.shutdown_event.set()
        voltage_test.stop_logging = True
        time.sleep(0.5)
        with contextlib.suppress(Exception):
            voltage_test.print_test_summary()
            voltage_test.turn_off_console_and_tx()
        with contextlib.suppress(Exception):
            voltage_test.cleanup_interface()
        sys.exit(0)
    except Exception as e:
        voltage_test.logger.error(f"\nFatal error: {e}")
        with contextlib.suppress(Exception):
            voltage_test.print_test_summary()
        with contextlib.suppress(Exception):
            voltage_test.turn_off_console_and_tx()
        with contextlib.suppress(Exception):
            voltage_test.cleanup_interface()
        sys.exit(1)


if __name__ == "__main__":
    main()