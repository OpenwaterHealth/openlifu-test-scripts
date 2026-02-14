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

import openlifu
from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.db import Database
from openlifu.geo import Point
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Solution
from prodreqs_base_class import TestSonicationDurationBase, parse_arguments, NUM_MODULES

class transmitter_heating_placeholder(TestSonicationDurationBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.logger = self._setup_logging()

    def _select_frequency(self) -> None:
        """Select TX frequency in kHz (100–500), CLI override or interactive."""
        # CLI override
        if self.args.frequency is not None:
            self.frequency_khz = self.args.frequency
            return

        while True:
            choice = input("Enter TX frequency in kHz (100–500): ").strip()
            try:
                freq = int(choice)
            except ValueError:
                self.logger.info("Invalid input. Enter an integer value.")
                continue

            if 100 <= freq <= 500:
                self.frequency_khz = freq
                return

            self.logger.info("Frequency must be between 100 and 500 kHz.")

    def _select_num_modules(self) -> None:
        """Interactively select number of modules."""
        # CLI override
        if self.args.num_modules is not None:
            self.num_modules = self.args.num_modules
            return

        while True:
            choice = input(f"Select number of modules {list(NUM_MODULES)}: ")
            if choice.isdigit() and int(choice) in NUM_MODULES:
                self.num_modules = int(choice)
                break
            self.logger.info("Invalid selection. Please try again.")

def main() -> None:
    """Main entry point for the script."""

    args = parse_arguments()
    temp_and_voltage_stability_test = transmitter_heating_placeholder(args)

    try:
        temp_and_voltage_stability_test.run()
    except KeyboardInterrupt:
        temp_and_voltage_stability_test.logger.warning("temp_and_voltage_stability_test aborted by user KeyboardInterrupt. Shutting down...")
        temp_and_voltage_stability_test.shutdown_event.set()
        temp_and_voltage_stability_test.stop_logging = True
        time.sleep(0.5)
        with contextlib.suppress(Exception):
            temp_and_voltage_stability_test.print_test_summary()
            temp_and_voltage_stability_test.turn_off_console_and_tx()
        with contextlib.suppress(Exception):
            temp_and_voltage_stability_test.cleanup_interface()
        sys.exit(0)
    except Exception as e:
        temp_and_voltage_stability_test.logger.error(f"\nFatal error: {e}")
        with contextlib.suppress(Exception):
            temp_and_voltage_stability_test.print_test_summary()
        with contextlib.suppress(Exception):
            temp_and_voltage_stability_test.turn_off_console_and_tx()
        with contextlib.suppress(Exception):
            temp_and_voltage_stability_test.cleanup_interface()
        sys.exit(1)

if __name__ == "__main__":
    main()