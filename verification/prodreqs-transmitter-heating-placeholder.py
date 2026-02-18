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

class TransmitterHeatingPlaceholder(TestSonicationDurationBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.logger = self._setup_logging()

def main() -> None:
    """Main entry point for the script."""

    args = parse_arguments()
    temp_and_voltage_stability_test = TransmitterHeatingPlaceholder(args)

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