#!/usr/bin/env python3
import logging
from dut.dut import Dut  # change to actual import path

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    known_cpu_id = "1C0017000950325557323120"
    #second_known_cpu_id = "12002E000A50325557323120"

    dut = Dut(
                    name="test_name",
                    cpu_id=known_cpu_id,
                    #usb_port=d["usb_port"],
                    relay_port=1,
                    relay_ctl=1,
                    verbose=True,
                )
    print(f"{known_cpu_id} hash is: {dut.cpu_id_hash}")
    """
    dut2 = Dut(
                    name="test_name_2",
                    cpu_id=second_known_cpu_id,
                    #usb_port=d["usb_port"],
                    relay_port=1,
                    relay_ctl=1,
                    verbose=True,
                )
    
    print(f"{second_known_cpu_id} hash is: {dut.cpu_id_hash}")
    
    # Create a Dut instance without running __init__
    dummy_dut = Dut.__new__(Dut)

    dummy_dut.verbose = False

    dummy_dut.vcp = None 

    port, detected_cpu_id = dummy_dut.find_usb_port(cpu_id_expected=known_cpu_id)
    if port:
        print(f"✅ Found DUT at {port} with CPU ID: {detected_cpu_id}")
    else:
        print("❌ No matching DUT found")

    """

