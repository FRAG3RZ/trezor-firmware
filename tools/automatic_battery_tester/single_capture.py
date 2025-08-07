import sys
import time
from pathlib import Path
import argparse

from dut import Dut
from hardware_ctl.gdm8351 import GDM8351
from serial.tools import list_ports

output_directory = Path("single_capture_test_results")

test_description = ""
temp_description = ""

external_thermocouple_sensor = False # Set to True if using an external thermocouple sensor


"""
This script will connect to a signle DUT over VCP port and will run a continous
log of the power manager data (continously calls pm-report command) into the
log file. User can also select to log the temepertature readings from an
external thermocouple sensor connected to the GDM8351 multimeter.
"""


def main():

    global output_directory, test_description, temp_description, external_thermocouple_sensor

    print("**********************************************************")
    print("  DUT port selection ")
    print("**********************************************************")

    ports = list_ports.comports()

    available_ports = {}
    port_count = 0
    print("Available VCP ports:")
    for port in ports:
        if "usb" in port.device:
            port_count += 1
            available_ports[port_count] = port.device
            print(f"    [{port_count}]: {port.device} - {port.description}")

    if port_count == 0:
        print("No device conneceted. Exiting.")
        return

    dut_port_selection = input("Select VCP port number (or Q to quit the selection): ")

    if dut_port_selection.lower() == "q":
        print("Exiting script.")
        sys.exit(0)

    selected_port = None
    for port_id, port_name in available_ports.items():
        if int(dut_port_selection) == port_id:
            selected_port = port_name
            break

    try:
        dut = Dut(name="Trezor", usb_port=selected_port)
    except Exception as e:
        print(f"Failed to initialize DUT on port {selected_port}: {e}")
        sys.exit(1)
    # Initialize DUT


    # Creat test time ID
    test_time_id = f"{time.strftime('%y%m%d%H%M')}"

    # Create output data directory
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print("Failed to create output directory:", e)
        sys.exit(1)

    #########################################################################
    # Test setup section
    #########################################################################

    def get_input(prompt, type_=int):
        while True:
            try:
                return type_(input(prompt))
            except ValueError:
                print(f"Please enter a valid {type_.__name__}.")

    parser = argparse.ArgumentParser(description="DUT Configuration")
    #parser.add_argument("--soc", type=int, help="Set SOC limit (0-100)")
    parser.add_argument("--backlight", type=int, help="Set backlight level (0-255)")
    #parser.add_argument("--charging", action="store_true", help="Enable charging")
    parser.add_argument("--ext_temp_sensor", action="store_true", help="External temperature sensor enabled - default: False")
    parser.add_argument("--test_description", type=str, default="non_specified_test",
                        help="Description of the test")
    parser.add_argument("--temp_description", type=str, default="ambient",
                        help="Environmental temperature description")

    args = parser.parse_args()
    """
    # SOC limit
    if args.soc is not None:
        dut.set_soc_limit(args.soc)
    else:
        soc = get_input("Enter SOC limit (0-100): ")
        dut.set_soc_limit(soc)

    # Charging
    if args.charging:
        dut.enable_charging()
    else:
        choice = input("Enable charging? (y/n): ").strip().lower()
        if choice == 'y':
            dut.enable_charging()
    """
    # Temp sensor 
    if args.ext_temp_sensor is not None and args.ext_temp_sensor:
        external_thermocouple_sensor = True

    #Initialize external temperature sensor
    if(external_thermocouple_sensor):
        print("**********************************************************")
        print("  GDM8351 port selection (temp measurement) ")
        print("**********************************************************")

        # Initialize the GDM8351 multimeter
        gdm8351 = GDM8351()

        # Get the device ID to confirm connection
        try:
            device_id = gdm8351.get_id()
            print(f"Connected to device: {device_id}")
        except Exception as e:
            print(f"Error getting device ID: {e}")
            return

        # Configure temperature sensing
        try:
            gdm8351.configure_temperature_sensing(sensor_type="K", junction_temp_deg=29.0)
            print("Temperature sensing configured successfully.")
        except ValueError as ve:
            print(f"Configuration error: {ve}")
        except Exception as e:
            print(f"Error configuring temperature sensing: {e}")

    # Backlight
    if args.backlight is not None:
        dut.set_backlight(args.backlight)
    else:
        backlight = get_input("Enter backlight level (0-255): ")
        dut.set_backlight(backlight)

    dut.set_soc_target(100)  # Set SOC limit to 100% for testing
    dut.enable_charging()

    #########################################################################
    # Main test loop
    #########################################################################
    try:
        print("**********************************************************")
        print("  Test execution started ")
        print("**********************************************************")
        print(f"Test time ID: {test_time_id}")
        print(f"Test description: {args.test_description}")
        print(f"Temperature description: {args.temp_description}")

        while True:

            dut.log_data(
                output_directory=output_directory,
                test_time_id=test_time_id,
                test_scenario="single_capture",
                test_phase=test_description,
                temp=temp_description,
                verbose=True,
            )
            if(external_thermocouple_sensor):
                # Read temperature from GDM8351
                gdm8351.log_temperature(
                    output_directory=output_directory,
                    test_time_id=test_time_id,
                    verbose=True,
                )

            time.sleep(1)

    except KeyboardInterrupt:
        print("Test execution interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"FATAL ERROR during test execution: {e}")
    finally:

        dut.close()
        gdm8351.close()


if __name__ == "__main__":
    main()
