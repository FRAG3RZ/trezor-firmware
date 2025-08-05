


from utils.data_loader import *
from pathlib import Path
import argparse
import argcomplete

parser = argparse.ArgumentParser(
                    prog='analyze_charging_profile')


group = parser.add_mutually_exclusive_group(required=True)

group.add_argument('-f', '--input_file', help="Charging profile data file")
group.add_argument('-d', '--input_dir', help="Charging profile data directory")

argcomplete.autocomplete(parser)

def main():

    args = parser.parse_args()

    if args.input_file:

        input_file_path = Path(args.input_file)

        data = load_measured_data(input_file_path)
        charging, discharging = split_profile_phases(data)

        test_characteristics = input_file_path.stem.split(".")

        test_type = test_characteristics[1]
        test_temperature = test_characteristics[2]

        for i, ch_profile in enumerate(charging):

            if(i > 0):
                file_name = input_file_path.parent / f"charge.{test_type}.{test_temperature}_cut_{i}.csv"
            else:
                file_name = input_file_path.parent / f"charge.{test_type}.{test_temperature}.csv"

            export_profile_data(data=ch_profile, output_file_path=file_name)

        for i, dis_profile in enumerate(discharging):

            if(i > 0):
                file_name = input_file_path.parent / f"discharge.{test_type}.{test_temperature}_cut_{i}.csv"
            else:
                file_name = input_file_path.parent / f"discharge.{test_type}.{test_temperature}.csv"

            export_profile_data(data=dis_profile, output_file_path=file_name)

    elif args.input_dir:

        input_dir_path = Path(args.input_dir)

        linear_dir = input_dir_path / "linear"
        switching_dir = input_dir_path / "switching"

        linear_dir.mkdir(parents=True, exist_ok=True)
        switching_dir.mkdir(parents=True, exist_ok=True)

        for input_file_path in input_dir_path.glob("charge_discharge.*.csv"):

            data = load_measured_data(input_file_path)
            charging, discharging = split_profile_phases(data)

            test_characteristics = input_file_path.stem.split(".")

            test_type = test_characteristics[1]
            test_temperature = test_characteristics[2]

            for i, ch_profile in enumerate(charging):

                if(i > 0):
                    file_name = f"charge.{test_type}.{test_temperature}_cut_{i}.csv"
                else:
                    file_name = f"charge.{test_type}.{test_temperature}.csv"

                if test_type == "linear":
                    file_name = linear_dir / file_name
                elif test_type == "switching":
                    file_name = switching_dir / file_name
                else:
                    print("Parsing issue, skiping file {file_name}")
                    continue

                export_profile_data(data=ch_profile, output_file_path=file_name)

            for i, ch_profile in enumerate(discharging):

                    if(i > 0):
                        file_name = f"discharge.{test_type}.{test_temperature}_cut_{i}.csv"
                    else:
                        file_name = f"discharge.{test_type}.{test_temperature}.csv"

                    if test_type == "linear":
                        file_name = linear_dir / file_name
                    elif test_type == "switching":
                        file_name = switching_dir / file_name
                    else:
                        print("Parsing issue, skiping file {file_name}")
                        continue

                    export_profile_data(data=ch_profile, output_file_path=file_name)

if __name__ == "__main__":
    main()





