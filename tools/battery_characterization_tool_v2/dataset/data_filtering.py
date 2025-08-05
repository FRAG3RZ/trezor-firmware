from collections import defaultdict
import copy
from InquirerPy import inquirer

# ===========Filtering section===========


def get_unique_keys(dataset):
    batteries, modes, phases, temps = set(), set(), set(), set()

    for battery in dataset:
        batteries.add(battery)
        for timestamp in dataset[battery]:
            for mode in dataset[battery][timestamp]:
                modes.add(mode)
                for phase in dataset[battery][timestamp][mode]:
                    phases.add(phase)
                    for temp in dataset[battery][timestamp][mode][phase]:
                        temps.add(temp)

    return {
        "battery": sorted(batteries),
        "mode": sorted(modes),
        "phase": sorted(phases),
        "temp": sorted(temps),
    }


# If filtered_list is None or empty, do not filter (allow all)
def filter_dataset_constants(
    dataset,
    filtered_batteries=None,
    filtered_modes=None,
    filtered_phases=None,
    filtered_temps=None,
):

    def is_filtered(value, filtered_list):
        if not filtered_list:  # empty or None -> no filtering
            return True
        return value in filtered_list

    filtered = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    for b in dataset:
        if not is_filtered(b, filtered_batteries):
            continue
        for t in sorted(dataset[b]):  # All timestamps always included
            for m in dataset[b][t]:
                if not is_filtered(m, filtered_modes):
                    continue
                for p in dataset[b][t][m]:
                    if not is_filtered(p, filtered_phases):
                        continue
                    for temp in dataset[b][t][m][p]:
                        if not is_filtered(temp, filtered_temps):
                            continue
                        filtered[b][t][m][p][temp] = dataset[b][t][m][p][temp]
    return filtered


def filter_dataset_interactive(dataset):
    keys = get_unique_keys(dataset)

    def ask_with_all(label, options):
        choices = ["All"] + options
        selected = inquirer.checkbox(f"Select {label} to KEEP:", choices).execute()
        if "All" in selected:
            return options  # return all options
        return selected

    selected_batteries = ask_with_all("batteries", keys["battery"])
    selected_modes = ask_with_all("modes", keys["mode"])
    selected_phases = ask_with_all("phases", keys["phase"])
    selected_temps = ask_with_all("temps", keys["temp"])

    return filter_dataset_constants(
        dataset,
        filtered_batteries=selected_batteries,
        filtered_modes=selected_modes,
        filtered_phases=selected_phases,
        filtered_temps=selected_temps,
    )


# Helper function to count entries in the dataset
def count_entries(dataset):
    count = 0
    for b in dataset:
        for t in dataset[b]:
            for m in dataset[b][t]:
                for p in dataset[b][t][m]:
                    count += len(dataset[b][t][m][p])
    return count


# print whole dictionary
def print_nested_dict_structure(d, indent=0, path=None):
    """
    Recursively prints the structure and keys of a deeply nested dictionary.

    Args:
        d (dict): The nested dictionary.
        indent (int): The current indentation level (used in recursion).
        path (list): The accumulated key path (used to show where in the structure we are).
    """
    if path is None:
        path = []

    if not isinstance(d, dict):
        print("  " * indent + f"{' > '.join(path)}: {type(d).__name__} → {str(d)[:80]}")
        return

    for k, v in d.items():
        current_path = path + [str(k)]
        if isinstance(v, dict):
            print("  " * indent + f"{' > '.join(current_path)} (dict)")
            print_nested_dict_structure(v, indent + 1, current_path)
        else:
            print(
                "  " * indent
                + f"{' > '.join(current_path)}: {type(v).__name__} → {str(v)[:80]}"
            )


# ==========Graphing utility functions==========


def low_pass_ma_filter(data_vector, filter_length):

    filtered_vector = copy.copy(data_vector)
    #
    for i, sample in enumerate(data_vector):
        if i < filter_length:
            filtered_vector[i] = sum(data_vector[:i] / (filter_length))
        else:
            filtered_vector[i] = sum(
                data_vector[i - filter_length : i] / (filter_length)
            )

    return filtered_vector
