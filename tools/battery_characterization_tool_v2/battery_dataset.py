"""
Battery Dataset Manager

A class to organize and manage battery profile CSV files with hierarchical structure:
battery_id -> timestamp_id -> battery_mode -> mode_phase -> temperature -> data

File naming convention: <battery_id>.<timestamp_id>.<battery_mode>.<mode_phase>.<temperature>.csv
"""

from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Dict, List, Optional, Union, Any
from utils.data_convertor import load_measured_data_new


class BatteryDataset:
    """
    A class to manage battery profile datasets with hierarchical organization.

    Structure: battery_id -> timestamp_id -> battery_mode -> mode_phase -> temperature -> data
    """

    def __init__(self, dataset_path: Union[str, Path], load_data: bool = True):
        """
        Initialize the battery dataset.

        Args:
            dataset_path: Path to directory containing CSV files or glob pattern
            load_data: Whether to load CSV data immediately or just catalog files
        """
        self.dataset_path = Path(dataset_path)
        self.load_data = load_data

        # Create nested defaultdict structure: battery -> timestamp -> mode -> phase -> temp -> data
        self._data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(dict)
                )
            )
        )

        # Store file metadata for each entry
        self._file_metadata = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(dict)
                )
            )
        )

        # Statistics
        self._stats = {
            'total_files': 0,
            'loaded_files': 0,
            'skipped_files': 0,
            'error_files': 0
        }

        # Load the dataset
        self._load_dataset()

    def _parse_filename(self, file_path: Path) -> Optional[Dict[str, str]]:
        """
        Parse filename according to the convention:
        <battery_id>.<timestamp_id>.<battery_mode>.<mode_phase>.<temperature>.csv

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary with parsed components or None if parsing fails
        """
        parts = file_path.stem.split('.')

        if len(parts) < 5:
            return None

        return {
            'battery_id': parts[0],
            'timestamp_id': parts[1],
            'battery_mode': parts[2],
            'mode_phase': parts[3],
            'temperature': parts[4],
            'file_path': file_path,
            'file_name': file_path.name
        }

    def _load_dataset(self):
        """Load all CSV files from the dataset path and organize them."""

        # Handle both directory paths and glob patterns
        if self.dataset_path.is_dir():
            csv_files = list(self.dataset_path.glob("*.csv"))
        else:
            # Assume it's a glob pattern
            csv_files = list(self.dataset_path.parent.glob(self.dataset_path.name))

        print(f"Found {len(csv_files)} CSV files to process...")

        for file_path in csv_files:
            self._stats['total_files'] += 1

            # Parse filename
            file_info = self._parse_filename(file_path)
            if file_info is None:
                print(f"WARNING: Could not parse filename: {file_path.name}")
                self._stats['skipped_files'] += 1
                continue

            battery_id = file_info['battery_id']
            timestamp_id = file_info['timestamp_id']
            battery_mode = file_info['battery_mode']
            mode_phase = file_info['mode_phase']
            temperature = file_info['temperature']

            # Skip files with 'done' phase
            if mode_phase == 'done':
                print(f"SKIPPING: {file_path.name} - phase 'done'")
                self._stats['skipped_files'] += 1
                continue

            # Store file metadata
            self._file_metadata[battery_id][timestamp_id][battery_mode][mode_phase][temperature] = file_info

            # Load data if requested
            if self.load_data:
                try:
                    data = load_measured_data_new(file_path)
                    self._data[battery_id][timestamp_id][battery_mode][mode_phase][temperature] = data
                    self._stats['loaded_files'] += 1
                except Exception as e:
                    print(f"ERROR loading {file_path.name}: {e}")
                    self._stats['error_files'] += 1
            else:
                # Store file path for lazy loading
                self._data[battery_id][timestamp_id][battery_mode][mode_phase][temperature] = file_path
                self._stats['loaded_files'] += 1

        print(f"Dataset loaded: {self._stats}")

    def get_battery_ids(self) -> List[str]:
        """Get list of all battery IDs in the dataset."""
        return sorted(list(self._data.keys()))

    def get_timestamp_ids(self, battery_id: str) -> List[str]:
        """Get list of all timestamp IDs for a given battery."""
        if battery_id not in self._data:
            return []
        return sorted(list(self._data[battery_id].keys()))

    def get_battery_modes(self, battery_id: str, timestamp_id: str) -> List[str]:
        """Get list of all battery modes for a given battery and timestamp."""
        if battery_id not in self._data or timestamp_id not in self._data[battery_id]:
            return []
        return sorted(list(self._data[battery_id][timestamp_id].keys()))

    def get_mode_phases(self, battery_id: str, timestamp_id: str, battery_mode: str) -> List[str]:
        """Get list of all mode phases for a given battery, timestamp, and mode."""
        if (battery_id not in self._data or
            timestamp_id not in self._data[battery_id] or
            battery_mode not in self._data[battery_id][timestamp_id]):
            return []
        return sorted(list(self._data[battery_id][timestamp_id][battery_mode].keys()))

    def get_temperatures(self, battery_id: str, timestamp_id: str, battery_mode: str, mode_phase: str) -> List[str]:
        """Get list of all temperatures for a given battery, timestamp, mode, and phase."""
        if (battery_id not in self._data or
            timestamp_id not in self._data[battery_id] or
            battery_mode not in self._data[battery_id][timestamp_id] or
            mode_phase not in self._data[battery_id][timestamp_id][battery_mode]):
            return []
        return sorted(list(self._data[battery_id][timestamp_id][battery_mode][mode_phase].keys()))

    def get_data(self, battery_id: str, timestamp_id: str, battery_mode: str, mode_phase: str, temperature: str):
        """
        Get data for specific battery profile.

        Returns:
            Loaded data array or None if not found
        """
        try:
            data = self._data[battery_id][timestamp_id][battery_mode][mode_phase][temperature]

            # If lazy loading (data is still a file path), load it now
            if isinstance(data, Path):
                loaded_data = load_measured_data_new(data)
                # Cache the loaded data
                self._data[battery_id][timestamp_id][battery_mode][mode_phase][temperature] = loaded_data
                return loaded_data

            return data
        except KeyError:
            return None

    def get_file_info(self, battery_id: str, timestamp_id: str, battery_mode: str, mode_phase: str, temperature: str) -> Optional[Dict]:
        """Get file metadata for specific battery profile."""
        try:
            return self._file_metadata[battery_id][timestamp_id][battery_mode][mode_phase][temperature]
        except KeyError:
            return None

    def filter(self,
               battery_ids: Optional[List[str]] = None,
               timestamp_ids: Optional[List[str]] = None,
               battery_modes: Optional[List[str]] = None,
               mode_phases: Optional[List[str]] = None,
               temperatures: Optional[List[str]] = None) -> 'BatteryDataset':
        """
        Create a filtered copy of the dataset.

        Args:
            battery_ids: List of battery IDs to include (None = all)
            timestamp_ids: List of timestamp IDs to include (None = all)
            battery_modes: List of battery modes to include (None = all)
            mode_phases: List of mode phases to include (None = all)
            temperatures: List of temperatures to include (None = all)

        Returns:
            New BatteryDataset instance with filtered data
        """
        # Create new instance
        filtered_dataset = BatteryDataset.__new__(BatteryDataset)
        filtered_dataset.dataset_path = self.dataset_path
        filtered_dataset.load_data = self.load_data
        filtered_dataset._data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(dict)
                )
            )
        )
        filtered_dataset._file_metadata = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(dict)
                )
            )
        )
        filtered_dataset._stats = {'total_files': 0, 'loaded_files': 0, 'skipped_files': 0, 'error_files': 0}

        # Apply filters
        for battery_id in self._data:
            if battery_ids and battery_id not in battery_ids:
                continue

            for timestamp_id in self._data[battery_id]:
                if timestamp_ids and timestamp_id not in timestamp_ids:
                    continue

                for battery_mode in self._data[battery_id][timestamp_id]:
                    if battery_modes and battery_mode not in battery_modes:
                        continue

                    for mode_phase in self._data[battery_id][timestamp_id][battery_mode]:
                        if mode_phases and mode_phase not in mode_phases:
                            continue

                        for temperature in self._data[battery_id][timestamp_id][battery_mode][mode_phase]:
                            if temperatures and temperature not in temperatures:
                                continue

                            # Copy data and metadata
                            filtered_dataset._data[battery_id][timestamp_id][battery_mode][mode_phase][temperature] = \
                                self._data[battery_id][timestamp_id][battery_mode][mode_phase][temperature]
                            filtered_dataset._file_metadata[battery_id][timestamp_id][battery_mode][mode_phase][temperature] = \
                                self._file_metadata[battery_id][timestamp_id][battery_mode][mode_phase][temperature]
                            filtered_dataset._stats['loaded_files'] += 1

        return filtered_dataset

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = self._stats.copy()
        stats.update({
            'unique_battery_ids': len(self.get_battery_ids()),
            'total_profiles': sum(
                len(self.get_temperatures(bid, tid, bm, mp))
                for bid in self.get_battery_ids()
                for tid in self.get_timestamp_ids(bid)
                for bm in self.get_battery_modes(bid, tid)
                for mp in self.get_mode_phases(bid, tid, bm)
            )
        })
        return stats

    def print_structure(self, max_depth: int = 5):
        """Print the hierarchical structure of the dataset."""
        print("Dataset Structure:")
        print(f"├── Battery IDs: {self.get_battery_ids()}")

        for battery_id in self.get_battery_ids()[:2]:  # Show first 2 batteries as example
            timestamp_ids = self.get_timestamp_ids(battery_id)
            print(f"│   ├── {battery_id}/")
            print(f"│   │   ├── Timestamps: {timestamp_ids}")

            if max_depth > 2:
                for timestamp_id in timestamp_ids[:1]:  # Show first timestamp as example
                    battery_modes = self.get_battery_modes(battery_id, timestamp_id)
                    print(f"│   │   │   ├── {timestamp_id}/")
                    print(f"│   │   │   │   ├── Modes: {battery_modes}")

                    if max_depth > 3:
                        for battery_mode in battery_modes:
                            mode_phases = self.get_mode_phases(battery_id, timestamp_id, battery_mode)
                            print(f"│   │   │   │   │   ├── {battery_mode}/")
                            print(f"│   │   │   │   │   │   ├── Phases: {mode_phases}")

                            if max_depth > 4:
                                for mode_phase in mode_phases:
                                    temperatures = self.get_temperatures(battery_id, timestamp_id, battery_mode, mode_phase)
                                    print(f"│   │   │   │   │   │   │   ├── {mode_phase}/")
                                    print(f"│   │   │   │   │   │   │   │   └── Temps: {temperatures}")

        if len(self.get_battery_ids()) > 2:
            print(f"│   └── ... and {len(self.get_battery_ids()) - 2} more batteries")

    def __len__(self) -> int:
        """Return total number of profiles in the dataset."""
        return self.get_statistics()['total_profiles']

    def __repr__(self) -> str:
        """String representation of the dataset."""
        stats = self.get_statistics()
        return (f"BatteryDataset(path='{self.dataset_path}', "
                f"batteries={stats['unique_battery_ids']}, "
                f"profiles={stats['total_profiles']})")


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    dataset_path = Path("dataset") / "*.csv"  # or just Path("dataset") for directory

    # Load dataset
    battery_dataset = BatteryDataset(dataset_path)

    # Print statistics
    print(battery_dataset)
    print("\nStatistics:", battery_dataset.get_statistics())

    # Print structure
    battery_dataset.print_structure()

    # Access data
    battery_ids = battery_dataset.get_battery_ids()
    if battery_ids:
        first_battery = battery_ids[0]
        timestamps = battery_dataset.get_timestamp_ids(first_battery)
        if timestamps:
            first_timestamp = timestamps[0]
            modes = battery_dataset.get_battery_modes(first_battery, first_timestamp)
            print(f"\nModes for {first_battery} at {first_timestamp}: {modes}")

    # Filter example
    filtered = battery_dataset.filter(
        battery_modes=['linear', 'switching'],
        mode_phases=['charging', 'discharging']
    )
    print(f"\nFiltered dataset: {filtered}")
