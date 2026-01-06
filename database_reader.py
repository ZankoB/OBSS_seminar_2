import os
import wfdb
import numpy as np

class database_reader:
    """
    Reader for MIT-BIH Arrhythmia Database using WFDB logic.
    """

    def __init__(self) -> None:
        self.db_dir = "./data"

    def get_data(self, record_num: int) -> dict:
        """
        Loads MIT-BIH record using WFDB and returns signals, metadata, and annotations.

        :param record_num: Record number to load (e.g., 100, 101, ..., 234)
        :return: Dictionary with keys:
                 - "signal": np.ndarray of shape (num_samples, num_channels)
                 - "fs": Sampling frequency
                 - "channels": List of channel names
                 - "units": List of units for each channel
                 - "annotations_samples": np.ndarray of annotation sample indices
                 - "annotations_symbols": List of annotation symbols
                 - "raw_record": Raw WFDB record object
                 - "raw_annotation": Raw WFDB annotation object
        """
        record_str = str(record_num)

        full_path = os.path.join(self.db_dir, record_str)

        # Load header + signals
        record = wfdb.rdrecord(full_path)

        # Load annotations (.atr)
        annotation = wfdb.rdann(full_path, 'atr')

        data = {
            "signal": record.p_signal,
            "fs": record.fs,
            "channels": record.sig_name,
            "units": record.units,
            "annotations_samples": annotation.sample,
            "annotations_symbols": annotation.symbol,
            "raw_record": record,
            "raw_annotation": annotation
        }

        return data


if __name__ == "__main__":
    reader = database_reader()
    test_record = 105

    result = reader.get_data(test_record)

    print("\n=== MIT-BIH Record Loaded Successfully ===\n")
    print("Sampling frequency:", result["fs"])
    print("Channels:", result["channels"])
    print("Signal shape:", result["signal"].shape)

    print("\nFirst 10 samples (channel 0):")
    print(result["signal"][:10, 0])

    print("\nFirst 10 annotations:")
    print("Positions:", result["annotations_samples"][:10])
    print("Symbols:  ", result["annotations_symbols"][:10])
