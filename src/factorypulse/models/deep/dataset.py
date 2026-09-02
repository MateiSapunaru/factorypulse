from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceMetadata:
    machine_ids: list[str]
    end_timestamps: list[pd.Timestamp]
    labels: list[int]


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


def build_lstm_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    machine_id_column: str,
    timestamp_column: str,
    target_column: str,
    sequence_length: int,
) -> tuple[np.ndarray, SequenceMetadata]:
    sequences: list[np.ndarray] = []
    machine_ids: list[str] = []
    end_timestamps: list[pd.Timestamp] = []
    labels: list[int] = []

    for machine_id, group in df.groupby(machine_id_column):
        group = group.sort_values(timestamp_column).reset_index(drop=True)

        x = group[feature_columns].to_numpy(dtype=np.float32)
        y = group[target_column].to_numpy(dtype=int)
        timestamps = pd.to_datetime(group[timestamp_column]).to_list()

        for end_idx in range(sequence_length - 1, len(group)):
            start_idx = end_idx - sequence_length + 1

            seq_x = x[start_idx : end_idx + 1]
            seq_y = y[start_idx : end_idx + 1]

            sequences.append(seq_x)
            machine_ids.append(str(machine_id))
            end_timestamps.append(pd.Timestamp(timestamps[end_idx]))
            labels.append(int(seq_y.max()))

    sequence_array = (
        np.stack(sequences) if sequences else np.empty((0, sequence_length, len(feature_columns)))
    )
    metadata = SequenceMetadata(
        machine_ids=machine_ids,
        end_timestamps=end_timestamps,
        labels=labels,
    )

    return sequence_array, metadata
