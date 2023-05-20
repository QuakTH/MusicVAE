import os
import zipfile
from typing import List, Optional
from urllib import request

import numpy as np
import pandas as pd
import pretty_midi
from pretty_midi import PrettyMIDI
from tqdm import tqdm

from musicvae import music_utils


class DownloadProgressBar(tqdm):
    def update_progress(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None):
        """Show download progress bar.
        Code and reference from:
        https://github.com/tqdm/tqdm#hooks-and-callbacks:~:text=%5B...%5D-,import,-urllib%2C%20os

        :param b: Number of blocks transferred so far, defaults to 1.
        :param bsize: Size of each block, defaults to 1.
        :param tsize: Total size, defaults to None
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_datasets(directory: str) -> None:
    """Download the Groove MIDI dataset to the specific directory.

    :param directory: directory to download the dataset.
    """
    zipfile_path = os.path.join(directory, "dataset.zip")

    # download the dataset using urlretrieve
    # progress bar is for checking progression
    with DownloadProgressBar(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc="Download Progress",
    ) as pbar:
        request.urlretrieve(
            "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip",
            zipfile_path,
            reporthook=pbar.update_progress,
        )

    # unzip the files in the same directory
    with zipfile.ZipFile(zipfile_path, "r") as zipf:
        zipf.extractall(directory)
    print("Unzip completed.")


def get_vectorized_drums(
    inst: List[pretty_midi.Instrument], fs: float, start_time: float, end_time: float
) -> np.ndarray:
    """Generate a 2D matrix which have a
       1st dim as a time slot
       2nd dim as a which note has been played in a particular time slot

    :param inst: List of instruments.
    :param fs: Sampling rate.
    :param start_time: Instrument start time.
    :param end_time: Instrument end time.
    :return: 2D matrix of drum representation.
    """
    fs_time = 1 / fs

    time_slots = np.arange(start_time, end_time + fs_time, fs_time)
    drum_mat = np.zeros((time_slots.shape[0], len(music_utils.PAPER_MAP_TABLE)))

    for note in inst.notes:
        if note.pitch not in music_utils.PITCH_USED:
            continue

        # find the nearest start end end time slot of the note
        start_index = np.argmin(np.abs(time_slots - note.start))
        end_index = np.argmin(np.abs(time_slots - note.end))

        # create a time slot range
        if start_index == end_index:
            end_index += 1
        range_index = np.arange(start_index, end_index)

        # each 2nd dim index is a encoded class of a representative pitch
        # if there was a pitch played in the found time slot,
        # set the index of the pitch to 1
        for pitches, represent_pitch in music_utils.PAPER_MAP_TABLE.items():
            if note.pitch in pitches:
                enc_idx = music_utils.PITCH_ENCODED[represent_pitch]

        for index in range_index:
            drum_mat[index, enc_idx] = 1

    return drum_mat


def create_batch(
    vectorized_drums: np.ndarray,
    sequence_size: int = music_utils.NOTE_PER_BAR * music_utils.BAR_SIZE,
) -> Optional[np.ndarray]:
    """Create a batch of drum vectors sequence with the size of a bar size of interest.

    :param vectorized_drums: A vector sequence representation of drums.
    :param sequence_size: A sequence size for each batch, defaults to music_utils.NOTE_PER_BAR*music_utils.BAR_SIZE
    :return: A 3D array of [ Batch, Sequence, Encode ].
    """
    batches = []
    batch_count = vectorized_drums.shape[0] // sequence_size

    # if the matrix row is smaller than the seq size return None
    if not batch_count:
        return None

    # only use the batches where there is a full drum sequence of `sequence_size`
    # meaning the last beats may be dropped
    for batch_idx in range(batch_count):
        start_index = sequence_size * batch_idx
        end_index = sequence_size * (batch_idx + 1)

        batches.append(np.expand_dims(vectorized_drums[start_index:end_index], axis=0))

    return np.vstack(batches)


def one_hot_encode(batched_drums: np.ndarray) -> np.ndarray:
    """Encode the binary like 1d array to a one-hot encoding array.
    For example:
    If there is a binary like array of [0, 1, 0, 1],
    this means this list will represent 0 ~ 2**4 - 1 in decimal
    This function will then create a array of which is the length of 2 ** 4
    and assign 1 to the index of the binary array represents. (in this example index 5).

    :param batched_drums: Batch to one hot encode.
    :return: One hot encoded Batch.
    """

    def bin_array_one_hot(bin_array: np.ndarray) -> np.ndarray:
        """A inner function that actually does the one hot encoding.

        :param bin_array: Binary array to one hot encode.
        :return: One hot encoded array.
        """
        decimal = 0
        length = len(bin_array)

        one_hot = np.zeros(2**length)

        for idx, value in enumerate(bin_array):
            decimal += 2 ** (length - idx - 1) * value

        one_hot[int(decimal)] = 1.0

        return one_hot

    batch_one_hot = np.apply_along_axis(
        bin_array_one_hot, batched_drums.ndim - 1, batched_drums
    )

    return batch_one_hot


def parse_one_hot_encode(one_hot_encoded: np.ndarray) -> np.ndarray:
    """Return a binary array representation of the one hot encoded array.

    :param one_hot_encoded: Batch containing one hot encoded vectors.
    :return: Decoded batch.
    """

    def one_hot_to_bin_array(encoded_array: np.ndarray) -> np.ndarray:
        """Inner function that actually does the decoding.

        :param encoded_array: One hot encoded vector.
        :return: Decoded vector.
        """
        bin_length = int(np.log2(len(encoded_array)))
        bin_array = np.zeros(bin_length)

        encoded_value = (encoded_array > 0.0).nonzero()[0][0]
        for idx in range(bin_length):
            mod = encoded_value % 2
            divided = encoded_value // 2

            bin_array[bin_length - 1 - idx] = mod
            if divided == 0:
                break
            encoded_value = divided

        return bin_array

    batch_bin_array = np.apply_along_axis(
        one_hot_to_bin_array, one_hot_encoded.ndim - 1, one_hot_encoded
    )
    return batch_bin_array


def create_train_val_test(directory: str) -> None:
    """Generate the train, test, validation datasets
    using the info.csv and the midi files.

    :param directory: directory where the midi dataset is stored.
    """
    info_df = pd.read_csv(os.path.join(directory, "groove/info.csv"))

    for split_type in tqdm(info_df["split"].unique(), desc="Generating dataset"):
        batches = []

        splited_df = info_df[
            info_df["split"].eq(split_type) & info_df["time_signature"].eq("4-4")
        ]  # only use midi files with 4/4 time signature

        for midif_file_path in splited_df["midi_filename"]:
            midi = PrettyMIDI(os.path.join(directory, "groove", midif_file_path))

            start_time = midi.get_onsets()[0]
            beats = midi.get_beats(start_time)
            # This is used as a sample rate when vectorizing drum data
            fs = music_utils.change_fs(beats)

            # If a midi is too short skip
            if fs is None:
                continue

            for inst in midi.instruments:
                if inst.is_drum:
                    # Create a batch of vectorized drum features for train, valid, test
                    batch = create_batch(
                        get_vectorized_drums(inst, fs, start_time, inst.get_end_time())
                    )
                    if batch is not None:
                        one_hot_encoded = one_hot_encode(batch)
                        batches.append(one_hot_encoded)
        np.save(os.path.join(directory, f"{split_type}.npy"), np.vstack(batches))
