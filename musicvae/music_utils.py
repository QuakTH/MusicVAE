from typing import List, Optional, Tuple
import numpy as np
import pretty_midi

# This is a mapping of each pitch to a paper mapping pitch
# cited from : https://magenta.tensorflow.org/datasets/groove#:~:text=if%201%2Dindexed).-,Drum,-Mapping
PAPER_MAP_TABLE = {
    (36,): 36,
    (38, 40, 37): 38,
    (48, 50): 50,
    (45, 47): 47,
    (43, 58): 43,
    (26, 46): 46,
    (22, 42, 44): 42,
    (49, 52, 55, 57): 49,
    (51, 53, 59): 51,
}

PITCH_USED = (
    36,
    38,
    40,
    37,
    48,
    50,
    45,
    47,
    43,
    58,
    26,
    46,
    22,
    42,
    44,
    49,
    52,
    55,
    57,
    51,
    53,
    59,
)

# each paper mapped note meaning
PITCH_CLASS = {
    36: "Bass",
    38: "Snare",
    50: "Hight Tom",
    47: "Low-Mid Tom",
    43: "High Floor Tom",
    46: "Open Hi-Hat",
    42: "Closed High-Hat",
    49: "Crash Cymbal",
    51: "Ride Cymbal",
}

# encoded note mapping
PITCH_ENCODED = {
    36: 0,
    38: 1,
    50: 2,
    47: 3,
    43: 4,
    46: 5,
    42: 6,
    49: 7,
    51: 8,
}

# will be using 16th note
NOTE_PER_BAR = 16

# only interested in 4 bars
BAR_SIZE = 4


def change_fs(beats: np.ndarray, note_per_bar: int = NOTE_PER_BAR) -> Optional[float]:
    """Change the sampling rate.

    :param beats: list of beat location.
    :param note_per_bar: how many notes for one bar.
                         Use only the multiple of four, defaults to 16
    :return: changed sampling rate.
    """
    try:
        quarter_length = beats[1] - beats[0]
        changed_length = quarter_length / (note_per_bar / 4)
        changed_fs = 1 / changed_length
    except IndexError:
        return None

    return changed_fs


def numpy_to_midi(array:np.ndarray, fs:int) -> pretty_midi.PrettyMIDI:
    """Create a PrettyMIDI object containing drum session from `array`.

    :param array: A sequence of drum features.
    :param fs: Sampling rate, A integer of 8 to 10 is preferred.
    :return: A PrettyMIDI object.
    """
    fs_time = 1 / fs

    encoding_reversed = {v: k for k, v in PITCH_ENCODED.items()}

    # create a PrettyMIDI instance
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=32, is_drum=True)
    midi.instruments.append(instrument)

    # create notes using the time slot number to calculate the start and end time
    # and reversed encoded pitches to arrange which notes have been played
    for time_slot, vectorized_drums in enumerate(array):
        if vectorized_drums.sum() == 0:
            continue

        start_time = fs_time * time_slot
        end_time = fs_time * (time_slot + 1)

        for enc_idx in np.nonzero(vectorized_drums)[0]:
            pitch = encoding_reversed[enc_idx]
            instrument.notes.append(pretty_midi.Note(80, pitch, start_time, end_time))

    return midi
