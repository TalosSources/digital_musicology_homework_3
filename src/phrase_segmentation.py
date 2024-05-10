import numpy as np

"""
Params:
    beats (list[float]) : ordered list of beats timing in seconds
    sig (int) : time signature of the piece (actually only the beat count per measure)
Returns an array of numeric values akin to probabilities, 
for each beat of being a phrase boundary.
"""
def beat_length_segmentation(beats: list[float], sig: int, kernel_size=8) -> list[float]:
    values = np.zeros(len(beats))
    for i in range(0, len(beats)-kernel_size):
        # first idea: compute the mean beat length, and if the last one is longer, conclude there's a boundary
        window = np.array(beats[i:i+kernel_size])
        beat_durations = window[1:] - window[:-1]
        mean_duration = beat_durations.mean()
        difference = beat_durations[-1] - mean_duration
        values[i+kernel_size-1] = difference # we assign a high value on the first beat after a slowdown
    return values

def compute_smooth_velocity_curve(midi, beats):
    ...

def velocity_curve_segmentation(curve, kernel_size=8) -> list[float]:
    ...

def measure_length_prior(values) -> list[float]:
    ...

from src.data import read_annotations
beats = read_annotations("asap-dataset/Schubert/Impromptu_op.90_D.899/3/Hou06M_annotations.txt")
values = beat_length_segmentation(beats, 4, kernel_size=4)
for i in range(values.shape[0]):
    if i%4==0:
        print('/')
    print(f"{i/4+1}: {values[i]}")