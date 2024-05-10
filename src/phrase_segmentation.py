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

"""
Aggregates the velocity information of the piece in local averages at each beat.
Several approaches could make sense, here we want to detect sharp changes in velocity.
We could, for each beat, compute the average velocity of all the midi events between the considered beat and the consecutive beat.
Or we could also consider, for beat n, all the midi events between beat n-1 and beat n+1, so that the computed value for a beat
reflects the velocity acutally around it.
Some fancier approaches could be done (curve fitting all the events, and then evaluating the curve at each beat), but they probably
aren't relevant here  
"""
def compute_smooth_velocity_curve(midi, beats):
    ...

"""
Make use of the observation we made that stark changes in velocity often correspond to phrase boundaries.
Use a sliding window approach comparable to the function beat_length_segmentation, but this time tries
to find beats for which the absolute difference between the average velocities before the beat and after the beat are maximized.
For example, for a kernel_size of l, at beat n, we want to assign to beat n a value proportional to the absolute difference 
between the average velocities of beats [n - l/2: n] and the average velocities of beats [n : n + l/2], or something like that.
"""
def velocity_curve_segmentation(curve, kernel_size=8) -> list[float]:
    ...

"""
We have a prior on the length of phrases. There's the typical value of 8 (that may be 4 instead for the schubert piece).
We want to account for that prior when sampling boundaries using the computed values. We want that having a phrase boundary
becomes unlikely if we just had a boundary, and becomes more and more likely if the last boundary was long ago, to represent
the fact that our phrases shouldn't be too long.
I'm not sure of how to implement this, and of the data representations involved
"""
def phrase_length_prior(values, prior, sig) -> list[float]:
    ...

"""
Sample phrase boundaries using the methods above.
We need to find a way to aggregate the different methods. If we transform them all to a common comparable space (probability space?),
it may be straightforward to combine them. Then we also need to sample the actual boundaries, somehow taking the above length prior
into account. then we return the boundaries (could be an array of integer or boolean)
"""
def sample_boundaries(values):
    ...

# testing code, to remove
from src.data import read_annotations
beats = read_annotations("asap-dataset/Schubert/Impromptu_op.90_D.899/3/Hou06M_annotations.txt")
values = beat_length_segmentation(beats, 4, kernel_size=4)
for i in range(values.shape[0]):
    if i%4==0:
        print('/')
    print(f"{i/4+1}: {values[i]}")