import numpy as np
import pandas as pd
import os
import seaborn as sns
import pickle
import music21
import matplotlib.pyplot as plt

from fractions import Fraction
from collections import defaultdict, Counter
from iteration_utilities import deepflatten #flatten nested lists

from music21 import midi, note, stream, instrument, meter, key
import itertools
import random

import string
import pretty_midi

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


"""
Performs a 1D simple convolution on the array a with kernel size n.
"""
def moving_average(a, n=3):
    t = np.floor(n/2).astype(int)
    b = np.zeros(a.shape)
    for i in range(b.shape[-1]):
        b[:, i] = np.mean(a[:, max(0, i-t):min(i+t+1, a.shape[-1])], axis=1)
        #b[i] = np.mean(a[max(0,i-t):min(i+t+1,len(a))])
    
    return b

def exp_func(a, l=1):
    b = a * (a > 0)

    return 1 - np.exp(-l*b)

def compute_threshold(avg_diff):
    return np.mean(avg_diff, axis=1) #+ np.std(avg_diff, axis=1)

def beat_segmentation_marco(avg_diff, threshold, times):
    threshold = threshold[:, None]
    idx = avg_diff > threshold # Candidate breaks
    ''' Give probability depending on:
        - distance from previous candidate breaks (unlikely htat two breaks are close to each other)
        - distance from threshold (the longer the pause, the more likely it is a phase break) '''
    # Probability depending on distance from threshold
    prob1 = exp_func(avg_diff - threshold, l=50)
    print(f"prob1 = {prob1.mean(axis=0)}")
    plot_avg_diff(prob1.mean(axis=0)[None, :], threshold.mean(), sig=4)
    high = prob1.mean(axis=0) > 0.5
    print(np.arange(prob1.shape[-1])[high] / 4 + 1)

    # Probability depending on distance from previous candidate breaks 
    # (if there are consecutive candidate breaks, the distance is counted from the first one)
    dist = np.zeros(idx.shape)
    count = np.zeros(idx.shape[0])
    for i in range(dist.shape[-1]):
        increment_mask = np.logical_or(np.logical_not(idx[:,i]), idx[:,i-1])
        count[increment_mask] += 1
        dist[:, i] = count*np.logical_not(increment_mask)
        count[np.logical_not(increment_mask)] = 0
    prob2 = exp_func(dist, l=3)

    # Final probability estimate
    prob = prob1 * prob2
    prob = np.mean(prob, axis=0)
    #print(f"prob.shape={prob.shape}")

    # Selection of final breaks
    prob_threshold = 0.1
    break_idx = prob > prob_threshold
    breaks = times[0, 1:][break_idx] # use some random times
    # observations: actual breaks often come just after slow downs (maybe 1-2 seconds after)

    return break_idx, breaks
    
def plot_avg_diff(avg_diff, threshold, sig=4):
    plt.plot(np.arange(avg_diff.shape[1]) / sig, avg_diff.T)
    plt.axhline(y=threshold.mean(), color='r', linestyle='--')

def plot_segmentation(avg_diff, threshold, break_idx, sig=4):
    plt.plot(np.arange(avg_diff.shape[1]) / sig, avg_diff.T, label="Intervals between beats")
    plt.axhline(y=threshold.mean(), color='r', linestyle='--', label="Interval threshold")
    points = (np.arange(avg_diff.shape[1]) / sig)[break_idx]
    print(f"points={points.shape}, break_idx={break_idx.shape}, avg_diff={avg_diff.shape}, sum={break_idx.sum()}")
    plt.plot(points, avg_diff[0, break_idx], 'r*', label="Selected phase breaks")
    #plt.legend()
    plt.title('Phase breaks')
    plt.show()

def read_diff_timings(filenames):
    times_list = []
    for f in filenames:
        df = pd.read_csv(f, sep="\t", header=None)
        times = df.iloc[:,0].values
        times_list.append(times)
    return times_list, [times[1:] - times[:-1] for times in times_list]



"""
2 different approaches:
in my function, we check whether the current beat is significantly longer than the n beats that came before
in marco's function, we check whether the average beat length around the current beat is longer than other average beat length in the whole piece
"""

# testing code, to remove
#from src.data import read_annotations
#beats = read_annotations("asap-dataset/Schubert/Impromptu_op.90_D.899/3/Hou06M_annotations.txt")
#values = beat_length_segmentation(beats, 4, kernel_size=4)
#for i in range(values.shape[0]):
#    if i%4==0:
#        print('/')
#    print(f"{i/4+1}: {values[i]}")