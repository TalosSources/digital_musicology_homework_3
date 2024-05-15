import itertools
import os
import pickle
import random
import string
from collections import Counter, defaultdict
from fractions import Fraction

import matplotlib.pyplot as plt
import music21
import numpy as np
import pandas as pd
import pretty_midi
import seaborn as sns
from iteration_utilities import deepflatten  # flatten nested lists
from music21 import instrument, key, meter, midi, note, stream


# """
# Aggregates the velocity information of the piece in local averages at each beat.
# Several approaches could make sense, here we want to detect sharp changes in velocity.
# We could, for each beat, compute the average velocity of all the midi events between the considered beat and the consecutive beat.
# Or we could also consider, for beat n, all the midi events between beat n-1 and beat n+1, so that the computed value for a beat
# reflects the velocity acutally around it.
# Some fancier approaches could be done (curve fitting all the events, and then evaluating the curve at each beat), but they probably
# aren't relevant here
# """
def compute_smooth_velocity_curve(midi, beats, kernel_size=1):
    notes = []
    for inst in midi.instruments:
        for note in inst.notes:
            if note.end - note.start > 0:  # ignore accompaniment?
                notes.append((note.start, note.velocity))
    notes.sort(key=lambda n: n[0])  # sort notes by start time
    velocity_averages = np.zeros(beats.shape[-1])
    beat_index = 0
    current_count = 0
    for note in notes:
        if beat_index < beats.shape[-1] - 1 and note[0] > beats[beat_index + 1]:
            if current_count > 0:
                velocity_averages[
                    beat_index
                ] /= current_count  # What todo with beats without velocity? average velocity?
            current_count = 0
            beat_index += 1
        velocity_averages[beat_index] += note[1]
        current_count += 1
    if current_count > 0:
        velocity_averages[beat_index] /= current_count  # rescale in the end too
    return np.convolve(velocity_averages, np.ones(kernel_size) / kernel_size, "same")


def compute_all_smooth_velocity_curves(midis, beats, kernel_size=1):
    return np.array(
        [
            compute_smooth_velocity_curve(midi, beat, kernel_size)
            for (midi, beat) in zip(midis, beats)
        ]
    )


# """
# Performs a 1D simple convolution on the array a with kernel size n.
# """
def moving_average(a, n=3):
    t = np.floor(n / 2).astype(int)
    b = np.zeros(a.shape)
    for i in range(b.shape[-1]):
        b[:, i] = np.mean(a[:, max(0, i - t) : min(i + t + 1, a.shape[-1])], axis=1)

    return b


def exp_func(a, l=1):
    b = a * (a > 0)

    return 1 - np.exp(-l * b)


# """
# Weighted sigmoid
# """
def sigmoid(x, s=1):
    return 1.0 / (1 + np.exp(-s * x))


def compute_threshold(avg_diff):
    return np.mean(avg_diff, axis=1)  # + np.std(avg_diff, axis=1)


def alternative_beat_values(avg_diff, threshold):
    threshold = threshold[:, None]
    idx = avg_diff > threshold  # Candidate breaks
    # Probability depending on distance from threshold
    prob1 = exp_func(avg_diff - threshold, l=50)

    return idx, prob1.mean(axis=0)


# """
# Params:
# 	beat_lengths (list[float]) : ordered list of beats timing in seconds
# 	kernel_size (int) : size of the window used to check whether a beat is slower than preceding beats
# Returns an array of numeric values akin to probabilities,
# for each beat of being a phrase boundary.
# """
def beat_values(beat_lengths, kernel_size, s=4):
    beat_count = beat_lengths.shape[-1]
    values = np.zeros(beat_count)
    for i in range(0, beat_count - kernel_size):
        # first idea: compute the mean beat length, and if the last one is longer, conclude there's a boundary
        window = beat_lengths[:, i : i + kernel_size]
        mean_window_length = window.mean(axis=1)
        relative_differences = (window[:, -1] - mean_window_length) / mean_window_length
        values[
            i + kernel_size - 1
        ] = (
            relative_differences.mean()
        )  # we assign a high value on the first beat after a slowdown

    values = sigmoid(values, s)

    # if a beat is slow, it means the beat just after is probably a phrase beginning:
    # shift everything to the right
    values[1:] = values[:-1]
    values[0] = 0
    return values


def velocity_values(velocity_curves, half_kernel_size=4):
    kernel_size = 2 * half_kernel_size

    # build the kernels
    kernel_increase = np.ones(kernel_size)
    kernel_increase[:half_kernel_size] = -1
    kernel_average = np.ones(kernel_size) / (
        kernel_size
    )  # used to normalize the resuslts
    # print(f"kernels = {kernel_increase}, {kernel_average}")
    # perform convolutions
    increase_convolution = np.array(
        [np.convolve(v, kernel_increase, "same") for v in velocity_curves]
    )
    average_convolution = np.array(
        [np.convolve(v, kernel_average, "same") for v in velocity_curves]
    )

    # by performing abs, we give high values for increases and decreases
    values = np.abs(increase_convolution) / average_convolution
    values[:, :half_kernel_size] = 0
    values[:, -half_kernel_size:] = 0
    return sigmoid(values.mean(axis=0))


def performance_segmentation(
    avg_beat_length,
    beat_times,
    velocity_curve,
    beat_kernel=4,
    velocity_kernel=4,
    sig=4,
    weights=(0.5, 0.5),
    decay=0.3,
    selection_threshold=0.65,
    plot=False,
):
    """Give probability depending on:
    - distance from previous chosen breaks (unlikely that two breaks are close to each other)
    - beat value: beat duration compared to beat duration before: detects slowing down at the end of phrases
    - velocity gradient: detects changes in velocity, that could indicate a new phrase starting
    """

    beat_v = beat_values(avg_beat_length, beat_kernel)
    velocity_v = velocity_values(velocity_curve, velocity_kernel)

    prob_ = weights[0] * velocity_v[:-1] + weights[1] * beat_v

    prob = prob_ * (
        np.arange(prob_.shape[0]) % sig == 0
    )  # heuristic: phrases start and finish at bars

    # Selection of final breaks: make breaks less probable just after a break
    break_idx = np.full(prob.shape[0], False)
    beats_since_boundary = 0
    for i in range(prob.shape[0]):
        beats_since_boundary += 1
        # we make a boundary more or less probable if the measure afterwards has a lesser or higher value, and less probable if we just had a boundary
        p = (prob[i] + prob[i] - prob[min(i + sig, prob.shape[0] - 1)]) * (
            1 - np.exp(-decay * beats_since_boundary)
        )
        if p > selection_threshold:
            break_idx[i] = True
            beats_since_boundary = 0

    breaks = beat_times[0, 1:][break_idx]  # use some random times
    # observations: actual breaks often come just after slow downs (maybe 1-2 seconds after)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        plot_segmentation(axes[2], beat_v, velocity_v[:-1], prob_, break_idx, sig=sig)
        plot_velocity_curve(axes[1], velocity_curve[0][:-1], break_idx, sig=sig)
        plot_beat_curve(axes[0], avg_beat_length[0], break_idx, sig=sig)
        plt.legend()
        fig.tight_layout()
        plt.savefig("full_plot.pdf", dpi=600)
        plt.show()

    return break_idx, breaks


def plot_avg_diff(avg_diff, sig=4):
    plt.plot(np.arange(avg_diff.shape[1]) / sig, avg_diff.T)


def plot_segmentation(axes, beat_v, velocity_v, probs, break_idx, sig=4):
    axes.plot(
        np.arange(beat_v.shape[0]) / sig,
        beat_v,
        label="Timing likelihood values",
        alpha=0.5,
        color="#2b83ba",
    )
    axes.plot(
        np.arange(velocity_v.shape[0]) / sig,
        velocity_v,
        label="Velocity likelihood values",
        alpha=0.5,
        color="#fdae61",
    )
    axes.plot(
        np.arange(probs.shape[0]) / sig,
        probs,
        label="Combined likelihood values",
        color="green",
    )
    points = (np.arange(beat_v.shape[0]) / sig)[break_idx]
    axes.plot(
        points,
        probs[break_idx],
        "*",
        label="Selected phrase boundaries",
        color="#d7191c",
    )
    axes.set_xlabel("Measure")
    axes.set_ylabel("Likelihood")
    # plt.legend()
    axes.set_title("Likelihood values and phrase segmentation")
    # plt.savefig("segmentation_plot.pdf", dpi=600)
    # plt.show()


def plot_velocity_curve(axes, velocity_curve, break_idx, sig=4):
    axes.plot(
        np.arange(velocity_curve.shape[0]) / sig,
        velocity_curve,
        label="Velocity curve",
        color="#fdae61",
    )
    points = (np.arange(velocity_curve.shape[0]) / sig)[break_idx]
    axes.plot(
        points,
        velocity_curve[break_idx],
        "*",
        label="Selected phase breaks",
        color="#d7191c",
    )
    axes.set_xlabel("Measure")
    axes.set_ylabel("Velocity curve")
    # plt.legend()
    axes.set_title("Phrase boundaries over velocity curve")
    # plt.savefig("velocity_plot.pdf", dpi=600)
    # plt.show()


def plot_beat_curve(axes, avg_beat_length, break_idx, sig=4):
    axes.plot(
        np.arange(avg_beat_length.shape[0]) / sig,
        avg_beat_length,
        label="Beat duration",
        color="#2b83ba",
    )
    points = (np.arange(avg_beat_length.shape[0]) / sig)[break_idx]
    axes.plot(
        points,
        avg_beat_length[break_idx],
        "*",
        label="Selected phase breaks",
        color="#d7191c",
    )
    axes.set_xlabel("Measure")
    axes.set_ylabel("Beat duration")
    # plt.legend()
    axes.set_title("Phrase boundaries over beat_duration")
    # plt.savefig("beat_plot.pdf", dpi=600)
    # plt.show()


def read_diff_timings(filenames):
    times_list = []
    for f in filenames:
        df = pd.read_csv(f, sep="\t", header=None)
        times = df.iloc[:, 0].values
        times_list.append(times)
    return times_list, [times[1:] - times[:-1] for times in times_list]


# """
# 2 different approaches:
# in my function, we check whether the current beat is significantly longer than the n beats that came before
# in marco's function, we check whether the average beat length around the current beat is longer than other average beat length in the whole piece
# """
