import sys
from os import listdir
from os.path import isfile, join
import numpy as np

from scipy.fft import fft, fftfreq
from scipy.io import wavfile


def sample(signal_fft):
    result = signal_fft
    for i in range(5):
        samples = signal_fft[::i + 1]
        result = np.multiply(result[:len(samples)], samples)
        result = [a * b for a, b in zip(result[:len(samples)], samples)]
    return np.array(result)


def find_human_freq(freqs, samples):
    end_freq = len(freqs) / 5
    end_samples = len(samples) / 5

    max_freq_index = samples.argmax()
    max_freq = freqs[max_freq_index]
    iter = 0
    # best results were given with given values
    while max_freq < 85 or max_freq > 255 and len(freqs) > end_freq and len(samples) > end_samples:
        freqs = np.delete(freqs, max_freq_index)
        samples = np.delete(samples, max_freq_index)
        max_freq_index = samples.argmax()
        max_freq = freqs[max_freq_index]
        iter += 1
    return max_freq


def freq(file):
    sr, data = wavfile.read(file)
    if data.ndim > 1:
        data = data[:, 0]
    signal = data[:100000]  # iterating over more takes too much time

    signal_fft = fft(signal)
    signal_fft = signal_fft[: int(len(signal_fft) / 2)]
    signal_fft = abs(signal_fft) / sr

    samples = sample(signal_fft)
    freqs = fftfreq(len(signal), 1.0 / sr)
    highest_freq = find_human_freq(freqs, samples)
    result = 'K' if highest_freq > 180 else 'M'
    return result


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print('No command line argument')
    # res = freq(sys.argv[1])
    # print(res)
    files = [f for f in listdir('trainall/') if isfile(join('trainall/', f))]
    correct = 0
    for file in files:
        res = freq('trainall/' + file)
        correct += 1 if res in file else 0
    print(str(correct) + '/' + str(len(files)))
