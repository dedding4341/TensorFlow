from __future__ import print_function
import librosa
import numpy as np
import os

batchCounter = 0
data_num = 8000

# parameter determines loop terminator which is batch_size * batchNum (e.g. 80)
def spectralcentroid(batch_size):
	i = 0
	array = []
	while i < batch_size:
		y, sr = librosa.load(os.path.dirname(os.path.realpath(__file__)) + '/training/' + str(i) + '.flac')
		count = 0
		total = 0
		for x in np.nditer(librosa.feature.spectral_centroid(y=y, sr=sr)):
			total += x
			count += 1
		array.append(total/count)
		i += 1
	return array


print(spectralcentroid())


def next_batch(batch_size):
	#if batchNum * batch_size >= data_num:
	batchCounter += 1
	return (spectralcentroid(batch_size), ) # find way to get the output here as second parameter (use parsing ???)
