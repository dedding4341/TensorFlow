from __future__ import print_function
import librosa
import numpy as np
import os

def spectralcentroid():
	i = 0
	array = []
	while i <= 5:
		y, sr = librosa.load(os.path.dirname(os.path.realpath(__file__)) + '/training/' + str(i) + '.flac')
		count = 0
		total = 0
		for x in np.nditer(librosa.feature.spectral_centroid(y=y, sr=sr)):
			total += x
			count += 1
		array.append(total/count)
		i += 1
	return array


# print(spectralcentroid())
