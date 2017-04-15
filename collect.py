from __future__ import print_function
import librosa
import numpy as np
import os

batchCounter = 0
data_num = 8000

# parameter determines loop terminator which is batch_size * batchCounter (e.g. 80)
def spectralcentroid(batchSize):
	i = batchCounter * batchSize
	j = i
	array = []
	while i < j + batchSize:
		y, sr = librosa.load(os.path.dirname(os.path.realpath(__file__)) + '/training/' + str(i) + '.flac')
		dat = librosa.feature.spectral_centroid(y=y, sr=sr)[0].tolist()
		while(len(dat) < 500):
			dat.append(0)
		array.append(np.asarray(dat))
		i += 1
	return array

def read(batchSize):
	stray = []
	curr = batchCounter*batchSize
	with open(os.path.dirname(os.path.realpath(__file__)) + '/training/labels.txt') as f:
		strings = f.readlines()
	for s in strings:
		if (s.split('|')[1].rstrip() == 'M'):
			stray.append([1,0])
		else:
			stray.append([0,1])
	return stray[curr:curr+batchSize]

def nextbatch(batchSize):
	global batchCounter
	batchCounter += 1
	return (spectralcentroid(batchSize), read(batchSize))
print(nextbatch(5))
