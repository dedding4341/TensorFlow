from __future__ import print_function
import librosa
import numpy as np
import os

batchNum = 0
data_num = 8000

# parameter determines loop terminator which is batch_size * batchNum (e.g. 80)
def spectralcentroid(batchSize):
	i = batchNum * batchSize
	j = i
	array = []
	while i < j + batchSize:
		y, sr = librosa.load(os.path.dirname(os.path.realpath(__file__)) + '/training/' + str(i) + '.flac')
		array.append(librosa.feature.spectral_centroid(y=y, sr=sr))
		i += 1
	return array

print(spectralcentroid(5))

def read(batchSize):
	stray = []
	curr = batchNum*batchSize
	with open(os.path.dirname(os.path.realpath(__file__)) + '/training/labels.txt') as f:
		strings = f.readlines()
	for s in strings:
		stray.append(s.split('|')[1].rstrip())
	return stray[curr:curr+batchSize]
	
def next_batch(batchSize):
	#if batchNum * batch_size >= data_num:
	batchCounter += 1
	return (spectralcentroid(batchSize), read(batchSize))
