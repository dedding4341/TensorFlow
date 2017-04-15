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
	# array = []
	# while i < j + batchSize:
	# 	y, sr = librosa.load(os.path.dirname(os.path.realpath(__file__)) + '/training/' + str(i) + '.flac')
	# 	dat = librosa.feature.spectral_centroid(y=y, sr=sr)[0].tolist()
	# 	while(len(dat) < 500):
	# 		dat.append(0)
	# 	array.append(np.asarray(dat))
	# 	i += 1
	# return array
	array  = np.zeros((batchSize, 500))
	while i < j + batchSize:
		y, sr = librosa.load(os.path.dirname(os.path.realpath(__file__)) + '/training/' + str(i) + '.flac')
		dat = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
		q = 0
		w = 0
		while q < len(dat):
			array[w, q] = dat[q]
			q += 1
		i += 1
		w += 1
		if (w == 5):
			break
	return array

def read(batchSize):
	# stray = []
	# curr = batchCounter*batchSize
	# with open(os.path.dirname(os.path.realpath(__file__)) + '/training/labels.txt') as f:
	# 	strings = f.readlines()
	# for s in strings:
	# 	if (s.split('|')[1].rstrip() == 'M'):
	# 		stray.append([1,0])
	# 	else:
	# 		stray.append([0,1])
	# return stray[curr:curr+batchSize]
	stray = np.zeros((batchSize, 2))
	#print(stray.shape)
	curr = batchCounter*batchSize
	with open(os.path.dirname(os.path.realpath(__file__)) + '/training/labels.txt') as f:
		strings = f.readlines()

	myslice = strings[curr:curr+5]
	i = 0
	for s in myslice:
		if (s.split('|')[1].rstrip() == 'M'):
			stray[i, 0] = 1
			stray[i, 1] = 0
		else:
			stray[i, 0] = 0
			stray[i, 1] = 1
		i += 1
		if (i == batchSize):
			break
	#print(curr, batchSize)
	return stray

def nextbatch(batchSize):
	global batchCounter
	batchCounter += 1
	return (spectralcentroid(batchSize), read(batchSize))
