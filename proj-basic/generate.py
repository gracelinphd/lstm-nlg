
import sys
import time
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


##
## load text and convert to lowercase
##
filename = 'text/wonderland.txt'
with open(filename) as inf:
	raw_text = inf.read().lower()

n_chars = len(raw_text)
print('Total chars:', n_chars)
# 144424


##
## convert characters to integers
##
# create a map of characters to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_vocab = len(chars)
print('Num of unique chars:', n_vocab)
# 47

print('-'*30)
print('Chars to int:', char_to_int)


##
## split book into 100 characters
##
# TODO: split by sentence; pad short truncate long.

# Each training pattern is comprised of 100 time steps of
# one character (X) followed by one character output (y).

# Slide the window along the whole book one character 
# at a time, allowing each character a chance to be 
# learned from the preceeding 100 characters
# (except for the first 100 characters).

# Example: if the sequence length if 5, then the first
# two training patterns are:
#	CHAPT -> E
#	HAPTE -> R

seq_length = 100
Xdata = []
Ydata = []

for i in range(0, len(raw_text)-seq_length, 1):
	seq_in = raw_text[i:i+seq_length]
	seq_out = raw_text[i+seq_length]
	Xdata.append([char_to_int[c] for c in seq_in])
	Ydata.append(char_to_int[seq_out])

n_patterns = len(Xdata)
print('Num of patterns:', n_patterns)
# 144324


##
## transform data for LSTM netowrk
##

# reshape format: [samples, time steps, features]
X = numpy.reshape(Xdata, (n_patterns, seq_length, 1))

# normalize so it's easier to learn by LSTM network
# and uses sigmoid activation by default
X = X / float(n_vocab)

# convert output to one hot encoding:
# 	sparse vector of length 47
# so the network can predict the probability of each 
# of the 47 different characters in the vocab 
# (an easier representation) rather than trying to force it to 
# predict precisely the next character.
y = np_utils.to_categorical(Ydata)


##
## LSTM network archetecture
##

# Single character classification problem with 47
# classees by optimizing the log loss (cross entropy)
# using the ADAM optimization algorithm for speed.

"""
model = Sequential()
# single hidden layer 256 memory units
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
# dropout with prob 0.2
model.add(Dropout(0.2))
# dense output layer (47 classes) with softmax
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
"""

##
## Train on entire dataset to learn the probability
## of each character in a sequence.
##

# We are interested in a generalization of dataset that
# minimizes the chose loss function.
# We are seeking a balance between generalization and
# overfitting but short of memorization.

# The network can be slow to train (~300 secs on NVidia K520 GPU).
# We use model checkpointing to record network weights
# to file each time an improvement in loss is observed at
# the end of the epoch.

# We will use the best set of weights (lowest loss) to
# instantiate our generative model later.

"""
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
				save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# Fit the model
epochs = 20
batch_size = 128

model.fit(X, y, epochs=epochs, batch_size=batch_size,
			callbacks=callbacks_list)
"""

# Results will be different since it's hard to fix the random seed
# for LSTM models to get 100% reproducible results.
# This is not a concern for this generative model.

# You'll see output files with different epochs and loss value.
# Keep the one with the smallest loss value.

# Network loss decreased almost every epoch, so more epochs
# could be helpful.

#----------------------------------------
# My rig's performance on GTX1060Ti
#
# GPU util: 40%
# Mem usage: 10.3G
# Time per epoch: 110s, 112s, 112s, 110s, etc.
#
# Best: epoch 17, loss = 1.9943
#----------------------------------------

##
## Generate text
##

"""
filename = 'weights-improvement-17-1.9943.hdf5'
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
"""

##
## reverse mapping to get original character
##
int_to_char = dict((i, c) for i, c in enumerate(chars))

##
## prediction
##

# Start with a seed sequence as input, generate next char
# then update the seed sequence to add generated char
# on the end and trim off the first character.
# Repeat process to keep predicting new char.

"""
start = numpy.random.randint(0, len(Xdata)-1)
pattern = Xdata[start]
seed_txt = ''.join([int_to_char[value] for value in pattern])
print('Seed:', seed_txt)

# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	print(result, end='')
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

print('\nDone.')
"""

#----------------------------------------

# Seed:
# dn’t afford to learn it.’ said the mock turtle with a sigh. ‘i
# only took the regular course.’
#
# ‘what

# Generated text:
# io io ’ou ’hul i seoll t alned the mors, and thnt heke to the seit wai oo the raatin an in hro aegin the was so tie was  ‘it was i then to tee she matthr was oo the sood, and whnt ier iese toee a foed hing of the soele  the woold hareen  and the mote oad the wiite rabbit was oo the tablit an thel, and the whst hood the doolo thth the gorst on the horst, and whnt an all rhe was so toeke the was no the tinl, and whlt ier iene the tiele was oo the table  she was so tore to ben to the woole, and saed to aeicn  she was so tore ti the corro sf the soeee of the tait, and whnt aelut to tee thot hord the ras of the grore, and the shilg tabbit was so the wool, ‘ht was t ain sea yhut aelinse the merter whth she west winng ’hu anl the wiile  and the toene of the sabtin and the care and the was so toeke the was no the tinl, and whlt ier iese toen a coutoe oo the hooke, and whnt all she wist sade to the white rabdin wh the tas of the grure, and the samt oo the was  soe kidgt woul she tealed to the

#----------------------------------------

##
## Larger LSTM
##

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# longer training time
epochs = 200 #50

# new checkpoint files
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger-epochs-"+str(epochs)+".hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# smaller batch size to allow network more of an opportunity to
# update and learn
batch_size = 64

# fit model
t0 = time.time()
model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
print('Runtime:', time.time() - t0, 'secs')


#----------------------------------------
# My rig's performance on GTX1060Ti
#
# epochs = 50
# 	GPU util: 30%
# 	Mem usage: 10.3G
# 	Time per epoch: 398s, 180s, 464s, 467s, 462s, etc.
# 	Run time: ~7 hours
# 	Best: epoch 49, loss: 1.2534
#
# epochs = 100
#	GPU Util: 40%
#	Run time: 42012 secs = ~11.67 hrs
#	Best: weights-improvement-70-1.2341-bigger-epochs-100.hdf5
#
#----------------------------------------


def gen_chars(pattern, length=1000):
	for i in range(length):
		x = numpy.reshape(pattern, (1, len(pattern), 1))
		x = x / float(n_vocab)
		prediction = model.predict(x, verbose=0)
		index = numpy.argmax(prediction)
		result = int_to_char[index]
		seq_in = [int_to_char[value] for value in pattern]
		print(result, end='')
		pattern.append(index)
		pattern = pattern[1:len(pattern)]	


# load the network weights
filename = "weights-improvement-70-1.2341-bigger-epochs-100.hdf5"

model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# pick a random seed
start = numpy.random.randint(0, len(Xdata)-1)
pattern = Xdata[start]

print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# generate characters
gen_chars(pattern)

print("\nDone.")


