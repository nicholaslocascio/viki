import matplotlib.pyplot as plt
from scipy.io import wavfile # get the api
import scipy.fftpack as sfft
import numpy as np
import math
import pickle
import sys
import theano
import json

sys.setrecursionlimit(200000)

MAX_FILES_PER_SPEAKER = 2

#NUM_COMPONENTS_IN_SPECTOGRAM = 500
NUM_COMPONENTS_IN_SPECTOGRAM = 200

NUM_SPECTOGRAMS_IN_SEQUENCE = 20
NUM_DATAPOINTS_IN_SPECTOGRAM = 400
NUM_DATAPOINTS_IN_SEQUENCE = NUM_DATAPOINTS_IN_SPECTOGRAM*NUM_SPECTOGRAMS_IN_SEQUENCE

GLOBAL_MODEL = None

#SPEAKER_IDS = [84, 174, 251, 422, 652, 777, 1272]
#SPEAKER_IDS = [103,	1553, 201, 2691, 3235, 3947, 4406, 5192, 6019, 6848, 7511, 8324, 1034, 1578, 2092, 27, 3240, 3982, 441, 5322, 6064, 6880, 7517, 839]
SPEAKER_IDS = [103, 1553]

#SPEAKER_PATH = 'clean_speech/'
SPEAKER_PATH = 'big_speech/clean100/'
TRAINED_MODEL_FILE_NAME = 'model_temp/weights.hdf5'

def normalize_outliers(data, m):
    u = np.mean(data)
    s = np.std(data)
    filtered = [e if (u - m*s < e < u + m * s) else u+((m/10.0)*s) for e in data]
    return np.array(filtered)

def normalize_extreme_outliers(data):
	return normalize_outliers(data, 50)

def make_spectogram_sequence_matricies_for_file(filename, k_cutoff=NUM_COMPONENTS_IN_SPECTOGRAM):
	"""Each element is a matrix that is (k_cutoff x MAX_SEQUENCE_LENGTH)."""
	#filename = 'test2.wav'
	fs, raw_data = wavfile.read(open(filename, 'r')) # load the data
	if len(raw_data.T) == 2:
		data = raw_data.T[0]
	else:
		data = raw_data.T
	waveform = [(ele/(2.**14)) for ele in data] # this is 8-bit track, b is now normalized on [-1,1)
	num_chunks_from_waveform = len(waveform)/(NUM_DATAPOINTS_IN_SEQUENCE)
	waveform_chunks = [waveform[index*NUM_DATAPOINTS_IN_SEQUENCE:(index+1)*NUM_DATAPOINTS_IN_SEQUENCE] for index in range(num_chunks_from_waveform)]
	spectogram_sequence_matricies_list = []
	for chunk in waveform_chunks:
		num_segments_in_chunk = len(chunk)/NUM_DATAPOINTS_IN_SPECTOGRAM
		segments = [chunk[index*NUM_DATAPOINTS_IN_SPECTOGRAM:(index+1)*NUM_DATAPOINTS_IN_SPECTOGRAM] for index in range(num_segments_in_chunk)]
		spectogram_sequence_matrix = np.zeros((NUM_SPECTOGRAMS_IN_SEQUENCE, NUM_COMPONENTS_IN_SPECTOGRAM))
		for seg_i, segment in enumerate(segments):
			spectogram = make_spectogram(segment)
			for c_i, component in enumerate(spectogram):
				spectogram_sequence_matrix[seg_i, c_i] = component
		spectogram_sequence_matricies_list.append(spectogram_sequence_matrix)
	return spectogram_sequence_matricies_list

def make_spectogram(waveform, k_cutoff=NUM_COMPONENTS_IN_SPECTOGRAM):
	"""Makes an fft spectogram from a waveform and applies a low-pass filter to the fft components."""
	spectogram = abs(sfft.fft(waveform)) # create a list of complex number
	spectogram = spectogram[:k_cutoff]
	spectogram = [math.log1p(component) for component in spectogram]
	#spectogram = normalize_extreme_outliers(spectogram)
	return spectogram

def plot_spectogram(spectogram):
	d = len(spectogram)/2  # you only need half of the fft list
	plt.plot(spectogram[:(d-1)],'r') 
	plt.show()

def plot_waveform(waveform):
	plt.plot(waveform)
	plt.show()

import os, fnmatch
import re

def find_files(directory, pattern):
	answers = []
	for root, dirs, files in os.walk(directory):
		for basename in files:
			if pattern.match(basename):
				filename = os.path.join(root, basename)
				answers.append(filename)
	return answers

def get_wave_file_names_for_id(person_id):
	regex = '^' + str(person_id) + '\-.*\.wav'
	regexp = re.compile(regex)
	wave_form_file_names = find_files(SPEAKER_PATH, regexp)
	print "found all files"
	return wave_form_file_names

def make_spectogram_sequences_matricies_for_id(person_id):
	wave_file_names = get_wave_file_names_for_id(person_id)
	l = len(wave_file_names)
	num_files = min(l, MAX_FILES_PER_SPEAKER)
	wave_file_names = wave_file_names[0:num_files]
	print "found", len(wave_file_names), 'files for person_id:', person_id
	all_spectogram_sequences = []
	for filename in wave_file_names:
		spectogram_sequences = make_spectogram_sequence_matricies_for_file(filename)
		all_spectogram_sequences.extend(spectogram_sequences)
	return all_spectogram_sequences

def make_all_spectogram_sequence_data():
	X = []
	Y = []
	speaker_ids = SPEAKER_IDS
	Y_labels = range(len(speaker_ids))
	speaker_ids_to_y_labels = {speaker_ids[i] : [1 if Y_labels[i] is j else 0 for j in range(len(speaker_ids))] for i in range(len(speaker_ids))}
	print speaker_ids_to_y_labels
	for speaker_id in speaker_ids:
		print "processing speaker:", speaker_id
		spectograms_sequences_matricies = make_spectogram_sequences_matricies_for_id(speaker_id)
		for seq_i, spectogram_sequence_matrix in enumerate(spectograms_sequences_matricies):
			X.append(spectogram_sequence_matrix)
			Y.append(speaker_ids_to_y_labels[speaker_id])
	X = np.array(X)
	Y = np.array(Y)
	return X, Y


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM, GRU
from sklearn.cross_validation import train_test_split
import random

from keras.callbacks import ModelCheckpoint
'''
saves the model weights after each epoch if the validation loss decreased
'''

def train_nn(X_train, Y_train, X_test, Y_test):
	print "train it!"
	model = create_model_structure()
	print "ok!"
	checkpointer = ModelCheckpoint(filepath=TRAINED_MODEL_FILE_NAME, verbose=1, save_best_only=True)
	model.fit(X_train, Y_train, batch_size=32, nb_epoch=100, verbose=2, validation_data=(X_test, Y_test), callbacks=[checkpointer])
	score = model.evaluate(X_test, Y_test, batch_size=16)
	print score
	print "Saving..."

def process_wav_file_into_input_matrix(waveform):
	return make_spectogram_sequence_matricies_for_file(waveform)

def predict_waveform(waveform_file_name, model=None, model_file_name='trained_models/small_net'):
	input_matrix = process_wav_file_into_input_matrix(waveform_file_name)
	if model == None:
		model = get_model()
		print "Loaded model"
	output = model._predict(input_matrix)
	print "PREDICTED:", output
	return output

def get_activations(model, layer, X_batch):
    X_batch = np.array([X_batch])
    get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(X_batch) # same result as above
    return activations

def get_last_layer(model, example):
	activations = get_activations(model, -2, example)
	return activations[0]


def create_model_structure():
	final_output_nodes = len(SPEAKER_IDS)
	#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
	model = Sequential()
	# Add a mask_zero=True to the Embedding connstructor if 0 is a left-padding value in your data
	max_features = NUM_COMPONENTS_IN_SPECTOGRAM
	model.add(LSTM(max_features, 1024, activation='sigmoid', inner_activation='hard_sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(1024, 512))
	model.add(Dense(512, final_output_nodes))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', class_mode="categorical")
	return model

def predict_waveform_nearest_neighbor(waveform_file_name, model=None, model_file_name=TRAINED_MODEL_FILE_NAME):
	input_matrix = process_wav_file_into_input_matrix(waveform_file_name)
	if model == None:
		model = get_model()
		print "Loaded model"
	last_layer = get_last_layer(model, input_matrix)
	query_vector = last_layer
	candidate_vector_map = get_candidates_vector_map()
	prediction = get_closest_speaker(query_vector, candidate_vector_map)
	print "Prediction:", prediction
	return prediction

def predict_waveform_nearest_neighbor_matrix(input_matrix, model=None, model_file_name=TRAINED_MODEL_FILE_NAME):
	if model == None:
		model = get_model()
		print "Loaded model"
	# import pdb
	# pdb.set_trace()
	last_layer = get_last_layer(model, input_matrix)
	query_vector = last_layer
	candidate_vector_map = get_candidates_vector_map()
	prediction = get_closest_speaker(query_vector, candidate_vector_map)
	print "Prediction:", prediction
	return prediction

def split_data(X, Y, test_size=0.2, random_state=0):
	random.seed(random_state)
	random_indicies = range(len(X))
	random.shuffle(random_indicies)
	print random_indicies
	num_test_samples = int(test_size*len(X))
	random_test_indicies = random_indicies[:num_test_samples]
	random_train_indicies = random_indicies[num_test_samples:]
	X_train, X_test, Y_train, Y_test = [], [], [], []
	for rand_i in random_train_indicies:
		X_train.append(X[rand_i])
		Y_train.append(Y[rand_i])

	for rand_i in random_test_indicies:
		X_test.append(X[rand_i])
		Y_test.append(Y[rand_i])

	X_train, X_test, Y_train, Y_test = np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)

	return X_train, X_test, Y_train, Y_test


def train_on_dataset():
	X, Y = make_all_spectogram_sequence_data()
	test_percentage = 0.2
	print X.shape
	print Y.shape
	X_train, X_test, Y_train, Y_test = split_data(X, Y, test_percentage, 0)
	print "GO!"
	print X_train.shape
	print Y_train.shape
	print len(X_train), len(Y_train)
	print len(X_test), len(Y_test)
	train_nn(X_train, Y_train, X_test, Y_test)

def get_model(filename=TRAINED_MODEL_FILE_NAME):
	global GLOBAL_MODEL
	if GLOBAL_MODEL:
		return GLOBAL_MODEL
	GLOBAL_MODEL = create_model_structure()
	GLOBAL_MODEL.load_weights(filename)
	return GLOBAL_MODEL

def get_vectors_for_file_names(file_names):
	for file_name in files:
		model

def get_all_new_predictions_for_meeting(file_name):
	predictions = []
	input_matricies = process_wav_file_into_input_matrix(file_name)
	ave_sounds = process_wav_file_into_sounds(file_name)
	print "done processing!", len(ave_sounds), len(input_matricies)
	for matrix, sound in zip(input_matricies, ave_sounds):
		print "sound"
		if sound > 0.0001:
			prediction = predict_waveform_nearest_neighbor_matrix(matrix)
			print "did prediction"
			predictions.append(prediction)
		else:
			predictions.append(-1)
	return predictions

def process_wav_file_into_sounds(filename):
	#filename = 'test2.wav'
	fs, raw_data = wavfile.read(open(filename, 'r')) # load the data
	if len(raw_data.T) == 2:
		data = raw_data.T[0]
	else:
		data = raw_data.T
	waveform = [(ele/(2.**14)) for ele in data] # this is 8-bit track, b is now normalized on [-1,1)
	num_chunks_from_waveform = len(waveform)/(NUM_DATAPOINTS_IN_SEQUENCE)
	waveform_chunks = [waveform[index*NUM_DATAPOINTS_IN_SEQUENCE:(index+1)*NUM_DATAPOINTS_IN_SEQUENCE] for index in range(num_chunks_from_waveform)]
	sounds = []
	print "wattup"
	for chunk in waveform_chunks:
		sounds.append(sum([abs(chunker) for chunker in chunk])/(1.0*len(chunk)))
	return sounds

def get_ave_vector_from_file_name(file_name):
	input_matricies = process_wav_file_into_input_matrix(file_name)
	ave_vector = np.zeros(500)
	model = get_model()
	print "Loaded model"
	for matrix in input_matricies:
		last_layer = get_last_layer(model, matrix)
		for i in range(len(last_layer)):
			ave_vector[i] += last_layer[i]
	return ave_vector/len(ave_vector)

SPEAKER_DATA_PATH = 'data/speakers.json'
candidates_vector_map = None
def get_candidates_vector_map():
	global candidates_vector_map
	if candidates_vector_map:
		return candidates_vector_map

	with open('data/speakers.json') as data_file:   
		candidate_map = json.load(data_file)
	candidate_files = {speaker_id : ob["filename"] for speaker_id, ob in candidate_map.iteritems()}
	candidate_vector_averages = {}
	for speaker_id, file_name in candidate_files.iteritems():
		ave_vector = get_ave_vector_from_file_name(file_name)
		candidate_vector_averages[speaker_id] = ave_vector
	return candidate_vector_averages

def distance(x, y):
	return abs(x-y)**2.0

def get_closest_speaker(query, candidates_vector_averages):
	dists = []
	for speaker_id, vector in candidates_vector_averages.iteritems():
		dists.append((distance(vector, query), speaker_id))
	print "min is:", min(dists, key=lambda x: x[0])
	return min(dists, key=lambda x: x[0])[1]

def get_interruptions(predictions):
	past = None
	streak = 0
	interruptions = []
	for prediction in predictions:
		if past == prediction:
			streak += 1
		else:
			if streak > 4:
				if past != -1:
					interruption = (past, prediction)
					interruptions.append(interruption)
			streak = 0
			past = prediction
	return interruptions

#train_on_dataset()
#predict_waveform_nearest_neighbor('test_251.wav', candidates_vector_map, model=None, model_file_name='trained_models/small_net'):
# predicted = predict_waveform('test_251.wav')
#predict = predict_waveform_nearest_neighbor('harini.wav')
#predictions = get_all_new_predictions_for_meeting('harini.wav')
#print predict
#print get_all_new_predictions_for_meeting('harini.wav')


if __name__ == "__main__":
	train_on_dataset()
    #print get_all_new_predictions_for_meeting('harini.wav')

