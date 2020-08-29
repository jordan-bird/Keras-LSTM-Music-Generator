""" This module generates notes for a midi file using the
	trained neural network """
import pickle
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation

from keras.layers import Bidirectional, CuDNNLSTM

from keras.optimizers import Adam
from keras.layers import concatenate
from keras.layers import Merge
from keras.layers import Input

from keras import Model

def generate():
	""" Generate a piano midi file """
	#load the notes used to train the model
	with open('data/notes', 'rb') as filepath:
		notes = pickle.load(filepath)
	
	with open('data/durations', 'rb') as filepath:
		durations = pickle.load(filepath)
	
	with open('data/offsets', 'rb') as filepath:
		offsets = pickle.load(filepath)

	# Get all pitch names
	#pitchnames = sorted(set(item for item in notes))
	# Get all pitch names
	#n_vocab = len(set(notes))
	
	
	notenames = sorted(set(item for item in notes))
	n_vocab_notes = len(set(notes))
	network_input_notes, normalized_input_notes = prepare_sequences(notes, notenames, n_vocab_notes)
	
	offsetnames = sorted(set(item for item in offsets))
	n_vocab_offsets = len(set(offsets))
	network_input_offsets, normalized_input_offsets = prepare_sequences(offsets, offsetnames, n_vocab_offsets)
	
	durationames = sorted(set(item for item in durations))
	n_vocab_durations = len(set(durations))
	network_input_durations, normalized_input_durations = prepare_sequences(durations, durationames, n_vocab_durations)

	#model = create_network(network_input_notes, n_vocab_notes, network_input_offsets, n_vocab_offsets, network_input_durations, n_vocab_durations)
	
	model = create_network(normalized_input_notes, n_vocab_notes, normalized_input_offsets, n_vocab_offsets, normalized_input_durations, n_vocab_durations)
	
	
	
	
	

	#network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
	#model = create_network(normalized_input, n_vocab)
	
	prediction_output = generate_notes(model, network_input_notes, network_input_offsets, network_input_durations, notenames, offsetnames, durationames, n_vocab_notes, n_vocab_offsets, n_vocab_durations)
	create_midi(prediction_output)

def prepare_sequences(notes, pitchnames, n_vocab):
	""" Prepare the sequences used by the Neural Network """
	# map between notes and integers and back
	note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

	sequence_length = 100
	network_input = []
	output = []
	for i in range(0, len(notes) - sequence_length, 1):
		sequence_in = notes[i:i + sequence_length]
		sequence_out = notes[i + sequence_length]
		network_input.append([note_to_int[char] for char in sequence_in])
		output.append(note_to_int[sequence_out])

	n_patterns = len(network_input)

	# reshape the input into a format compatible with LSTM layers
	normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
	# normalize input
	normalized_input = normalized_input / float(n_vocab)

	return (network_input, normalized_input)

def create_network(network_input_notes, n_vocab_notes, network_input_offsets, n_vocab_offsets, network_input_durations, n_vocab_durations):
	# Branch of the network that considers notes
	inputNotesLayer = Input(shape=(network_input_notes.shape[1], network_input_notes.shape[2]))
	inputNotes = CuDNNLSTM(
		256,
		input_shape=(network_input_notes.shape[1], network_input_notes.shape[2]),
		return_sequences=True
	)(inputNotesLayer)
	inputNotes = Dropout(0.2)(inputNotes)
	
	# Branch of the network that considers note offset
	inputOffsetsLayer = Input(shape=(network_input_offsets.shape[1], network_input_offsets.shape[2]))
	inputOffsets = CuDNNLSTM(
		256,
		input_shape=(network_input_offsets.shape[1], network_input_offsets.shape[2]),
		return_sequences=True
	)(inputOffsetsLayer)
	inputOffsets = Dropout(0.2)(inputOffsets)
	
	# Branch of the network that considers note duration
	inputDurationsLayer = Input(shape=(network_input_durations.shape[1], network_input_durations.shape[2]))
	inputDurations = CuDNNLSTM(
		256,
		input_shape=(network_input_durations.shape[1], network_input_durations.shape[2]),
		return_sequences=True
	)(inputDurationsLayer)
	#inputDurations = Dropout(0.3)(inputDurations)
	inputDurations = Dropout(0.2)(inputDurations)
	
	#Concatentate the three input networks together into one branch now
	inputs = concatenate([inputNotes, inputOffsets, inputDurations])
	
	# A cheeky LSTM to consider everything learnt from the three separate branches
	x = CuDNNLSTM(512, return_sequences=True)(inputs)
	x = Dropout(0.3)(x)
	x = CuDNNLSTM(512)(x)
	x = BatchNorm()(x)
	x = Dropout(0.3)(x)
	x = Dense(256, activation='relu')(x)
	
	#Time to split into three branches again...
	
	# Branch of the network that classifies the note
	outputNotes = Dense(128, activation='relu')(x)
	outputNotes = BatchNorm()(outputNotes)
	outputNotes = Dropout(0.3)(outputNotes)
	outputNotes = Dense(n_vocab_notes, activation='softmax', name="Note")(outputNotes)
	
	# Branch of the network that classifies the note offset
	outputOffsets = Dense(128, activation='relu')(x)
	outputOffsets = BatchNorm()(outputOffsets)
	outputOffsets = Dropout(0.3)(outputOffsets)
	outputOffsets = Dense(n_vocab_offsets, activation='softmax', name="Offset")(outputOffsets)
	
	# Branch of the network that classifies the note duration
	outputDurations = Dense(128, activation='relu')(x)
	outputDurations = BatchNorm()(outputDurations)
	outputDurations = Dropout(0.3)(outputDurations)
	outputDurations = Dense(n_vocab_durations, activation='softmax', name="Duration")(outputDurations)
	
	# Tell Keras what our inputs and outputs are 
	model = Model(inputs=[inputNotesLayer, inputOffsetsLayer, inputDurationsLayer], outputs=[outputNotes, outputOffsets, outputDurations])
	
	#Adam seems to be faster than RMSProp and learns better too 
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.load_weights('weights-improvement-140-2.7821-bigger.hdf5')

	return model

def generate_notes(model, network_input_notes, network_input_offsets, network_input_durations, notenames, offsetnames, durationames, n_vocab_notes, n_vocab_offsets, n_vocab_durations):
	""" Generate notes from the neural network based on a sequence of notes """
	# pick a random sequence from the input as a starting point for the prediction
	start = numpy.random.randint(0, len(network_input_notes)-1)
	start2 = numpy.random.randint(0, len(network_input_offsets)-1)
	start3 = numpy.random.randint(0, len(network_input_durations)-1)

	int_to_note = dict((number, note) for number, note in enumerate(notenames))
	print(int_to_note)
	int_to_offset = dict((number, note) for number, note in enumerate(offsetnames))
	int_to_duration = dict((number, note) for number, note in enumerate(durationames))

	pattern = network_input_notes[start]
	pattern2 = network_input_offsets[start2]
	pattern3 = network_input_durations[start3]
	prediction_output = []

	# generate notes or chords
	for note_index in range(300):
		note_prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
		predictedNote = note_prediction_input[-1][-1][-1]
		#print(note_prediction_input.shape)
		
		#print(n_vocab_notes)
		#print(n_vocab_offsets)
		#print(n_vocab_durations)
		
		
		note_prediction_input = note_prediction_input / float(n_vocab_notes)
		
		offset_prediction_input = numpy.reshape(pattern2, (1, len(pattern2), 1))
		offset_prediction_input = offset_prediction_input / float(n_vocab_offsets)
		
		duration_prediction_input = numpy.reshape(pattern3, (1, len(pattern3), 1))
		duration_prediction_input = duration_prediction_input / float(n_vocab_durations)

		prediction = model.predict([note_prediction_input, offset_prediction_input, duration_prediction_input], verbose=0)

		index = numpy.argmax(prediction[0])
		#print(index)
		result = int_to_note[index]
		#print(result)
		
		offset = numpy.argmax(prediction[1])
		offset_result = int_to_offset[offset]
		#print("offset")
		#print(offset_result)
		
		duration = numpy.argmax(prediction[2])
		duration_result = int_to_duration[duration]
		#print("duration")
		#print(duration_result)
		
		print("Next note: " + str(int_to_note[predictedNote]) + " - Duration: " + str(int_to_duration[duration]) + " - Offset: " + str(int_to_offset[offset]))
		
		
		#
		prediction_output.append([result, offset_result, duration_result])

		pattern.append(index)
		pattern2.append(offset)
		pattern3.append(duration)
		pattern = pattern[1:len(pattern)]
		pattern2 = pattern2[1:len(pattern2)]
		pattern3 = pattern3[1:len(pattern3)]

	return prediction_output

def create_midi(prediction_output_all):
	""" convert the output from the prediction to notes and create a midi file
		from the notes """
	offset = 0
	output_notes = []
	
	#prediction_output = prediction_output_all
	
	offsets = []
	durations = []
	notes = []
	
	for x in prediction_output_all:
		print(x)
		notes = numpy.append(notes, x[0])
		try:
			offsets = numpy.append(offsets, float(x[1]))
		except:
			num, denom = x[1].split('/')
			x[1] = float(num)/float(denom)
			offsets = numpy.append(offsets, float(x[1]))
			
		durations = numpy.append(durations, x[2])
	
	print("---")
	print(notes)
	print(offsets)
	print(durations)

	# create note and chord objects based on the values generated by the model
	x = 0 # this is the counter
	for pattern in notes:
		# pattern is a chord
		if ('.' in pattern) or pattern.isdigit():
			notes_in_chord = pattern.split('.')
			notes = []
			for current_note in notes_in_chord:
				new_note = note.Note(int(current_note))
				new_note.storedInstrument = instrument.Piano()
				notes.append(new_note)
			new_chord = chord.Chord(notes)
			
			try:
				new_chord.duration.quarterLength = float(durations[x])
			except:
				num, denom = durations[x].split('/')
				new_chord.duration.quarterLength = float(num)/float(denom)
			
			new_chord.offset = offset
			
			output_notes.append(new_chord)
		# pattern is a note
		else:
			new_note = note.Note(pattern)
			new_note.offset = offset
			new_note.storedInstrument = instrument.Piano()
			try:
				new_note.duration.quarterLength = float(durations[x])
			except:
				num, denom = durations[x].split('/')
				new_note.duration.quarterLength = float(num)/float(denom)
			
			output_notes.append(new_note)

		# increase offset each iteration so that notes do not stack
		try:
			offset += offsets[x]
		except:
			num, denom = offsets[x].split('/')
			offset += num/denom
				
		x = x+1

	midi_stream = stream.Stream(output_notes)

	midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
	generate()
