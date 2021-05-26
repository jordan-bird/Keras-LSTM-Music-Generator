""" This module prepares midi file data and feeds it to the neural
	network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from keras.layers import Bidirectional, LSTM

from keras.optimizers import Adam
from keras.layers import concatenate
from keras.layers import Input

from keras import Model

def train_network():
	""" Train a Neural Network to generate music """
	notes, offsets, durations = get_notes()

	# Prepare notes 
	n_vocab_notes = len(set(notes))
	network_input_notes, network_output_notes = prepare_sequences(notes, n_vocab_notes)
	
	# Prepare notes 
	n_vocab_offsets = len(set(offsets))
	network_input_offsets, network_output_offsets = prepare_sequences(offsets, n_vocab_offsets)
	
	# Prepare notes 
	n_vocab_durations = len(set(durations))
	network_input_durations, network_output_durations = prepare_sequences(durations, n_vocab_durations)

	model = create_network(network_input_notes, n_vocab_notes, network_input_offsets, n_vocab_offsets, network_input_durations, n_vocab_durations)

	#train(model, network_input_notes, network_output_notes)
	
	train(model, network_input_notes, network_input_offsets, network_input_durations, network_output_notes, network_output_offsets, network_output_durations)

def get_notes():
	""" Get all the notes and chords from the midi files in the ./midi_songs directory """
	notes = []
	offsets = []
	durations = []

	for file in glob.glob("classical-piano-type0/*.mid"):
		midi = converter.parse(file)

		print("Parsing %s" % file)

		notes_to_parse = None

		try: # file has instrument parts
			s2 = instrument.partitionByInstrument(midi)
			notes_to_parse = s2.parts[0].recurse() 
		except: # file has notes in a flat structure
			notes_to_parse = midi.flat.notes
		
		
		offsetBase = 0
		for element in notes_to_parse:
			isNoteOrChord = False
			
			if isinstance(element, note.Note):
				notes.append(str(element.pitch))
				isNoteOrChord = True
			elif isinstance(element, chord.Chord):
				notes.append('.'.join(str(n) for n in element.normalOrder))
				isNoteOrChord = True
			
			if isNoteOrChord:
				#print("Offset Base = " + str(offsetBase))
				#print(element.offset - offsetBase)
				
				offsets.append(str(element.offset - offsetBase))
				
				#print(element.duration)
				#print(element.duration.quarterLength)
				
				durations.append(str(element.duration.quarterLength))
				
				isNoteOrChord = False
				offsetBase = element.offset
				

	with open('data/notes', 'wb') as filepath:
		pickle.dump(notes, filepath)
	
	with open('data/durations', 'wb') as filepath:
		pickle.dump(durations, filepath)
		
	with open('data/offsets', 'wb') as filepath:
		pickle.dump(offsets, filepath)
	
	print(durations)
	return notes, offsets, durations

def prepare_sequences(notes, n_vocab):
	""" Prepare the sequences used by the Neural Network """
	sequence_length = 100

	# get all pitch names
	pitchnames = sorted(set(item for item in notes))

	 # create a dictionary to map pitches to integers
	note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

	network_input = []
	network_output = []

	# create input sequences and the corresponding outputs
	for i in range(0, len(notes) - sequence_length, 1):
		sequence_in = notes[i:i + sequence_length]
		sequence_out = notes[i + sequence_length]
		network_input.append([note_to_int[char] for char in sequence_in])
		network_output.append(note_to_int[sequence_out])

	n_patterns = len(network_input)

	# reshape the input into a format compatible with LSTM layers
	network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
	# normalize input
	network_input = network_input / float(n_vocab)

	network_output = np_utils.to_categorical(network_output)

	return (network_input, network_output)

def create_network(network_input_notes, n_vocab_notes, network_input_offsets, n_vocab_offsets, network_input_durations, n_vocab_durations):
	
	# Branch of the network that considers notes
	inputNotesLayer = Input(shape=(network_input_notes.shape[1], network_input_notes.shape[2]))
	inputNotes = LSTM(
		256,
		input_shape=(network_input_notes.shape[1], network_input_notes.shape[2]),
		return_sequences=True
	)(inputNotesLayer)
	inputNotes = Dropout(0.2)(inputNotes)
	
	# Branch of the network that considers note offset
	inputOffsetsLayer = Input(shape=(network_input_offsets.shape[1], network_input_offsets.shape[2]))
	inputOffsets = LSTM(
		256,
		input_shape=(network_input_offsets.shape[1], network_input_offsets.shape[2]),
		return_sequences=True
	)(inputOffsetsLayer)
	inputOffsets = Dropout(0.2)(inputOffsets)
	
	# Branch of the network that considers note duration
	inputDurationsLayer = Input(shape=(network_input_durations.shape[1], network_input_durations.shape[2]))
	inputDurations = LSTM(
		256,
		input_shape=(network_input_durations.shape[1], network_input_durations.shape[2]),
		return_sequences=True
	)(inputDurationsLayer)
	#inputDurations = Dropout(0.3)(inputDurations)
	inputDurations = Dropout(0.2)(inputDurations)
	
	#Concatentate the three input networks together into one branch now
	inputs = concatenate([inputNotes, inputOffsets, inputDurations])
	
	# A cheeky LSTM to consider everything learnt from the three separate branches
	x = LSTM(512, return_sequences=True)(inputs)
	x = Dropout(0.3)(x)
	x = LSTM(512)(x)
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
	# Useful to try RMSProp though
	
	# LOAD WEIGHTS HERE IF YOU WANT TO CONTINUE TRAINING!
	#model.load_weights(weights_name)

	return model

def train(model, network_input_notes, network_input_offsets, network_input_durations, network_output_notes, network_output_offsets, network_output_durations):
	""" train the neural network """
	filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
	checkpoint = ModelCheckpoint(
		filepath,
		monitor='loss',
		verbose=0,
		save_best_only=True,
		mode='min'
	)
	callbacks_list = [checkpoint]

	model.fit([network_input_notes, network_input_offsets, network_input_durations], [network_output_notes, network_output_offsets, network_output_durations], epochs=1000, batch_size=64, callbacks=callbacks_list, verbose=1)

if __name__ == '__main__':
	#weights_name = 'weights-improvement-41-0.9199-bigger.hdf5'
	train_network()
