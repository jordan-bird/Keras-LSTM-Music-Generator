# Keras LSTM Music Generator
Learn to generate music with LSTM neural networks in Keras

# Requirements
Python 3.x

Keras and music21 

GPU strongly recommended for training

# Info
After reading Sigurður Skúli's towards data science article ['How to Generate Music using a LSTM Neural Network in Keras'](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5) - I was astounded at how well LSTM classification networks were at predicting notes and chords in a sequence, and ultimately then how they could generate really nice music. 

Sigurður's approach had some really nice and useful functions for parsing the data, creating dictionaries to translate between notes and class labels, and then using the trained model to generate pieces of music. 

The main limitations of this work is that notes generated are all the same length and offset from one another, so music can sound quite unnatural sometimes. In this extension, we instead use the Keras Functional API (instead of a Sequential model) to branch the neural network to consider multiple timeseries from the music. They are:

1. The Notes and Chords in the sequence (just referred to as 'notes' from here on)
2. The offsets of the note from the previous one (offset of the note from the start of the midi minus the current base (previous value))
3. The durations of the notes in the sequence

The above also serve as three separate outputs to the network.

Thus, three tasks are trained. Classifying the next note, its offset, and its duration.


# Model Diagram
This is the best model I have found so far:
![LSTM Model Diagram](model_plot.png)
Note: the number of outputs for the three final layers differ depending on what was detected during parsing of the files. For example, if you parse 100 midi files consisting of only three chords, the notes_output layer will only have three outputs. Thus, depending on what kind of music you're training on, the number of softmax neurons will change dynamically. 


# Training 
Everything in this repo can be run as-is to train on classical piano pieces:

1. Run train.py for a while until note/chord loss is below 1
2. Set line 155 in generate-music.py to the name of the .HDF5 file that was last saved (the model weights)
3. Run generate-music.py to generate a midi file from the model

## Using your own data
Change line 53 in train.py to specify where your .midi files are stored

# To Do
1. Upload some more examples of music
2. Try and experiment with smaller models that are quicker to train but still produce good results
3. Try out some different loss functions, Adam seems best so far
4. Add a fourth input and output branch for instrument classification and prediction for the note.
5. Maybe experiment with adding a genre classification network branch so the model doesn't need curated data as input

# Changing instruments
I hope to update the model to learn to predict the instrument, but at the moment I just use https://onlinesequencer.net/ if I want to hear it played by something other than a piano

You can train the network on any instrument, all it cares about are the notes and their offset and duration. But, that said, the network will set each note's instrument as piano. This can be changed via lines 265 and 282 which set notes and chords to piano respectively in generate-music.py
