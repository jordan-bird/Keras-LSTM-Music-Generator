# Keras LSTM Music Generator
Generate music with LSTMs in Keras

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


# Training 
Everything in this repo can be run as-is to train on classical piano pieces:

1. Run train.py for a while until note/chord loss is below 1
2. Set line 155 in generate-music.py to the name of the .HDF5 file that was last saved (the model weights)
3. Run generate-music.py to generate a midi file from the model

## Using your own data
Change line 53 in train.py to specify where your .midi files are stored
