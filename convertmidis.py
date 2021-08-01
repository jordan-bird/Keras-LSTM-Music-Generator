# Merges all midi tracks into one
# Useful for two-handed piano pieces - very bad for orchestras.

from collections import defaultdict
from mido import Message, MidiFile, MidiTrack, merge_tracks
import os

directory = 'videogame-music'
directoryNew = directory + '-type0/'

if not os.path.exists(directoryNew):
    os.makedirs(directoryNew)

for filename in os.listdir(directory):
	if filename.endswith(".mid"):
		print(filename)
		m = MidiFile(directory + '/' + filename)
		m = merge_tracks(m.tracks)
		mid = MidiFile()
		mid.tracks.append(m)
		mid.save(directoryNew + filename)
