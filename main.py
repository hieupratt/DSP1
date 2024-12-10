
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
from scipy.io import wavfile
from scipy.io.wavfile import write
import os

class Note():
    def __init__(self,ins,note,start,finish,duration):
        self.start = start
        self.finish = finish
        self.note = note
        self.ins = ins
        self.duration = duration
        self.audio = []
        self.get_music()

    def get_music(self):
        file = self.ins + "_" + self.note + '.wav'
        file = os.path.join(self.ins,file)
        data, sr = librosa.load(file, sr=44100)
        if  self.ins == 'string':
            stretch_factor = 1.2/(self.duration*1.1)
        elif self.ins == 'vibra':
            stretch_factor = 0.7 / (self.duration * 1)
        else:
            stretch_factor = 0.7 / (self.duration * 1.3)
        if stretch_factor != 1:
            self.audio = librosa.effects.time_stretch(data,rate = stretch_factor)
        else:
            self.audio = data
        self.audio = self.normalize(self.audio)

    def normalize(self, audio):
        # Normalize the audio signal between -1 and 1
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

class Music():
    def __init__(self, file_path):
        self.path = file_path
        self.note = []

    def read_note(self):
        data = pd.read_csv(self.path)
        N = data.shape[0]
        for i in range(N):
            ins,note,start,finish,duration = data.iloc[i]
            note = Note(ins, note, start, finish, duration)
            self.note.append(note)

    def create_music(self,name):
        length = max((obj.finish for obj in self.note), default=None)
        song = np.zeros(int((1+length)*44100))
        count_time = np.zeros(int((1+length)*44100))
        for x in self.note:
            start_index = int(44100*x.start)
            song[start_index:start_index + len(x.audio)] += x.audio
            #count_time[start_index:start_index + len(x.audio)] += 1
        #count_time[count_time == 0] = 1
        #song = song / count_time
        max_note = max(np.abs(song))
        song = song /max_note
        #song = np.clip(song, -0.95, 0.95)
        t = np.arange(0,len(song) /44100,1/44100)
        plt.plot(t,song)
        plt.show()
        write(name, 44100, (song * 32767).astype(np.int16))

