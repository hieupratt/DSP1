import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import pyaudio
import struct
from scipy.fftpack import fft
from scipy.ndimage import uniform_filter1d
import tkinter
import librosa
import librosa.display
import os
def time_spectrum(file):
    #phan tich tren mien thoi gian
    x, sr = librosa.load(file)
    onset_frames = librosa.onset.onset_detect(y=x, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames)
    plt.figure(figsize=(12, 8))
    librosa.display.waveshow(y=x, sr=sr, alpha=0.5)
    plt.vlines(onset_times, -1, 1, color='r', linestyle='--', label='Onset Times')
    plt.title('Detected Onsets')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.show()
    print(onset_times.shape)
    return onset_times
def frequency_spectrum(x,frequency_list):
    audio = AudioSegment.from_file(x, format="wav")
    samples = np.array(audio.get_array_of_samples())
    samples = samples.reshape(-1,audio.channels).T
    sample_rate = audio.frame_rate
    sample_len = int(len(samples[0]) / sample_rate)  # do dai audio
    peak_detect = time_spectrum(x) #tao danh sach cac dinh
    print(peak_detect)
    for i in range(len(peak_detect)):
        if peak_detect[i] == peak_detect[-1]:
            new_sample = samples[0][int((peak_detect[i]-0.05) * sample_rate):-1]
        else:
            new_sample = samples[0][int((peak_detect[i]-0.05)*sample_rate):int(peak_detect[i+1]*sample_rate)]
        fft_window_size = len(new_sample)
        blackman_window = np.blackman(len(new_sample))
        new_sample = new_sample*blackman_window
        fft_new_sample = np.fft.fft(new_sample)
        frequency = np.fft.fftfreq(fft_window_size, d = 1/audio.frame_rate)
        sorted_amp = np.sort(np.abs(fft_new_sample[:fft_window_size // 2]))[::-1]
        if sorted_amp[0] < 50:
            continue
        sorted_indices = np.argsort(np.abs(fft_new_sample[:fft_window_size // 2]))[::-1]
        sorted_frequencies = np.abs(frequency[sorted_indices])
        sorted_amp = np.abs(sorted_amp)

        normalized_amp = sorted_amp / sorted_amp[0]
        #print(sorted_frequencies,normalized_amp)
        list = []
        for j,l in zip(sorted_frequencies,normalized_amp):
            if len(list) == 0:
                list.append((j,l))
            else:
                check = True
                for k,m in list:
                    if j < 1500:
                        if np.abs(j-k) < 15 or l < 0.1:
                            check = False
                            break
                    else:
                        if np.abs(j-k) < 30 or l < 0.1:
                            check = False
                            break
                if check:
                    list.append((j,l))
            if len(list) == 15:
                break
        while len(list) < 15:
            list.append((0,0))
        list = np.array(list)
        list = list.T
        x = evaluate(list,frequency_list)
        list = freq_to_note(notes,list)
        print('nốt nhạc đang chơi:',number_to_note(x), 'tại ', peak_detect[i],' s')
        print(list.shape)
        print('[\n[', end = '')
        for j in list[0]:
            print(j,end = ',')
        print('],\n[',end = '')
        np.set_printoptions(precision=3, suppress=True)
        for j in list[1]:
            print(j, end=',')
        print(']\n]')
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(frequency[:fft_window_size // 2], np.abs(fft_new_sample[:fft_window_size // 2]) / sorted_amp[0])
        axs[0].set_xlim(0, 2000)
        axs[0].set_xlabel('Tần số (Hz)')
        axs[0].set_ylabel('Biên độ')

        axs[0].set_title('Biểu đồ tần số của tín hiệu tại thời điểm ' + str(peak_detect[i]) +' s')
        #print("note : ", number_to_note(x), ' at ', peak_detect[i], ' s')
        label = np.array(list[0], dtype=np.str_)
        value = np.array(list[1], dtype=float)
        colors = np.random.rand(len(label), 3)
        axs[1].bar(label,value,color=colors )
        axs[1].set_xlabel('nốt nhạc')
        axs[1].set_ylabel('biên độ')
        plt.tight_layout()
        plt.show()
    # final_array = np.load('music note.npy')
    # print(final_array)
def real_time_audio():
    CHUNK = 1024 * 2 #xu ly bao nhieu frame 1 luc
    FORMAT = pyaudio.paInt16 #byte per sample
    CHANNEL = 1 # dung micro
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format = FORMAT,
                    channels = CHANNEL,
                    rate= RATE,
                    input = True,
                    output = True,
                    frames_per_buffer=CHUNK)

    fig, (ax,ax2) = plt.subplots(2, figsize = (15,8))
    x = np.linspace(0,RATE, CHUNK)
    print(x)
    x_fft = np.linspace(0,RATE, CHUNK)
    line, = ax.plot(x,np.random.rand(CHUNK), '-', lw = 2)
    line_fft, = ax2.plot(x_fft,np.random.rand(CHUNK), '-', lw = 2)
    ax.set_xlim(0,4000)
    ax2.set_xlim(0,4000)
    ax.set_title("non filtered")
    ax2.set_title("filtered")
    while True:
        data = stream.read(CHUNK)  # lay du lieu tu mic
        data_int = np.frombuffer(data, dtype=np.int16)
        filtered_signal = uniform_filter1d(data_int, size = 200)
        Y_FFT = np.fft.fft(data_int)
        y_fft = np.fft.fft(filtered_signal)
        line.set_ydata(np.abs(Y_FFT[0:CHUNK]) /np.max(np.abs(Y_FFT[0:CHUNK])))
        line_fft.set_ydata(np.abs(y_fft[0:CHUNK]) /np.max(np.abs(y_fft[0:CHUNK])))
        sorted_indices = np.argsort(np.abs(y_fft[:CHUNK]))[::-1]
        dominant_frequency = 0
        for i in sorted_indices:
            if 20 <= np.abs(x_fft[i]) <= 4000:
                dominant_frequency = np.abs(x_fft[i])
                break
        print("Dominant Frequency:", dominant_frequency, "Hz")
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.show()
    p.close()

def number_to_note(x):
    if x == -1:
        return 'co cc'
    number = x % 12
    octive = int(x /12)
    note = ''
    #print(number,octive)
    if number == 0:
        note = 'C'
    if number == 1:
        note = 'C#'
    if number == 2:
        note = 'D'
    if number == 3:
        note = 'D#'
    if number == 4:
        note = 'E'
    if number == 5:
        note = 'F'
    if number == 6:
        note = 'F#'
    if number == 7:
        note = 'G'
    if number == 8:
        note = 'G#'
    if number == 9:
        note = 'A'
    if number == 10:
        note = 'A#'
    if number == 11:
        note = 'B'
    note += str(octive +3)
    return note

def evaluate(current_freq, freq_list):
    current_guess = -1 #dự đoán hiện tại
    max_accuracy = 99
    min_dif = 99
    match_freq = 0
    total_freq = 0
    non_zero_freq = 0
    for i in current_freq[0]:
        if i != 0:
            non_zero_freq += 1
    for i in range(len(freq_list)):
        #Số phần tử khác 0 trong nốt nhạc đang dùng để tham chiếu
        a = 0
        for j in freq_list[i][0]:
            if j != 0:
                a = a+1

        accuracy = 0 #Số tần số khớp giữa 2 phần tử
        total_dif = 0 #Tổng khác biệt
        for j in range(0,15):
            if current_freq[0][j] == 0: #nếu là tần số = 0 thì cook
                continue
            if current_freq[0][j] < 1200:
                x = 7
            else:
                x = 14
            check = False
            for k in range(0,15):
                if freq_list[i][0][k] == 0: #nếu là tần số = 0 thì cook
                    continue
                if abs(current_freq[0][j] - freq_list[i][0][k]) < x:
                    if (freq_list[i][1][k] > 0.4 and current_freq[1][j] < 0.2) \
                    or (freq_list[i][1][k] < 0.2 and current_freq[1][j] > 0.2): #Loại bỏ những trường hợp nhiễu
                        continue
                    accuracy += 1
                    total_dif += abs(current_freq[1][j] - freq_list[i][1][k])
                    check = True
                    break

            if not check:
                total_dif += current_freq[1][j]*2
        # đánh giá kết quả dựa trên trung bình độ khác biệt của tất cả các tần số đỉnh
        final_accuracy = (total_dif + (a - accuracy)*2) / (accuracy + (a-accuracy)*2+ (non_zero_freq-accuracy)*2)
        if final_accuracy > 0.8:
            continue
        if max_accuracy > final_accuracy:
            max_accuracy = final_accuracy
            current_guess = i
    if max_accuracy == -99:
        return -1
    print('acc =',1- max_accuracy)
    return current_guess
def freq_to_note(note,list):
    res = []
    for i in range(0,15):
        if list[0][i] == 0:
            continue
        if list[0][i] <= 126:
            multi = 0
        elif list[0][i] <= 252:
            multi = 1
        elif list[0][i] <= 504:
            multi = 2
        elif list[0][i] <= 1020:
            multi = 3
        else:
            multi = 4
        nearest_note = 'unknown'
        nearest_dis = 999
        for j in range(0+12*multi,12+12*multi):
            cur_dis = np.abs(list[0][i] - note[1][j])
            if cur_dis < nearest_dis:
                nearest_note = note[0][j]
                nearest_dis = cur_dis
        res.append([nearest_note,list[1][i]])
    res = np.array(res)
    res = res.T
    return res

notes = [
        ['C2','C#2/Db2','D2','D#2/Eb2','E2','F2','F#2/Gb2','G2','G#2/Ab2','A2','A#2/Bb2','B2',
         'C3','C#3/Db3','D3','D#3/Eb3','E3','F3','F#3/Gb3','G3','G#3/Ab3','A3','A#3/Bb3','B3',
         'C4','C#4/Db4','D4','D#4/Eb4','E4','F4','F#4/Gb4','G4','G#4/Ab4','A4','A#4/Bb4','B4',
         'C5','C#5/Db5','D5','D#5/Eb5','E5','F5','F#5/Gb5','G5','G#5/Ab5','A5','A#5/Bb5','B5',
         'C6','C#6/Db6','D6','D#6/Eb6','E6','F6','F#6/Gb6','G6','G#6/Ab6','A6','A#6/Bb6','B6',],
        [65.40,69.29,73.41,77.78,82.40,87.30,92.49,97.99,103.82,110.00,116.54,123.47,
         130.8, 138.58, 146.82, 155.56, 164.8, 174.6, 184.98, 195.98, 207.64, 220.0, 233.08, 246.94,
         261.6, 277.16, 293.64, 311.12, 329.6, 349.2, 369.96, 391.96, 415.28, 440.0, 466.16, 493.88,
         523.2, 554.32, 587.28, 622.24, 659.2, 698.4, 739.92, 783.92, 830.56, 880.0, 932.32, 987.76,
         1046.4, 1108.64, 1174.56, 1244.48, 1318.4, 1396.8, 1479.84, 1567.84, 1661.12, 1760.0, 1864.64, 1975.52
         ]
    ]

#frequency = np.load('music note.npy')

import numpy as np
import soundfile as sf
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
        self.audio = self.normalize(self.audio) /3

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
        song = np.clip(song, -0.95, 0.95)
        t = np.arange(0,len(song) /44100,1/44100)
        plt.plot(t,song)
        plt.show()
        write(name, 44100, (song * 32767).astype(np.int16))


music = Music("test4.csv")
music.read_note()
music.create_music("output4.wav")
# file = 'vibra/vibra_D#5.wav'
# data, sr = librosa.load(file, sr=44100)
# def audio_trim(data):
#   for i in range(len(data)):
#     if np.abs(data[i]) >0.005 and i > 0.5*44100:
#       return data[i:]
# data = audio_trim(data)
#
# t = np.arange(0,len(data)/44100,1/44100)
# plt.plot(t,data)
# plt.show()
# write(file, 44100, (data[:int(1.2*44100)] * 32767).astype(np.int16))
# folder = r"D:\code\pycharm\project 3\harp"
# for file_name in sorted(os.listdir(folder)):
#     if file_name.endswith('.wav'):  # Lọc chỉ file .wav
#         file_path = os.path.join(folder, file_name)
#         print(f"Processing file: {file_name}")
#
#         # Load file âm thanh
#         data, sr = librosa.load(file_path, sr=44100)
#
#         # Trim dữ liệu âm thanh
#         data = audio_trim(data)
#
#         # Tạo trục thời gian
#         t = np.arange(0, len(data) / sr, 1 / sr)
#         # Vẽ đồ thị
#         plt.figure(figsize=(10, 4))
#         plt.plot(t, data)
#         plt.title(f"Waveform of {file_name}")
#         plt.xlabel("Time (s)")
#         plt.ylabel("Amplitude")
#         plt.grid()
#         plt.show()
#
#         # Lưu lại file với độ dài 1.2 giây
#         output_data = (data[:int(0.7 * sr)] * 32767).astype(np.int16)
#         write(file_path, sr, output_data)
#         print(f"Processed and saved: {file_name}\n")