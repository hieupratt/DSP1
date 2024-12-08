import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
# Đọc file âm thanh gốc
data, sr = librosa.load("music.wav", sr=44800)

data = np.mean(data,axis =1)
plt.plot(data)
plt.show()
# Chọn điểm lặp (ví dụ từ 1s đến 2s)
loop_start = int(1 * sr)
loop_end = int(2 * sr)
loop_segment = data[loop_start:loop_end]

# Lặp lại đoạn âm thanh để đạt thời gian mong muốn (ví dụ, gấp đôi)
repeat_count = 5
looped_audio = np.concatenate([loop_segment] * repeat_count)

# Thêm phần mở đầu và fade-out
output_audio = np.concatenate([data[:loop_start], looped_audio])
fade_length = int(sr * 0.5)  # Fade-out 0.5 giây
fade_out = np.linspace(1, 0, fade_length)
output_audio[-fade_length:] *= fade_out

# Lưu âm thanh kết quả
sf.write("looped_sample.wav", output_audio, sr)