import csv
from PyQt5.QtWidgets import QMessageBox, QMainWindow, QLabel, QApplication, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt
import sys
import os
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


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("dd.ui", self)
        self.selected_column = None  # Lưu trạng thái cột được chọn

        # Danh sách các nốt nhạc từ C2 đến A6
        self.notes = [f"{note}{octave}" for octave in range(2, 6) for note in
                      ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]]

        # Tạo vị trí x cho các nốt
        base_positions = {"C": 20, "D": 50, "E": 80, "F": 110, "G": 140, "A": 170, "B": 200}
        sharp_offsets = {"C#": 40, "D#": 70, "F#": 130, "G#": 160, "A#": 190}

        self.note_positions = {}
        for octave in range(2, 6):
            for note in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
                if note in base_positions:
                    self.note_positions[f"{note}{octave}"] = base_positions[note] + (octave - 2) * 210
                elif note in sharp_offsets:
                    self.note_positions[f"{note}{octave}"] = sharp_offsets[note] + (octave - 2) * 210

        # Cấu hình thời gian bài hát
        self.song_duration = 30
        self.resolution = 10
        self.time_array_length = self.song_duration * self.resolution

        # Khởi tạo mảng tín hiệu và lưu trữ dữ liệu
        self.signal_array = {note: [0] * self.time_array_length for note in self.notes}
        self.used_intervals = {note: [] for note in self.notes}
        self.music_data = []  # Danh sách lưu dữ liệu cho CSV

        # Kết nối nút bấm
        self.p1.clicked.connect(self.spage_1)
        self.p2.clicked.connect(self.spage_2)
        self.p3.clicked.connect(self.spage_3)
        self.p4.clicked.connect(self.spage_4)
        self.p5.clicked.connect(self.spage_5)
        self.p6.clicked.connect(self.spage_6)
        self.p7.clicked.connect(self.spage_7)

        self.p_1.clicked.connect(self.tao)
        self.p_2.clicked.connect(self.sua)
        self.pushButton.clicked.connect(self.create_column)
        self.pushButton_2.clicked.connect(self.export_csv)  # Xuất file CSV
        self.create_time_markers()
        self.pushButton_4.clicked.connect(self.delete_column)  # Nút xóa
        self.pushButton_3.clicked.connect(self.edit_column)  # Nút Sửa
        self.pushButton_5.clicked.connect(self.clear_all)
        self.choose_file.clicked.connect(self.choose_csv_file) # Nút chọn file

        self.spage_1()
        self.tao()
    def spage_1(self):
        self.s1.setCurrentWidget(self.page_1)
        self.highlight_button1(self.p1)

    def spage_2(self):
        self.s1.setCurrentWidget(self.page_2)
        self.highlight_button1(self.p2)

    def spage_3(self):
        self.s1.setCurrentWidget(self.page_3)
        self.highlight_button1(self.p3)

    def spage_4(self):
        self.s1.setCurrentWidget(self.page_4)
        self.highlight_button1(self.p4)

    def spage_5(self):
        self.s1.setCurrentWidget(self.page_5)
        self.highlight_button1(self.p5)

    def spage_6(self):
        self.s1.setCurrentWidget(self.page_6)
        self.highlight_button1(self.p6)

    def spage_7(self):
        self.s1.setCurrentWidget(self.page_7)
        self.highlight_button1(self.p7)

    def tao(self):
        self.s2.setCurrentWidget(self.page1)
        self.highlight_button2(self.p_1)

    def sua(self):
        self.s2.setCurrentWidget(self.page2)
        self.highlight_button2(self.p_2)

    def highlight_button1(self, selected_button):
        # Đặt lại màu cho các nút không được bấm
        buttons = [self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7]
        for button in buttons:
            button.setStyleSheet("background-color: none;")  # Trạng thái mặc định

        # làm nổi bật nút được bấm
        selected_button.setStyleSheet("background-color: lightblue;")

    def highlight_button2(self, selected_button):
        # Đặt lại màu cho các nút không được bấm
        buttons = [self.p_1, self.p_2]
        for button in buttons:
            button.setStyleSheet("background-color: none;")  # Trạng thái mặc định

        # Làm nổi bật nút được bấm
        selected_button.setStyleSheet("background-color: yellow;")
    def is_valid_interval(self, start_time, end_time, note, instrument):
        """Kiểm tra xem khoảng thời gian, nốt nhạc và nhạc cụ có hợp lệ không."""
        for interval in self.used_intervals.get(note, []):
            existing_start, existing_end, existing_instrument = interval
            # Kiểm tra điều kiện chồng lấn thời gian và nhạc cụ khác
            if instrument == existing_instrument and not (end_time <= existing_start or start_time >= existing_end):
                return False
        return True

    def choose_csv_file(self):
        """"Chọn file và xử lý dữ liệu đọc từ file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn file CSV", "", "CSV Files (*.csv)") # Chọn file từ máy tính
        if file_path:
            file_name = file_path.split("/")[-1]  # Tách tên file từ đường dẫn
            self.textEdit_3.setText(file_name)
        else:
            return  # Trở lại giao diện chính nếu không chọn file

        try:
            # Đọc dữ liệu từ file CSV
            data = pd.read_csv(file_path)

            # Kiểm tra cột dữ liệu có hợp lệ không
            required_columns = ["type", "note", "start", "finish", "duration"]
            if not all(column in data.columns for column in required_columns):
                QMessageBox.warning(self, "Lỗi", "File CSV không đúng định dạng yêu cầu!")
                return

            # Xóa dữ liệu cũ và cập nhật lại dữ liệu từ file
            self.music_data.clear()
            self.signal_array = {note: [0] * self.time_array_length for note in self.notes}
            self.used_intervals = {note: [] for note in self.notes}

            for _, row in data.iterrows():
                instrument, note, start, finish, duration = row["type"], row["note"], row["start"], row["finish"], row[
                    "duration"]

                # Kiểm tra giá trị hợp lệ
                if note not in self.notes or not self.is_valid_interval(start, finish, note, instrument):
                    continue  # Bỏ qua các hàng không hợp lệ

                # Cập nhật dữ liệu và tạo cột tín hiệu
                self.music_data.append({
                    "type": instrument,
                    "note": note,
                    "start": start,
                    "finish": finish,
                    "duration": duration
                })

                start_index = int(start * self.resolution)
                end_index = int(finish * self.resolution)
                for i in range(start_index, end_index):
                    self.signal_array[note][i] = 1

                self.used_intervals[note].append((start, finish, instrument))

            # Cập nhật giao diện hiển thị
            self.update_display()
            QMessageBox.information(self, "Thành công", "Dữ liệu từ file CSV đã được cập nhật!")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể đọc file CSV: {e}")
    def create_column(self):
        """Xử lý tín hiệu nốt nhạc và hiển thị trên cột."""
        if not self.textEdit_1.toPlainText() or not self.textEdit_2.toPlainText():
            QMessageBox.warning(self, "Cảnh báo", "Yêu cầu bạn nhập START và END không được bỏ trống!")
            return

        try:
            start_time = float(self.textEdit_1.toPlainText())
            end_time = float(self.textEdit_2.toPlainText())
        except ValueError:
            QMessageBox.warning(self, "Lỗi", "START hoặc END phải là số!")
            return

        if start_time < 0 or end_time > self.song_duration or start_time >= end_time:
            QMessageBox.warning(self, "Lỗi", "Thời gian không hợp lệ!")
            return

        note_base = self.comboBox_2.currentText()
        note_number = self.comboBox_3.currentText()
        note = f"{note_base}{note_number}"

        if note not in self.notes:
            QMessageBox.warning(self, "Lỗi", f"Nốt nhạc '{note}' không hợp lệ!")
            return

        instrument = self.comboBox.currentText()

        # Kiểm tra nốt nhạc và khoảng thời gian
        if not self.is_valid_interval(start_time, end_time, note, instrument):
            QMessageBox.warning(self, "Lỗi", "Khoảng thời gian hoặc nốt nhạc bị chồng lấn!")
            return

        start_index = int(start_time * self.resolution)
        end_index = int(end_time * self.resolution)
        for i in range(start_index, end_index):
            self.signal_array[note][i] = 1

        self.used_intervals[note].append((start_time, end_time, instrument))
        self.music_data.append({
            "type": instrument,
            "note": note,
            "start": start_time,
            "finish": end_time,
            "duration": end_time - start_time
        })

        selected_column = self.selected_column

        self.update_display()

    def clear_all(self):
        """Xóa dữ liêu các nốt đã tạo để tạo bản nhạc mới"""
        if self.selected_column:
            # Xóa cột đang chọn nếu có
            # self.selected_column.setStyleSheet(
            #     self.selected_column.styleSheet().replace("border: 2px solid red;", "")
            # )
            self.selected_column = None
        self.music_data.clear()
        self.signal_array = {note: [0] * self.time_array_length for note in self.notes}
        self.used_intervals = {note: [] for note in self.notes}
        self.update_display()
        self.label_11.setText("")
    def update_display(self):
        """Cập nhật hiển thị cột tín hiệu theo từng nhạc cụ riêng biệt."""
        for sc_widget in [self.sc1, self.sc2, self.sc3, self.sc4, self.sc5, self.sc6, self.sc7]:

            sc_widget.setMinimumHeight(self.song_duration * 100)
            # sc_widget.updateGeometry()
            self.sa_1.verticalScrollBar().setValue(self.sa_1.verticalScrollBar().maximum())
            self.sa_2.verticalScrollBar().setValue(self.sa_2.verticalScrollBar().maximum())
            self.sa_3.verticalScrollBar().setValue(self.sa_3.verticalScrollBar().maximum())
            self.sa_4.verticalScrollBar().setValue(self.sa_4.verticalScrollBar().maximum())
            self.sa_5.verticalScrollBar().setValue(self.sa_5.verticalScrollBar().maximum())
            self.sa_6.verticalScrollBar().setValue(self.sa_6.verticalScrollBar().maximum())
            self.sa_7.verticalScrollBar().setValue(self.sa_7.verticalScrollBar().maximum())
            for child in sc_widget.findChildren(QLabel):
                if child.toolTip():  # Chỉ xóa các cột tín hiệu, giữ lại thời gian
                    child.deleteLater()

        sharp_notes = [data for data in self.music_data if "#" in data["note"]]
        natural_notes = [data for data in self.music_data if "#" not in data["note"]]

        # Hàm phụ để tạo cột
        def create_column1(note_data):
            if self.selected_column:
                self.selected_column.setStyleSheet(
                    self.selected_column.styleSheet().replace("border: 2px solid red;", "")
                )
            self.selected_column = None
            note = note_data["note"]
            start_time = note_data["start"]
            finish_time = note_data["finish"]
            instrument = note_data["type"]

            if instrument == "CFX Grand" or instrument == "cfx":
                sc_widget = self.sc1
            elif instrument == "Bosendorfer" or instrument == "bor":
                sc_widget = self.sc2
            elif instrument == "Harpsichord" or instrument == "harp":
                sc_widget = self.sc3
            elif instrument == "Vibraphone" or instrument == "vibra":
                sc_widget = self.sc4
            elif instrument == "Pipe Organ" or instrument == "pipe":
                sc_widget = self.sc5
            elif instrument == "Jazz Organ" or instrument == "jazz":
                sc_widget = self.sc6
            elif instrument == "Strings" or instrument == "string":
                sc_widget = self.sc7
            else:
                QMessageBox.warning(self, "Lỗi", "Nhạc cụ không hợp lệ!")
                return

            x_position = self.note_positions[note]
            y_start = self.song_duration * 100 - int(start_time * self.resolution) * 10
            y_end = self.song_duration * 100 - int(finish_time * self.resolution) * 10
            column_height = y_start - y_end

            column = QLabel(sc_widget)
            if "#" in note:
                column.setStyleSheet(
                    """
                    background-color: blue;
                    border-radius : 8px; 
                    """
                )
                column.setFixedSize(20, column_height)
            else:
                column.setStyleSheet(
                    """
                    background-color: green;
                    border-radius : 10px; 
                    """
                )
                column.setFixedSize(30, column_height)

            column.move(x_position, y_end)
            column.setToolTip(
                f"Instrument: {instrument}\n"
                f"Note: {note}\n"
                f"Start: {start_time}\n"
                f"Finish: {finish_time}\n"
                f"Duration: {round(finish_time - start_time,1)}"
            )

            def select_column(event):
                # Kiểm tra nếu có cột đã được chọn trước đó
                if self.selected_column and self.selected_column != column:
                    # Bỏ chọn cột cũ
                    self.selected_column.setStyleSheet(
                        self.selected_column.styleSheet().replace("border: 2px solid red;", "")
                    )
                    # Nếu cột hiện tại đã được chọn, bỏ chọn khi nhấn lần nữa
                if self.selected_column == column:
                    self.selected_column.setStyleSheet(
                                 self.selected_column.styleSheet().replace("border: 2px solid red;", "")
                             )
                    self.selected_column = None

                    self.label_11.setText("")  # Xóa thông tin trên label_11
                    return
                # Cập nhật cột được chọn (cột mới)
                self.selected_column = column
                column.setStyleSheet(column.styleSheet() + "border: 2px solid red;")

                # Hiển thị thông tin nốt nhạc trên label_11
                tooltip = column.toolTip()

                # Thay thế ký hiệu # để tránh lỗi hiển thị HTML
                tooltip = tooltip.replace("#", "&#35;")

                # Xóa chữ "Selected Note:" trong nội dung hiển thị
                self.label_11.setText(tooltip.replace("\n", "<br>"))

            column.mousePressEvent = select_column

            column.show()

        # Tạo cột cho các nốt không có dấu # trước
        for data in natural_notes:
            create_column1(data)

        # Tạo cột cho các nốt có dấu # sau
        for data in sharp_notes:
            create_column1(data)



    def export_csv(self):
        """Xuất dữ liệu ra file CSV với tên tùy chỉnh."""
        if not self.music_data:
            QMessageBox.warning(self, "Lỗi", "Không có dữ liệu để xuất!")
            return

        # Lấy tên tệp từ textEdit_3
        file_name = self.textEdit_3.toPlainText().strip()
        if not file_name:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập tên tệp trước khi xuất file!")
            return

        # Thêm đuôi ".csv" nếu chưa có
        if not file_name.endswith(".csv"):
            file_name += ".csv"

        # Kiểm tra trùng lặp tệp
        if os.path.exists(file_name):
            reply = QMessageBox.question(self, "Trùng Tên", "Tên file này đã tồn tại. Bạn có muốn ghi đè lên không?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        # Lưu tệp CSV
        try:
            with open(file_name, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["type", "note", "start", "finish", "duration"])
                writer.writeheader()
                for row in self.music_data:
                    row_copy = row.copy()
                    if row_copy["type"] == "CFX Grand":
                        row_copy["type"] = "cfx"
                    elif row_copy["type"] == "Bosendorfer":
                        row_copy["type"] = "bor"
                    elif row_copy["type"] == "Harpsichord":
                        row_copy["type"] = "harp"
                    elif row_copy["type"] == "Vibraphone":
                        row_copy["type"] = "vibra"
                    elif row_copy["type"] == "Pipe Organ":
                        row_copy["type"] = "pipe"
                    elif row_copy["type"] == "Jazz Organ":
                        row_copy["type"] = "jazz"
                    elif row_copy["type"] == "Strings":
                        row_copy["type"] = "string"
                    writer.writerow(row_copy)

            QMessageBox.information(self, "Thành công", f"Dữ liệu đã được lưu vào tệp '{file_name}'!")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể lưu tệp: {e}")

        music = Music(file_name)
        music.read_note()
        file_name = file_name[:-4] + ".wav"
        music.create_music(file_name)
    def create_time_markers(self):
        """Hiển thị mốc thời gian ở bên trái màn hình."""
        for widget in [self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.w7]:
            widget.setFixedSize(20, self.song_duration * 100)  # Cố định chiều cao ban đầu
            widget.setStyleSheet("background-color: lightgray;")  # Tùy chỉnh màu nền

            # Xóa các mốc cũ (nếu có)
            for child in widget.findChildren(QLabel):
                child.deleteLater()

            # Thêm các mốc thời gian
            for t in range(0, self.time_array_length + 1, self.resolution // 2):
                label = QLabel(widget)
                label.setStyleSheet("color: black; font-size: 10px;")
                label.setText(f"{t / self.resolution:.1f}s")  # Hiển thị thời gian
                label.adjustSize()
                y = self.song_duration * 100 - t * 10 - label.height() // 2  # Căn chỉnh vị trí
                label.move(0, y)
                label.show()

            # Cập nhật kích thước widget bên trong để scroll bar hoạt động
            widget.setMinimumHeight(self.song_duration * 100)
            widget.updateGeometry()

    def delete_column(self):
        if not self.selected_column:
            QMessageBox.warning(self, "Lỗi", "Không có cột nào được chọn để xóa!")
            return

        tooltip = self.selected_column.toolTip()
        details = dict(line.split(": ") for line in tooltip.split("\n"))
        note = details["Note"]
        start_time = float(details["Start"])
        end_time = float(details["Finish"])
        instrument = details["Instrument"]

        self.selected_column.deleteLater()
        self.selected_column = None
        # Xóa khỏi `used_intervals`
        self.used_intervals[note] = [
            interval for interval in self.used_intervals[note]
            if not (interval[0] == start_time and interval[1] == end_time and interval[2] == instrument)
        ]

        self.music_data = [
            data for data in self.music_data
            if not (data["note"] == note and data["start"] == start_time and data["finish"] == end_time and data[
                "type"] == instrument)
        ]

        start_index = int(start_time * self.resolution)
        end_index = int(end_time * self.resolution)
        for i in range(start_index, end_index):
            self.signal_array[note][i] = 0

        self.update_display()

    def edit_column(self):
        """Sửa cột nốt nhạc được chọn."""
        if not self.selected_column:
            QMessageBox.warning(self, "Lỗi", "Không có cột nào được chọn để sửa!")
            return

        # Lấy giá trị từ các trường đầu vào
        new_note_base = self.comboBox_4.currentText()
        new_note_number = self.comboBox_6.currentText()
        new_note = f"{new_note_base}{new_note_number}"

        try:
            new_start_time = float(self.textEdit_4.toPlainText())
            new_end_time = float(self.textEdit_5.toPlainText())
        except ValueError:
            QMessageBox.warning(self, "Lỗi", "START hoặc END phải là số!")
            return

        if new_start_time < 0 or new_end_time > self.song_duration or new_start_time >= new_end_time:
            QMessageBox.warning(self, "Lỗi", "Thời gian không hợp lệ!")
            return

        # Lấy thông tin nhạc cụ từ tooltip của cột được chọn
        tooltip = self.selected_column.toolTip()
        details = dict(line.split(": ") for line in tooltip.split("\n"))
        instrument = details["Instrument"]

        # Kiểm tra sự trùng lặp
        for note_data in self.music_data:
            if (
                    note_data["type"] == instrument
                    and note_data["note"] == new_note
                    and not (new_end_time <= note_data["start"] or new_start_time >= note_data["finish"])
            ):
                QMessageBox.warning(self, "Lỗi", "Nốt nhạc bị trùng với một nốt khác!")
                return

        # Xóa nốt cũ
        self.delete_column()

        # Thêm nốt mới
        start_index = int(new_start_time * self.resolution)
        end_index = int(new_end_time * self.resolution)

        for i in range(start_index, end_index):
            self.signal_array[new_note][i] = 1

        self.music_data.append({
            "type": instrument,
            "note": new_note,
            "start": new_start_time,
            "finish": new_end_time,
            "duration": new_end_time - new_start_time
        })

        self.update_display()


app = QApplication(sys.argv)
app.setStyleSheet("""
    QToolTip {
        background-color: none;
        color: black;
    }
""")
main_window = MainWindow()
main_window.show()
sys.exit(app.exec_())
