from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QSlider, QFileDialog, 
    QVBoxLayout, QWidget, QHBoxLayout, QGridLayout, QFrame, QStackedWidget, QMessageBox, QStyle
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon
import sys
import cv2
import pandas as pd
from scipy.integrate import cumulative_trapezoid
import numpy as np
import os,sys

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class SportsAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize timers before calling initUI()
        self.timer1 = QTimer()

        # Initialize video and IMU-related attributes
        self.video1_path = None
        self.imu_data_path = None
        self.cap1 = None
        self.is_playing1 = False
        self.frame_number1 = 0
        self.imu_data = None
        
        # Call the function to set up UI elements
        self.initUI()
        self.setWindowIcon(QIcon(resource_path('highjump.ico')))

    def initUI(self):
        # Get screen size and set the window to half the screen size
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        self.setGeometry(100, 100, screen_width // 2, screen_height // 2)
        self.setWindowTitle("High Jump Analysis App")

        # Upload Section
        upload_layout = QHBoxLayout()
        self.upload_btn_video1 = QPushButton("Upload Takeoff Video", self)
        self.upload_btn_imu = QPushButton("Upload IMU CSV", self)
        self.start_analysis_btn = QPushButton("Start Analysis", self)
        self.start_analysis_btn.setEnabled(False)

        upload_layout.addWidget(self.upload_btn_video1)
        upload_layout.addWidget(self.upload_btn_imu)
        upload_layout.addWidget(self.start_analysis_btn)

        # Video Layout
        video_layout = QVBoxLayout()

        # Video 1 Layout with StackedWidget for thumbnail and video
        self.video_label1 = QLabel("Video 1")
        self.video_label1.setFixedSize(500, 300)
        self.video_label1.setStyleSheet("background-color: black; color: white;")
        
        self.thumbnail1 = QLabel(self)
        self.thumbnail1.setFixedSize(500, 300)
        
        self.video_stack1 = QStackedWidget(self)
        self.video_stack1.addWidget(self.thumbnail1)  # Add the thumbnail as the first widget
        self.video_stack1.addWidget(self.video_label1)  # Add the actual video label

        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.sliderReleased.connect(self.seek_video1)
        
        slider_controls_layout1 = QHBoxLayout()
        self.play_pause_button1 = QPushButton()
        self.play_pause_button1.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))  # Start with Play icon
        self.play_pause_button1.setIconSize(QSize(30, 30))  # Icon size
         
        slider_controls_layout1.addWidget(self.play_pause_button1)
        slider_controls_layout1.addWidget(self.slider1)

        controls_layout1 = QHBoxLayout()
        self.frame_forward_button1 = QPushButton("Frame Forward")
        self.frame_backward_button1 = QPushButton("Frame Backward")

        controls_layout1.addWidget(self.frame_backward_button1)
        controls_layout1.addWidget(self.frame_forward_button1)

        # Video Layout Adjustments (Centering videos)
        video_layout.addWidget(self.video_stack1, alignment=Qt.AlignHCenter)
        video_layout.addLayout(slider_controls_layout1)
        video_layout.addLayout(controls_layout1)
        

        # Metrics Display
        metrics_layout = QGridLayout()
        self.takeoff_angle_label = QLabel("Takeoff Angle: Not calculated")
        self.horizontal_velocity_label = QLabel("Horizontal Velocity: Not calculated")
        self.jump_height_label = QLabel("Jump Height: Not calculated")
        self.conversion_efficiency_label = QLabel("Conversion Efficiency: Not calculated")

        metrics_layout.addWidget(self.takeoff_angle_label, 0, 0)
        metrics_layout.addWidget(self.horizontal_velocity_label, 0, 1)
        metrics_layout.addWidget(self.jump_height_label, 1, 0)
        metrics_layout.addWidget(self.conversion_efficiency_label, 1, 1)

        # Main Layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(upload_layout)
        main_layout.addLayout(video_layout)
        main_layout.addLayout(metrics_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Connect signals to functions
        self.upload_btn_video1.clicked.connect(self.upload_video1)
        self.upload_btn_imu.clicked.connect(self.upload_imu_csv)
        self.start_analysis_btn.clicked.connect(self.start_analysis)
        
        self.play_pause_button1.clicked.connect(self.toggle_play_pause_video1) 
        self.frame_forward_button1.clicked.connect(self.frame_forward_video1)
        self.frame_backward_button1.clicked.connect(self.frame_backward_video1)

        # Connect timers to update methods
        self.timer1.timeout.connect(self.update_frame1)

    def upload_video1(self):
        self.video1_path, _ = QFileDialog.getOpenFileName(self, "Open Video 1")
        if self.video1_path:
            self.cap1 = cv2.VideoCapture(self.video1_path)
            self.slider1.setMaximum(int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            self.show_thumbnail(self.cap1, self.thumbnail1)
            self.start_analysis_btn.setEnabled(True)

    def upload_imu_csv(self):
        self.imu_data_path, _ = QFileDialog.getOpenFileName(self, "Open IMU CSV File", filter="CSV Files (*.csv)")
        if self.imu_data_path:
            self.imu_data = pd.read_csv(self.imu_data_path)
            
    def show_thumbnail(self, cap, label):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (500, 300))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            label.setPixmap(pixmap)

    def start_analysis(self):
        if self.imu_data is not None:
            # Load the IMU data from the CSV (assuming it's already loaded in self.imu_data)
            df = self.imu_data

            # Define variables from the IMU data
            t = df['elapsed (s)'].values  # Time in seconds
            acc1 = df['x-axis (g)'].values * 9.81  # x-axis acceleration in m/s²
            acc2 = df['y-axis (g)'].values * 9.81  # y-axis acceleration in m/s²
            acc3 = df['z-axis (g)'].values * 9.81  # z-axis acceleration in m/s²

            # Initial velocity assumption
            initial_velocity = 0

            # Calculate velocities by integrating acceleration components over time
            velocity1 = initial_velocity + cumulative_trapezoid(acc1, t, initial=0)  # x-axis velocity
            velocity2 = initial_velocity + cumulative_trapezoid(acc2, t, initial=0)  # y-axis velocity
            velocity3 = initial_velocity + cumulative_trapezoid(acc3, t, initial=0)  # z-axis velocity

            # Calculate the horizontal velocity as the resultant of velocity2 (y-axis) and velocity3 (z-axis)
            horizontal_velocity = np.sqrt(velocity2**2 + velocity3**2)

            # Find the maximum horizontal velocity
            max_horizontal_velocity = np.max(horizontal_velocity)

            # Display the maximum horizontal velocity in the label
            self.horizontal_velocity_label.setText(f"Max Horizontal Velocity: {max_horizontal_velocity:.2f} m/s")

        # Placeholder for other analysis results like takeoff angle, jump height, etc.
        self.takeoff_angle_label.setText("Takeoff Angle: 45°")
        self.jump_height_label.setText("Jump Height: 2.1 m")
        self.conversion_efficiency_label.setText("Conversion Efficiency: 88%")

    def toggle_play_pause_video1(self):
        if self.cap1:
            # If the video has ended, reset to the beginning
            if not self.is_playing1 and self.frame_number1 >= int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)):
                self.frame_number1 = 0
                self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.update_frame1()

            if self.is_playing1:
                # Switch to Pause
                self.is_playing1 = False
                self.timer1.stop()  # Stop the video timer
                self.play_pause_button1.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))  # Set Play icon
            else:
                # Switch to Play
                self.is_playing1 = True
                self.timer1.start(30)  # Start the video timer (adjust interval as needed)
                self.play_pause_button1.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))  # Set Pause icon
                self.video_stack1.setCurrentIndex(1)


    def frame_forward_video1(self):
        if self.cap1:
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number1)
            self.update_frame1()

    def frame_backward_video1(self):
        if self.cap1:
            self.frame_number1 = max(0, self.frame_number1 - 2)
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number1)
            self.update_frame1()

    def update_frame1(self):
        if self.cap1:
            ret, frame = self.cap1.read()
            if ret:
                self.frame_number1 = int(self.cap1.get(cv2.CAP_PROP_POS_FRAMES))
                self.slider1.setValue(self.frame_number1)
                frame = cv2.resize(frame, (500, 300))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(image)
                self.video_label1.setPixmap(pixmap)
            else:
                # Stop the timer if the video ends
                self.timer1.stop()
                self.is_playing1 = False
                self.play_pause_button1.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))


    def seek_video1(self):
        if self.cap1:
            self.frame_number1 = self.slider1.value()
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number1)
            self.update_frame1()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SportsAnalysisApp()
    ex.show()
    sys.exit(app.exec_())
