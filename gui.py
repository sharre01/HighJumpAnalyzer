from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QSlider, QFileDialog, QStyle,
    QVBoxLayout, QWidget, QHBoxLayout, QStackedWidget, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap

import cv2
import pandas as pd
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from processing import (
    initial_IMU_reading, create_video_df, synch_IMU_video, pose_estimation, draw_keypoints
)
from utils import add_shadow
import ffmpeg

class SportsAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize attributes
        self.video_size = (1280, 720)
        self.timer = QTimer()
        self.video_path = None
        self.quart_data_path = None
        self.cap = None
        self.is_playing = False
        self.frame_number = 0
        self.imu_data = None
        self.quart_data = None
        self.global_acceleration_df = None
        self.metadata = None
        self.video_df = None
        self.synch_offset = None
        self.synched = False
        self.frame_jump = None
        self.sidebar_is_expanded = True 
        self.sidebar_animation = None     

        # Call the function to set up UI elements
        self.initUI()

    def initUI(self):
        # Main layout
        main_layout = QHBoxLayout()

        # Sidebar
        self.sidebar = QWidget()
        self.sidebar.setMaximumWidth(280) 
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setAlignment(Qt.AlignTop)
        sidebar_layout.setSpacing(10)
        

        # Buttons in the sidebar
        self.upload_btn_video = QPushButton("Upload Takeoff Video")
        self.upload_btn_imu = QPushButton("Upload Linear Acceleration CSV")
        self.upload_btn_quart = QPushButton("Upload Quaternion CSV")
        self.start_analysis_btn = QPushButton("Plot IMU Data")
        self.start_vidanalysis_btn = QPushButton("Pose Estimation")

        # Add buttons to the sidebar
        for button in [self.upload_btn_video, self.upload_btn_imu, self.upload_btn_quart, self.start_analysis_btn, self.start_vidanalysis_btn]:
            sidebar_layout.addWidget(button)

        self.sidebar.setLayout(sidebar_layout)

        # Toggle button
        self.toggle_btn = QPushButton("☰")  # Hamburger menu icon
        self.toggle_btn.setFixedSize(50, 50)
        self.toggle_btn.clicked.connect(self.toggle_sidebar)

        # Sidebar + Toggle button
        sidebar_toggle_layout = QVBoxLayout()
        sidebar_toggle_layout.setAlignment(Qt.AlignTop)
        sidebar_toggle_layout.addWidget(self.toggle_btn, alignment=Qt.AlignTop | Qt.AlignHCenter)
        sidebar_toggle_layout.addWidget(self.sidebar)

        sidebar_container = QWidget()
        sidebar_container.setLayout(sidebar_toggle_layout)

        # Video and Plot Area
        self.video_label = QLabel()
        self.video_label.setFixedSize(*self.video_size)
        self.video_label.setStyleSheet("background-color: black; color: white;")

        self.thumbnail1 = QLabel(self)
        self.thumbnail1.setFixedSize(*self.video_size)
        
        self.video_stack = QStackedWidget(self)
        self.video_stack.addWidget(self.thumbnail1)  # Add the thumbnail as the first widget
        self.video_stack.addWidget(self.video_label)  # Add the actual video label
        

        # Video Controls
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderReleased.connect(self.seek_video)
        self.play_pause_button = QPushButton()
        self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_pause_button.setIconSize(QSize(30, 30))
        self.play_pause_button.clicked.connect(self.toggle_play_pause_video)

        # Frame and Jump controls
        self.frame_forward_button = QPushButton()
        self.frame_forward_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        self.frame_forward_button.setIconSize(QSize(24, 24))

        self.frame_backward_button = QPushButton()
        self.frame_backward_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        self.frame_backward_button.setIconSize(QSize(24, 24))

        self.jump_forward_button = QPushButton()
        self.jump_backward_button = QPushButton()
        self.jump_forward_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.jump_backward_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))

        self.jump_forward_button.setEnabled(False)
        self.jump_backward_button.setEnabled(False)

        self.frame_forward_button.setToolTip("Frame Forward")
        self.frame_backward_button.setToolTip("Frame Backward")
        self.jump_forward_button.setToolTip("Next Jump")
        self.jump_backward_button.setToolTip("Previous Jump")
        self.frame_forward_button.clicked.connect(self.frame_forward_video)
        self.frame_backward_button.clicked.connect(self.frame_backward_video)
        self.jump_forward_button.clicked.connect(self.jump_forward_video)
        self.jump_backward_button.clicked.connect(self.jump_backward_video)

        # Plot Area
        self.figure = Figure(figsize=(20, 2.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Vertical linear acceleration")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Acceleration (g)")
        self.figure.tight_layout()
        self.canvas.draw()

        self.thumbnail2 = QLabel(self)
        self.fig_stack = QStackedWidget(self)
        self.fig_stack.addWidget(self.thumbnail2)  # Add the thumbnail as the first widget
        self.fig_stack.addWidget(self.canvas)  # Add the actual canvas label
        self.fig_stack.setCurrentWidget(self.canvas)  # Display the canvas

        # Plot Container
        plot_widget = QWidget()
        plot_layout = QVBoxLayout()
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.fig_stack, alignment=Qt.AlignHCenter)
        plot_widget.setLayout(plot_layout)

        # Metrics Section
        self.rk_measure_label = QLabel("Right Knee Angle: No data")
        self.lk_measure_label = QLabel("Left Knee Angle: No data")
        self.hv_measure_label = QLabel("Max Horizontal Velocity: No data")
        # Set maximum width
        for label in [self.rk_measure_label, self.lk_measure_label, self.hv_measure_label]:
            label.setMaximumWidth(200)

        metrics_layout = QVBoxLayout()
        metrics_layout.addWidget(self.rk_measure_label)
        metrics_layout.addWidget(self.lk_measure_label)
        metrics_layout.addWidget(self.hv_measure_label)
        # Buttons and slider
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(self.slider)
        controls_layout.addWidget(self.frame_backward_button)
        controls_layout.addWidget(self.frame_forward_button)
        controls_layout.addWidget(self.jump_backward_button)
        controls_layout.addWidget(self.jump_forward_button)

        # Plot and metrics
        plot_met_layout = QHBoxLayout()
        plot_met_layout.addWidget(plot_widget)
        plot_met_layout.addLayout(metrics_layout)

        # Display Layout
        display_layout = QVBoxLayout()
        display_layout.addWidget(self.video_stack, alignment=Qt.AlignCenter)
        display_layout.addLayout(controls_layout)
        display_layout.addLayout(plot_met_layout)
        # Combine Sidebar and Display
        main_layout.addWidget(sidebar_container)
        main_layout.addLayout(display_layout)
        # Set stretch factors to allow resizing
        main_layout.setStretch(0, 0)  # Sidebar container
        main_layout.setStretch(1, 1)  # Main display area

        # Set Main Layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.setWindowTitle("High Jump Analyzer")
        self.resize(1000, 600)

        # Initialize timers
        self.timer.timeout.connect(self.update_frame)
        self.timer.timeout.connect(self.update_IMU_plot)

        # Connect signals to functions
        self.upload_btn_video.clicked.connect(self.upload_video)
        self.upload_btn_imu.clicked.connect(self.upload_imu_csv)
        self.upload_btn_quart.clicked.connect(self.upload_quart_csv)
        self.start_analysis_btn.clicked.connect(self.start_analysis)
        self.start_vidanalysis_btn.clicked.connect(self.pose_est_analysis)

        # Buttons
        self.upload_btn_video.setObjectName("upload")
        self.upload_btn_imu.setObjectName("upload")
        self.upload_btn_quart.setObjectName("upload")
        self.start_analysis_btn.setObjectName("analysis")
        self.start_vidanalysis_btn.setObjectName("analysis")
        self.toggle_btn.setObjectName("toggle_button")

        # Labels
        self.rk_measure_label.setObjectName("results")
        self.lk_measure_label.setObjectName("results")
        self.hv_measure_label.setObjectName("results")

        # Containers
        container.setObjectName("main_window") 
        self.video_label.setObjectName("video_display")
        plot_widget.setObjectName("plot_container")

        self.showMaximized()

    def toggle_sidebar(self):
        if self.sidebar_is_expanded:
            start_width = self.sidebar.width()
            end_width = 0
        else:
            start_width = self.sidebar.width()
            end_width = 220

        if not self.sidebar_animation:
            self.sidebar_animation = QPropertyAnimation(self.sidebar, b"maximumWidth")
            self.sidebar_animation.setDuration(300)
            self.sidebar_animation.setEasingCurve(QEasingCurve.InOutQuart)

        self.sidebar_animation.setStartValue(start_width)
        self.sidebar_animation.setEndValue(end_width)
        self.sidebar_animation.start()

        self.sidebar_is_expanded = not self.sidebar_is_expanded

        # Update the toggle button text or icon
        if self.sidebar_is_expanded:
            self.toggle_btn.setText("☰")
        else:
            self.toggle_btn.setText("☰")

    def upload_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video 1")
        if self.video_path:
            self.metadata = ffmpeg.probe(self.video_path)
            self.cap = cv2.VideoCapture(self.video_path)
            self.slider.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            self.show_message("Video 1 Uploaded", f"Video 1 uploaded successfully: {self.video_path}")
            self.show_thumbnail(self.cap, self.video_label)
            self.start_vidanalysis_btn.setEnabled(True)

    def upload_imu_csv(self):
        self.imu_data_path, _ = QFileDialog.getOpenFileName(self, "Open IMU CSV File", filter="CSV Files (*.csv)")
        if self.imu_data_path:
            self.imu_data = pd.read_csv(self.imu_data_path)
            self.show_message("IMU CSV Uploaded", f"IMU CSV uploaded successfully: {self.imu_data_path}")
            if self.quart_data is not None:
                self.start_analysis_btn.setEnabled(True)

    def upload_quart_csv(self):
        self.quart_data_path, _ = QFileDialog.getOpenFileName(self, "Open Quaternion CSV File", filter="CSV Files (*.csv)")
        if self.quart_data_path:
            self.quart_data = pd.read_csv(self.quart_data_path)
            self.show_message("Quaternion CSV Uploaded", f"Quaternion CSV uploaded successfully: {self.quart_data_path}")
            if self.imu_data is not None:
                self.start_analysis_btn.setEnabled(True)

    def show_message(self, title, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle(title)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def show_thumbnail(self, cap, label):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, self.video_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            label.setPixmap(pixmap)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame

    def start_analysis(self):
        if self.imu_data is not None and self.quart_data is not None:
            self.global_acceleration_df = initial_IMU_reading(self.imu_data, self.quart_data)
            if self.metadata is not None:
                self.video_df = create_video_df(self.metadata)
                self.synched, self.synch_offset = synch_IMU_video(self.global_acceleration_df, self.video_df)
                self.update_IMU_plot()

    def update_IMU_plot(self):
        if self.global_acceleration_df is not None:
            # Check if the plot has been initialized
            if not hasattr(self, 'imu_plot_initialized') or not self.imu_plot_initialized:
                # Initial plot setup
                self.ax.plot(
                    self.global_acceleration_df['elapsed'], 
                    self.global_acceleration_df['g_z'], 
                    linestyle='-', 
                    color=(59/255, 171/255, 154/255)
                )
                self.ax.set_title("Acceleration upwards")
                self.ax.set_xlabel("Time (s)")
                self.ax.set_ylabel("Vertical Acceleration (g)")

                # Highlight jump regions
                find_jump_df = self.global_acceleration_df.loc[self.global_acceleration_df['g_z'] > 2.2]
                jump_list = []
                time = -2
                for index, row in find_jump_df.iterrows():
                    if row['elapsed'] - time > 2:
                        self.ax.axvspan(
                            xmin=row['elapsed'] - 1.5, 
                            xmax=row['elapsed'] + 0.2, 
                            color=(49/255, 113/255, 149/255), 
                            alpha=0.3  # Transparency
                        )
                        time = row['elapsed']
                        jump_list.append(time - 1.5)

                # Synchronize video frames with jumps
                if self.video_df is not None and self.synched:
                    self.frame_jump = []
                    for jump_time in jump_list:
                        closest_frame = (self.video_df['Elapsed time (s)'] - self.synch_offset - jump_time).abs().idxmin()
                        self.frame_jump.append(closest_frame)
                    self.frame_jump = np.array(self.frame_jump)
                    self.jump_forward_button.setEnabled(True)
                    self.jump_backward_button.setEnabled(True)

                # Add vertical line to indicate current video frame
                if self.synched:
                    current_time = self.video_df['Elapsed time (s)'][self.frame_number] - self.synch_offset
                    self.vertical_line = self.ax.axvline(
                        x=current_time, 
                        color="red"
                    )
                else:
                    self.vertical_line = None

                self.imu_plot_initialized = True  # Mark plot as initialized
                self.canvas.draw()
            else:
                # Update only the vertical line to improve efficiency
                if self.synched and self.vertical_line is not None:
                    current_time = self.video_df['Elapsed time (s)'][self.frame_number] - self.synch_offset
                    self.vertical_line.set_xdata([current_time])  # Update the x-coordinate
                    self.canvas.draw()

    def pose_est_analysis(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
            ret, frame = self.cap.read()
            if ret:
                keypoints_with_scores = pose_estimation(frame)
                frame_with_keypoints, right_knee_angle, left_knee_angle = draw_keypoints(
                    frame, keypoints_with_scores[0], 0.05
                )
                self.display_frame(frame_with_keypoints)

                # Update the labels with the calculated angles
                if right_knee_angle is not None:
                    self.rk_measure_label.setText(f"{round(right_knee_angle, 3)} degrees")
                    self.rk_measure_label.setStyleSheet("color: rgb(28,99,140);")
                else:
                    self.rk_measure_label.setText("No data")

                if left_knee_angle is not None:
                    self.lk_measure_label.setText(f"{round(left_knee_angle, 3)} degrees")
                    self.lk_measure_label.setStyleSheet("color: rgb(122, 15, 15);")
                else:
                    self.lk_measure_label.setText("No data")

    def display_frame(self, frame):
        frame = cv2.resize(frame, self.video_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap)

    # Video playback methods (play, pause, frame forward, frame backward, update frames)
    def toggle_play_pause_video(self):
        if self.cap:
            # If the video has ended, reset to the beginning
            if not self.is_playing and self.frame_number >= int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                self.frame_number = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.update_frame()

            if self.is_playing:
                # Switch to Pause
                self.is_playing = False
                self.timer.stop()  # Stop the video timer
                self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))  # Set Play icon
            else:
                # Switch to Play
                self.is_playing = True
                self.timer.start(30)  # Start the video timer (adjust interval as needed)
                self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))  # Set Pause icon
                self.video_stack.setCurrentIndex(1)

    def frame_forward_video(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
            self.update_frame()
            self.update_IMU_plot()

    def frame_backward_video(self):
        if self.cap:
            self.frame_number -= 2
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
            self.update_frame()
            self.update_IMU_plot()

    def jump_backward_video(self):
        backward_jumps = self.frame_jump[self.frame_jump-self.frame_number<-1]
        if self.cap and self.synched and len(backward_jumps)>0:
            self.frame_number = max(backward_jumps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
            self.update_frame()
            self.update_IMU_plot()

    def jump_forward_video(self):
        forward_jumps = self.frame_jump[self.frame_jump-self.frame_number>=0]
        if self.cap and self.synched and len(forward_jumps)>0:
            self.frame_number = min(forward_jumps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
            self.update_frame()
            self.update_IMU_plot()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.setValue(self.frame_number)
            self.display_frame(frame)
        else:
            self.timer.stop()
            self.is_playing = False
            self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def seek_video(self):
        if self.cap:
            self.frame_number = self.slider.value()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
