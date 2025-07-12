import sys
import subprocess
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, 
                             QFileDialog, QTextEdit, QLabel, QHBoxLayout, QLineEdit, 
                             QWidget, QComboBox, QDoubleSpinBox, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap

class ProcessThread(QThread):
    output_signal = pyqtSignal(str)

    def __init__(self, point_cloud, model, output_dir, batch_size):
        super().__init__()
        self.point_cloud = point_cloud
        self.model = model
        self.output_dir = output_dir
        self.batch_size = batch_size

    def run(self):
        try:
            command = [
                    'python', "scripts/predict_rgb.py",
                    '--batch_size', str(int(self.batch_size)),
                    '--model', self.model,
                    '--point_cloud', self.point_cloud,
                    '--output_dir', self.output_dir
                ]
        
            # Print the command to verify
            print("Running Command:", " ".join(command))
            self.output_signal.emit(f"Running Command: {' '.join(command)}")

            # Run the process and capture output
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

            # Capture stdout
            for line in process.stdout:
                self.output_signal.emit(line.strip())

            # Capture stderr
            for line in process.stderr:
                self.output_signal.emit(f"Error: {line.strip()}")

            process.stdout.close()
            process.stderr.close()
            process.wait()
        
        except Exception as e:
            self.output_signal.emit(f"Error: {str(e)}")

class PointCloudClassificationGUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        # Main layout
        layout = QVBoxLayout()

        # Load Point Cloud to Classify (LAS)
        self.pointcloud_label = QLabel('Load Point Cloud to Classify:')
        self.pointcloud_path = QLineEdit(self)
        self.pointcloud_path.setReadOnly(True)
        self.pointcloud_btn = QPushButton('...', self)
        self.pointcloud_btn.clicked.connect(self.select_pointcloud)

        # Layout for Load Point Cloud to Classify
        pointcloud_layout = QHBoxLayout()
        pointcloud_layout.addWidget(self.pointcloud_path)
        pointcloud_layout.addWidget(self.pointcloud_btn)

        # Load Model
        self.model_label = QLabel('Load Model:')
        self.model_path = QLineEdit(self)
        self.model_path.setReadOnly(True)
        self.model_btn = QPushButton('...', self)
        self.model_btn.clicked.connect(self.select_model)

        # Layout for Model
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(self.model_btn)

        # Output Directory input
        self.output_label = QLabel('Select Output Directory:')
        self.output_path = QLineEdit(self)
        self.output_path.setReadOnly(True)
        self.output_btn = QPushButton('...', self)
        self.output_btn.clicked.connect(self.select_output_file)

        # Directory Layout
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(self.output_btn)

        # Advanced Options
        self.advanced_options_label = QPushButton('Advanced Options ▼')
        self.advanced_options_label.setCheckable(True)
        self.advanced_options_label.setChecked(False)
        self.advanced_options_label.clicked.connect(self.toggle_advanced_options)

        self.batch_size_label = QLabel('Batch Size:')
        self.batch_size = QDoubleSpinBox(self)
        self.batch_size.setValue(int(16))

        # Advanced Options Layout
        advanced_layout = QVBoxLayout()
        advanced_layout.addWidget(self.advanced_options_label)
        advanced_layout.addWidget(self.batch_size_label)
        advanced_layout.addWidget(self.batch_size)

        # Buttons (Start and Replay)
        buttons_layout = QHBoxLayout()

        # Start Button
        self.start_btn = QPushButton('Start', self)
        self.start_btn.setMinimumWidth(150)
        self.start_btn.clicked.connect(self.start_process)

        # Replay Button (Clear input fields)
        self.replay_btn = QPushButton(self)
        self.replay_btn.setIcon(QIcon("ui/replay.png"))  # Add your replay icon image here
        self.replay_btn.setFixedWidth(100)
        self.replay_btn.clicked.connect(self.clear_inputs)

        # Add Start and Replay buttons to layout
        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(self.replay_btn)

        # Loading bar (progress bar)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(100)  # Indeterminate progress
        self.progress_bar.setMinimum(0)

        # Console log
        self.log_console = QTextEdit(self)
        self.log_console.setReadOnly(True)

        # Add widgets to layout
        layout.addWidget(self.pointcloud_label)
        layout.addLayout(pointcloud_layout)
        layout.addWidget(self.model_label)
        layout.addLayout(model_layout)
        layout.addWidget(self.output_label)
        layout.addLayout(output_layout)
        layout.addLayout(advanced_layout)
        layout.addLayout(buttons_layout)
        layout.addWidget(self.progress_bar)  # Add progress bar here
        layout.addWidget(QLabel('Console Log:'))
        layout.addWidget(self.log_console)

        # Watermark Layout
        watermark_layout = QHBoxLayout()
        watermark_logo = QLabel(self)
        watermark_logo.setPixmap(QPixmap("ui/ugm.png").scaled(40, 40))
        watermark_layout.addWidget(watermark_logo)
        watermark_text = QLabel("Department of Geodetic Engineering\nFaculty of Engineering Universitas Gadjah Mada")
        watermark_layout.addWidget(watermark_text)
        watermark_layout.addStretch(1)
        watermark_layout.setContentsMargins(0, 20, 0, 0)
        layout.addLayout(watermark_layout)

        # Set layout for main window
        self.setLayout(layout)
        self.setWindowTitle('Point Cloud Classifier')
        self.setGeometry(100, 100, 800, 600)

        # Hide the advanced options by default
        self.advanced_options_label.setVisible(False)
        self.batch_size_label.setVisible(False)
        self.batch_size.setVisible(False)

        # Set the window icon
        self.setWindowIcon(QIcon("ui/logo.png"))  # Replace with the path to your logo image

        # Hide the advanced options by default
        self.toggle_advanced_options()

    def toggle_advanced_options(self):
        is_visible = self.advanced_options_label.isChecked()
        self.batch_size_label.setVisible(is_visible)
        self.batch_size.setVisible(is_visible)

        # Update the button label to reflect current state
        self.advanced_options_label.setText('Advanced Options ▼' if not is_visible else 'Advanced Options ▲')

    # Functions for selecting files/directories
    def select_model(self):
        model, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.t7)")
        if model:
            self.model_path.setText(model)
            self.log_console.append(f"Selected Model: {model}")

    def select_pointcloud(self):
        pointcloud, _ = QFileDialog.getOpenFileName(self, "Select Point Cloud (*.las)", "", "LAS Files (*.las)")
        if pointcloud:
            self.pointcloud_path.setText(pointcloud)
            self.log_console.append(f"Selected Point Cloud: {pointcloud}")

    def select_output_file(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_path.setText(directory)
            self.log_console.append(f"Selected Output Directory: {directory}")

    def start_process(self):
        # Disable the Start button to prevent multiple clicks
        self.start_btn.setEnabled(False)

        # Gather inputs
        model = self.model_path.text()
        point_cloud = self.pointcloud_path.text()
        output_file_path = self.output_path.text()
        batch_size = self.batch_size.value()

        # Validate input
        if not model or not point_cloud or not output_file_path:
            self.log_console.append("Error: Please fill all required fields")
            self.start_btn.setEnabled(True)  # Re-enable the Start button
            return

        batch_size = self.batch_size.value()
        
        # Start the process in a separate thread
        self.process_thread = ProcessThread(point_cloud, model, output_file_path, batch_size)
        self.process_thread.output_signal.connect(self.update_console_log)
        self.process_thread.finished.connect(self.process_finished)  # Call a function when the process finishes
        self.process_thread.start()

        # Indeterminate progress during execution
        self.progress_bar.setMaximum(0)  # Indeterminate mode

    def process_finished(self):
        # Reset the progress bar to initial state when the process is done
        self.progress_bar.setMaximum(100)  # Switch back to determinate mode
        self.progress_bar.setValue(0)      # Reset progress bar
        self.start_btn.setEnabled(True)    # Re-enable the Start button
    
    def update_console_log(self, message):
        self.log_console.append(message)

    def clear_inputs(self):
        # Clear all the input fields
        self.model_path.clear()
        self.pointcloud_path.clear()
        self.output_path.clear()
        self.log_console.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PointCloudClassificationGUI()
    ex.show()
    sys.exit(app.exec_())