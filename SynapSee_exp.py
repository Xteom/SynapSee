import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PySide6.QtGui import QPixmap
from PySide6.QtCore import QTimer, Qt
from utils.brainflow_streamer import brainflow_streamer 
import os
import time
import datetime

class ImageApp(QMainWindow):
    def __init__(self, timestamps_output_file="timestamps.csv", eeg_output_file="output_file.csv", port="synthetic"):
        print("Initializing ImageApp")
        super().__init__()
        self.eeg_output_file = eeg_output_file
        self.timestamps_output_file = timestamps_output_file
        self.setWindowTitle("BCI Image Viewer")
        self.initUI()
        self.image_index = 0
        self.timestamps = []
        self.bci_streamer = brainflow_streamer(port)
        self.setStyleSheet("background-color: gray;")
        print("Initialized ImageApp")

    def initUI(self):
        print("Initializing UI")
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout(self.central_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)  # Center the image
        self.layout.addWidget(self.image_label)

        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_display)
        self.layout.addWidget(self.start_button, alignment=Qt.AlignCenter)
        print("Initialized UI")

    def start_display(self):
        print("Starting display")
        self.start_button.hide()  # Hide the start button
        self.bci_streamer.start_bci()  # Start BCI
        self.start_timestamp = time.time() # Start timestamp (should be the same as the BCI start timestamp)
        print("Started BCI")
        self.load_images()  # Load images
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_next_image)
        self.timer.start(1000)  # Start the timer
        print("Started display")

    def load_images(self):
        print("Loading images")
        self.images = []
        categories = ['cat', 'dog', 'rabbit', 'control']
        num_images_per_category = 300  # Total number of images per category
        images_per_batch = 50  # Number of images to show before a break

        for batch in range(0, num_images_per_category, images_per_batch):
            for category in categories:
                for i in range(batch, min(batch + images_per_batch, num_images_per_category)):
                    image_path_jpg = f"images/train/{category}/{i}.jpg"
                    image_path_jpeg = f"images/train/{category}/{i}.jpeg"

                    if os.path.exists(image_path_jpg):
                        self.images.append(image_path_jpg)
                    elif os.path.exists(image_path_jpeg):
                        self.images.append(image_path_jpeg)

                self.images.append(None)  # Marker for rest period after each batch of 50 images
                    
        print("Loaded images")
        
    def show_next_image(self):
        if self.image_index < len(self.images):
            image_path = self.images[self.image_index]

            if image_path is None:
                # Record timestamp for rest period
                current_timestamp = time.time()
                elapsed_time = current_timestamp - self.start_timestamp
                self.timestamps.append(("no_stimuli", elapsed_time))

                # Begin rest period
                self.timer.stop()
                self.image_label.setStyleSheet("background-color: black;")
                self.image_label.setPixmap(QPixmap())  # Remove the current image
                QTimer.singleShot(5000, self.resume_timer)  # 5 seconds break
            else:
                pixmap = QPixmap(image_path)
                self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio))
                current_timestamp = time.time()
                elapsed_time = current_timestamp - self.start_timestamp  # Calculate elapsed time
                self.timestamps.append((image_path, elapsed_time))

            self.image_index += 1
        else:
            self.timer.stop()
            self.bci_streamer.stop_bci(output_file=self.eeg_output_file)  # Save the BCI data
            self.save_timestamps()  # Save timestamps
            QApplication.instance().quit()

    def resume_timer(self):
        self.image_label.setStyleSheet("")  # Reset the style to default
        self.timer.start(1000)
        

    def save_timestamps(self):
        with open(self.timestamps_output_file, 'w') as f:
            for path, timestamp in self.timestamps:
                f.write(f'{path}, {timestamp}\n')

def run_app(port="synthetic"):
    '''
    This function runs the app to collect data for the SynapSee project.
    args:
        port: the port number to use for the brainflow streamer
    '''
    name = input("Enter subject name: ")
    sex = input("Enter subject sex (m/w/o): ")
    age = input("Enter subject age: ")
    has_cat = input("Does the subject have a cat? (y/n): ")
    has_dog = input("Does the subject have a dog? (y/n): ")
    has_rabbit = input("Does the subject have a rabbit? (y/n): ")

    app = QApplication(sys.argv)

    now = datetime.datetime.now()
    str_now = now.strftime("%Y%m%d%H%M")
    str_name = f"{name}-{sex}-{age}-{has_cat}-{has_dog}-{has_rabbit}-{str_now}"

    ex = ImageApp(timestamps_output_file=f"SynapSee_data/timestamps_{str_name}.csv",
                   eeg_output_file=f"SynapSee_data/output_file_{str_name}.csv", port=port)
    ex.show()
    sys.exit(app.exec())

# this needs to be modified to take in the port number
def main():
    run_app()

if __name__ == '__main__':
    main()
