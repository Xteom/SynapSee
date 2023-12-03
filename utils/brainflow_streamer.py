import pandas as pd
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
import numpy as np
import time

class brainflow_streamer:
    def __init__(self, port='COM4'):
        self.params = BrainFlowInputParams()
        
        if port.lower() == 'synthetic':
            self.board_id = brainflow.BoardIds.SYNTHETIC_BOARD.value
        else:
            self.params.serial_port = port
            self.board_id = brainflow.BoardIds.CYTON_DAISY_BOARD.value

        self.board = None

    def start_bci(self):
        print('start bci')
        
        BoardShim.enable_dev_board_logger()
        self.board = BoardShim(self.board_id, self.params)
        self.board.prepare_session()
        self.board.start_stream()
        print("BCI stream started.")

    def stop_bci(self, output_file):
        if self.board:
            data = self.board.get_board_data()
            self.board.stop_stream()
            self.board.release_session()
            print("BCI stream stopped.")
            
            # Adjusted to only capture the first 16 channels for the synthetic board
            eeg_channels = BoardShim.get_eeg_channels(self.board_id)[:16]
            eeg_names = BoardShim.get_eeg_names(self.board_id)[:16]

            df = pd.DataFrame(np.transpose(data))
            df_eeg = df[eeg_channels]
            df_eeg.columns = eeg_names
            df_eeg.to_csv(output_file, sep=',', index=False)
            print(df_eeg)
