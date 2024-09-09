import pandas as pd
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
import numpy as np
import time

class BrainflowStreamer:
    def __init__(self, port='COM4'):
        '''
        Initializes the BrainflowStreamer object.
        
        :param port: The COM port of the Cyton Daisy board. Default is COM4.
            if port is 'synthetic', a synthetic board will be used for testing
            the frequency of the synthetic board is not the same as the Cyton Daisy board. 
            In windows, you can find the COM port by going to Device Manager > Ports (COM & LPT)
            (usually port COM3 or COM4)
            In Mac, you can find the port by going to System Preferences > Network > USB Serial #TODO Someone with a Mac please confirm
            (usually '/dev/ttyUSB*') # this is confirmed
            In Linux, you can find the port by going to System Settings > Hardware > Ports (usually '/dev/ttyUSB*') # TODO Someone with Linux please confirm
        '''
        self.params = BrainFlowInputParams()
        
        if port.lower() == 'synthetic':
            #TODO: make the frequency of the synthetic board the same as the Cyton Daisy board
            self.board_id = brainflow.BoardIds.SYNTHETIC_BOARD.value
        else:
            self.params.serial_port = port
            self.board_id = brainflow.BoardIds.CYTON_DAISY_BOARD.value

        self.board = None

    def start_bci(self):
        '''
        Starts the BCI stream.
        '''
        print('start bci')
        
        BoardShim.enable_dev_board_logger()
        self.board = BoardShim(self.board_id, self.params)
        self.board.prepare_session()
        self.board.start_stream()
        print("BCI stream started.")

    def stop_bci(self, output_file = None, save_csv = True):
        '''
        Stops the BCI stream and optionally saves the EEG data to a CSV file.

        :param output_file: The file path to save the EEG data. Required if save_csv is True.
        :param save_csv: If True, saves the EEG data to a CSV file. Default is True.
                         If False, the data is not saved to a file.
        :raises ValueError: If save_csv is True and output_file is not provided.
        '''

        if save_csv and output_file is None:
            raise ValueError("output_file is required when save_csv is True")
        
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
            if save_csv:
                df_eeg.to_csv(output_file, sep=',', index=False)
            print(df_eeg)
