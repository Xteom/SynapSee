from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time

BoardShim.enable_dev_board_logger()
# use synthetic board for demo
params = BrainFlowInputParams()
board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
board.prepare_session()
board.start_stream()
time.sleep(3)
data = board.get_board_data()
board.stop_stream()
board.release_session()
eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
print(data)
print(data[:16].shape) # son 32 canales pero solo tomaríamos 16 por nuestro casco