from pynput.mouse import Listener
import time
import pandas as pd
import numpy as np
import threading

# Initiate empty data
data = {'time': [], 'x': [], 'y': []}
lock = threading.Lock()

# Define action on mouse move
def on_move(x, y):
    with lock:
        data['time'].append(time.time())
        data['x'].append(x)
        data['y'].append(y)

# Define a function to save data every X seconds
def save_data_every(interval=10):
    while True:
        time.sleep(interval)

        with lock:
            # Make a DataFrame of the data
            df = pd.DataFrame(data)

            # Calculate change in position
            df['dx'] = df['x'].diff()
            df['dy'] = df['y'].diff()

            # Calculate angle of movement
            df['angle'] = np.arctan2(df['dy'], df['dx'])

            # Save preprocessed data
            df.to_csv('mouse_data_preprocessed_' + fileName + '.csv', index=False)

fileName = input("Type of file: ")

# Start a thread that saves the data every X seconds
thread = threading.Thread(target=save_data_every)
thread.start()

# Start the listener
with Listener(on_move=on_move) as listener:
    listener.join()
