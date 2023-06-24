import time
import pandas as pd
import numpy as np
from keras.models import load_model
from pynput.mouse import Controller
from tkinter import TclError, Tk, Canvas

# Load trained model
model = load_model('mouse_direction_predictor.h5')

# Load preprocessed data
df = pd.read_csv('mouse_data_preprocessed_test.csv')

# Define window size and prediction length
window_size = 30
prediction_length = 40

# Prepare windowed data
input_data = []
output_data = []

for i in range(window_size, len(df)):
    input_data.append(df[['x', 'y', 'dx', 'dy', 'angle']].iloc[i-window_size:i].values)
    output_data.append(df['angle'].iloc[i])

# Convert to numpy arrays and get predictions
input_data = np.array(input_data)
output_data = np.array(output_data)
predictions = model.predict(input_data)

# Create Tkinter window
root = Tk()
canvas = Canvas(root, width=1920, height=1080)
canvas.pack()

# Initialize mouse controller
mouse = Controller()

# Function to update cursor positions
def update_positions(i):
    # Get recorded and predicted positions
    recorded_position = (df['x'][i+window_size], df['y'][i+window_size])
    predicted_angle = predictions[i]
    predicted_position = (int(np.round(recorded_position[0] + prediction_length * np.cos(predicted_angle))),
                          int(np.round(recorded_position[1] + prediction_length * np.sin(predicted_angle))))
    
    # Get actual positions
    actual_angle = df['angle'][i+window_size+1] if i+window_size+1 < len(df) else 0
    actual_position = (int(np.round(recorded_position[0] + prediction_length * np.cos(actual_angle))),
                       int(np.round(recorded_position[1] + prediction_length * np.sin(actual_angle))))
    
    # Update recorded cursor
    canvas.delete('recorded')
    canvas.create_oval(recorded_position[0]-5, recorded_position[1]-5,
                       recorded_position[0]+5, recorded_position[1]+5,
                       fill='white', tags='recorded')
    
    # Check if predicted position is within bounds
    if 0 <= predicted_position[0] <= canvas.winfo_width() and 0 <= predicted_position[1] <= canvas.winfo_height():
        # Update predicted cursor
        canvas.delete('predicted')
        canvas.create_oval(predicted_position[0]-5, predicted_position[1]-5,
                           predicted_position[0]+5, predicted_position[1]+5,
                           fill='blue', tags='predicted')
    else:
        print(f"Skipping data point {i} due to out-of-bounds prediction")

    # Check if actual position is within bounds
    if 0 <= actual_position[0] <= canvas.winfo_width() and 0 <= actual_position[1] <= canvas.winfo_height():
        # Update actual cursor
        canvas.delete('actual')
        canvas.create_oval(actual_position[0]-5, actual_position[1]-5,
                           actual_position[0]+5, actual_position[1]+5,
                           fill='lightblue', tags='actual')
    else:
        print(f"Skipping data point {i} due to out-of-bounds actual")

    # Continue updating positions if not at end of data
    if i < len(predictions) - 1:
        root.after(100, update_positions, i+1)
    else:
        root.quit()


# Start updating positions after short delay
root.after(1000, update_positions, 0)

# Start Tkinter event loop
root.mainloop()
