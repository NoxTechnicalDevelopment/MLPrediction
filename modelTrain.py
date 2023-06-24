import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow as tf

if len(tf.config.list_physical_devices('GPU')) == 1:
    print("GPU Found")
elif len(tf.config.list_physical_devices('GPU')) > 1:
    print("GPUs Found")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
else:
    print("GPU Not found")

# Load data
df = pd.read_csv('mouse_data_preprocessed_long_train.csv')

# Remove rows with missing values
df = df.dropna()

# Define window size
window_size = 30

# Prepare windowed data
input_data = []
output_data = []

for i in range(window_size, len(df)):
    input_data.append(df[['x', 'y', 'dx', 'dy', 'angle']].iloc[i-window_size:i].values)
    output_data.append(df['angle'].iloc[i])

# Convert to numpy arrays
input_data = np.array(input_data)
output_data = np.array(output_data)

# Reshape output_data to fit model
output_data = output_data.reshape(-1, 1)

# Load the previously trained model
model = load_model('mouse_direction_predictor.h5')

# Continue training the model
model.fit(input_data, output_data, epochs=200, verbose=1)

# Save the further trained model
model.save('mouse_direction_predictor.h5')