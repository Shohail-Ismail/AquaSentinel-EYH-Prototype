import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plot

# synthetic training data
training_samples = 1000
hrate = np.random.normal(loc = 80, scale = 10, size = training_samples)
bo2 = np.random.normal(loc = 97.5, scale =2.5, size = training_samples)
training_data = np.stack([hrate, bo2], axis = 1)

# min-max
data_min = np.min(training_data, axis = 0)
data_max = np.max(training_data, axis = 0)
norm_tr_data = (training_data - data_min) / (data_max - data_min)
# print(norm_tr_data[:10])


# synth test data 
test_samples = 100

# avg values
avg_hrate = np.random.normal(loc = 75, scale = 5, size = test_samples)
avg_bo2 = np.random.normal(loc = 98, scale = 1, size = test_samples)
avg_norm = np.stack([avg_hrate, avg_bo2], axis =1)

# panic vals
hrate_panic = np.random.normal(loc = 60, scale = 10, size = 10)
bo2_panic = np.random.normal(loc = 92, scale = 4, size = 10)
panic_norm = np.stack([hrate_panic, bo2_panic], axis =1)

test = np.concatenate([avg_norm, panic_norm], axis = 0)
norm_tst_data = (test - data_min) / (data_max - data_min)
#print(norm_tst_data)

# AUTOENCODER (ANN)
input_dim = norm_tr_data.shape[1]
input_layer = keras.Input(shape = (input_dim,))

# dense layer to encode input data into 1d with ReLU for complex patterns
# then decoding it back to originial shape with linear activ func
encoded = layers.Dense(1, activation = 'relu')(input_layer)
decoded = layers.Dense(input_dim, activation = 'linear')(encoded)

# build and compile w  adam and mse
demo_autoencoder = keras.Model(inputs = input_layer, outputs = decoded)
demo_autoencoder.compile(optimizer= 'adam', loss = 'mse')

demo_autoencoder.summary()

#train model
demo_autoencoder.fit(norm_tr_data, norm_tr_data, epochs = 50, batch_size =32, shuffle = True)

# convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(demo_autoencoder)
tflite_model = converter.convert()