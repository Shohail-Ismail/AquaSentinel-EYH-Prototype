import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plot

# Setting up reproducibility during eval
np.random.seed(19)
tf.random.set_seed(19)

# synthetic training data
training_samples = 1000
hrate = np.random.normal(loc = 80, scale = 10, size = training_samples)
bo2 = np.random.normal(loc = 97.5, scale =2.5, size = training_samples)
training_data = np.stack([hrate, bo2], axis = 1)

# min-max
data_min = np.min(training_data, axis = 0)
data_max = np.max(training_data, axis = 0)
norm_tr_data = (training_data - data_min) / (data_max - data_min)

# synth test data 
test_samples = 100

# avg values
avg_hrate = np.random.normal(loc = 75, scale = 5, size = test_samples)
avg_bo2 = np.random.normal(loc = 98, scale = 1, size = test_samples)
avg_norm = np.stack([avg_hrate, avg_bo2], axis =1)

# anomalous vals
hrate_anom = np.random.normal(loc = 60, scale = 10, size = 10)
bo2_anom = np.random.normal(loc = 92, scale = 4, size = 10)
anom_norm = np.stack([hrate_anom, bo2_anom], axis =1)

test = np.concatenate([avg_norm, anom_norm], axis = 0)
labels = np.concatenate([np.zeros(test_samples), np.ones(10)]) #0 = avg, 1 = panic 
norm_tst_data = (test - data_min) / (data_max - data_min)

# AUTOENCODER (ANN)
input_dim = norm_tr_data.shape[1]
input_layer = keras.Input(shape = (input_dim,))

# dense layer made to encode inp data to 1d with ReLU
# then decoding back to originial shape with linear activ func
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

# Save as AquaSentinel
with open("AquaSentinel.tflite", 'wb') as f:
    f.write(tflite_model)

# Get anomalous data from errors of autoencoder's reconstr of samples
def get_reconstr_err(s):
    
    interpreter = tf.lite.Interpreter("AquaSentinel.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # treat sample s as batch
    s = np.expand_dims(s, axis = 0).astype(np.float32)
    
    # inference on sample
    interpreter.set_tensor(input_details[0]['index'], s)
    interpreter.invoke()
    
    # reconstr err w MSE
    reconstruction = interpreter.get_tensor(output_details[0]['index'])
    error = np.mean((s - reconstruction) ** 2)
    return error

# Get reconstr errors for test data
reconstr_errors = []
for sample in norm_tst_data:
    error = get_reconstr_err(sample)
    reconstr_errors.append(error)
reconstr_errors = np.array(reconstr_errors)

# Get reconstr errors for training data
train_errors = []
for i in range(len(norm_tr_data)):
    sample = norm_tr_data[i]
    error = get_reconstr_err(sample)
    train_errors.append(error)
train_errors = np.array(train_errors)

#threshold separaing anom and avg valie
threshold = 0
for error in train_errors:
    threshold += error
    #std dev low so anoms detected even if false +ve
threshold = threshold / len(train_errors) + 0.5 * np.std(train_errors)
print(f"\nAnom threshold: {threshold:.4f}")  # TESTING

# Classifies each test sample as anomaly if error > threshold.
predicted_anomalies = []
for error in reconstr_errors:
    if error > threshold:
        predicted_anomalies.append(True)
    else:
        predicted_anomalies.append(False)
        
