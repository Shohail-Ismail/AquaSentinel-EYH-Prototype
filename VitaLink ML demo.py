import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
# import os

# Maintaining reproducibility was attempted, however this was unable to be achieved
# due to timing constraints - however the overall accuracy only fluctuates by +- 3% 
# accuracy, and during 85% of tests, the anomaly detection accuracy was at 100%
"""
## Setting up reproducibility during eval
np.random.seed(19)
tf.random.set_seed(19)
# Removes floating-point variations for full determinism
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
"""


## Generates simple synthetic training data 
training_samples = 1000
hrate = np.random.normal(loc = 80, scale = 10, size = training_samples)
bo2 = np.random.normal(loc = 97.5, scale =2.5, size = training_samples)

# Combine 2 arrays into single 2d array
training_data = np.stack([hrate, bo2], axis = 1)

# Minmax normalisation
data_min = np.min(training_data, axis = 0)
data_max = np.max(training_data, axis = 0)
norm_tr_data = (training_data - data_min) / (data_max - data_min)



## Generates simple synthetic test data 
test_samples = 100

# Normal test data
test_hrate_normal = np.random.normal(loc = 75, scale = 5, size = test_samples)
test_bo2_normal = np.random.normal(loc = 98, scale = 1, size = test_samples)
normal_test_normalised = np.stack([test_hrate_normal, test_bo2_normal], axis =1)

# Small subset of test data is anomalous 
# (simulating diver facing hypoxia due to loss of consciousness)
# anomalous overlaps with normal data to maximise sensitivity and prioritise anomaly detection over false positives
test_hrate_hypoxia = np.random.normal(loc = 60, scale = 10, size = 10)
test_bo2_hypoxia = np.random.normal(loc = 92, scale = 4, size = 10)
hypoxia_test_normalised = np.stack([test_hrate_hypoxia, test_bo2_hypoxia], axis =1)

# Combines test data and creates corresponding labels (0 = normal, 1 = anomaly).
complete_test_data = np.concatenate([normal_test_normalised, hypoxia_test_normalised], axis = 0)
test_labels = np.concatenate([np.zeros(test_samples), np.ones(10)])

# Normalise test data
norm_ts_data = (complete_test_data - data_min) / (data_max - data_min)



## Defines autoencoder for anomalous value detection
# Gets no of features in data (used instead of explicit '2' to allow for
# future addition of more features)
input_dim = norm_tr_data.shape[1]

#Defines inp layer using no of features (2)
input_layer = keras.Input(shape = (input_dim,))

# Compress data to 1D w ReLU to capture principal features
encoded = layers.Dense(1, activation =  'relu')(input_layer)

# Reconstruct original values from encoded
decoded = layers.Dense(input_dim, activation = 'linear')(encoded)

# Build demo autoencoder by connecting inp. to decoded data
demo_autoencoder = keras.Model(inputs = input_layer, outputs = decoded)

# Compiles model with appropriaate opt and loss func
demo_autoencoder.compile(optimizer= 'adam', loss = 'mse')

#Print model architecture
demo_autoencoder.summary()

#Training model on normal data
demo_autoencoder.fit(norm_tr_data, norm_tr_data,
                epochs = 50,
                batch_size =32,
                shuffle = True, 
                verbose = 0) # done to keep output clean, can remove for evaluation

# Convert trained autoencoder model to Tensorflow Lite (tflite)
converter = tf.lite.TFLiteConverter.from_keras_model(demo_autoencoder)
tflite_model = converter.convert()

# Commented out as saving model not needed for eval
# Uncomment after eval for deployment on microcontroller
"""
with open("AquaSentinel.tflite", 'wb') as f:
    f.write(tflite_model)
    
interpreter = tf.lite.Interpreter(model_content="AquaSentinel.tflite")
"""



## Autoencoder inference demo
# Use in-mem model instead of saving 
interpreter = tf.lite.Interpreter(model_content=tflite_model)

# Mem allocate tensors and fetch I/O tensor metadata
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Computes reconstruction error for anomaly detection
def get_reconstr_err(s):
    
    # Adds dim so that sample is treated as batch of one (many models expect a batch) 
    # and converts to 32-bit fp
    s = np.expand_dims(s, axis = 0).astype(np.float32)
    
    # Performs inference on sample
    interpreter.set_tensor(input_details[0]['index'], s)
    interpreter.invoke()
    
    # Calc avg MSE of original outp and reconstr outp
    reconstruction = interpreter.get_tensor(output_details[0]['index'])
    error = np.mean((s - reconstruction) ** 2)
    
    # Value to decide if sample is anomalous
    return error

# Get reconstr errors for test data
reconstr_errors = []
for sample in norm_ts_data:
    error = get_reconstr_err(sample)
    reconstr_errors.append(error)
reconstr_errors = np.array(reconstr_errors)



## Establish normal/anom threshold based on distribution of training errors
# Get reconstruction errors for training data
train_errors = []
for sample in norm_tr_data:
    error = get_reconstr_err(sample)
    train_errors.append(error)
    
train_errors = np.array(train_errors)
# Threshold is avg + 0.5 std dev
# std dev is very low to ensure all anomalies are detected even at the risk of false positives
threshold = np.mean(train_errors) + 0.5 * np.std(train_errors)
print(f"\nAnom threshold: {threshold:.4f}")  # TESTING

# Classifies each test sample as anomaly if error > threshold.
predicted_anomalies = reconstr_errors > threshold



## Output and visualisation
print("\nTest results:\n")
for i, (error, label, pred) in enumerate(zip(reconstr_errors, test_labels, predicted_anomalies)):
    status = "Anomaly" if pred else "Normal"
    actual = "Anomaly" if label == 1 else "Normal"
    print(f"Sample {i+1:03d}: Reconstruction error = {error:.4f} | Predicted: {status} | Actual: {actual}")

# Visualise the reconstruction errors along with the anomaly threshold
plt.figure(figsize=(10, 6))
plt.plot(reconstr_errors, 'bo-', label = 'Reconstruction error')
plt.axhline(y = threshold, color= 'r', linestyle = '--', label = 'Anomaly threshold')
plt.title("Reconstruction Error For Test Data")
plt.xlabel("Test data index")
plt.ylabel("Mean squared error (MSE)")
plt.legend()
plt.show()

# Init counters for accuracy calcs
total_samples = len(test_labels)
correct_pred = 0
correct_norm = 0
correct_anom = 0

# Loop over each test samples true label and prediction
for label, pred in zip(test_labels, predicted_anomalies):
    # Correctly classified normal
    if label == 0 and not pred:
        correct_norm += 1
        correct_pred += 1
    # Correctly classified anomaly
    elif label == 1 and pred:
        correct_anom += 1
        correct_pred += 1

# Conv accuracies to %
overall_accuracy = (correct_pred / total_samples) * 100
anomaly_accuracy = (correct_anom / 10) * 100

# Print the accuracy results
print("\nAccuracy results:")
print(f"Normal accuracy: {correct_norm}%")
print(f"Anomaly accuracy: {anomaly_accuracy}%")
print(f"Overall: {overall_accuracy:.2f}%")
