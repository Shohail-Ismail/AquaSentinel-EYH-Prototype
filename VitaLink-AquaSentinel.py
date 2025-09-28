import numpy as np
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite
import argparse


def create_data():
    # Synthetic training data 
    training_samples = 10000
    hrate = np.random.normal(80, 10, training_samples)
    bo2 = np.random.normal(97.5, 2.5, training_samples)
    train_data = np.stack([hrate, bo2], axis=1)
    data_min = np.min(train_data, axis=0)
    data_max = np.max(train_data, axis=0)
    train_data = (train_data - data_min) / (data_max - data_min)

    # Synthetic test data
    test_samples = 2000
    test_hrate_normal = np.random.normal(75, 5, test_samples)
    test_bo2_normal = np.random.normal(98, 1, test_samples)
    normal_test = np.stack([test_hrate_normal, test_bo2_normal], axis=1)

    # Subset of test data is anomalous, simulating hypoxic diver
    test_hrate_hypoxia = np.random.normal(60, 10, 10)
    test_bo2_hypoxia = np.random.normal(92, 4, 10)
    hypoxia_test = np.stack([test_hrate_hypoxia, test_bo2_hypoxia], axis=1)

    # Combine test data and labels as normal or anomaly
    complete_test = np.concatenate([normal_test, hypoxia_test], axis=0)
    test_labels = np.concatenate([np.zeros(test_samples), np.ones(10)])
    test_data = (complete_test - data_min) / (data_max - data_min)

    return train_data, test_data, test_labels


# Training mode (optional; run with --train)
def train(train_data):

    # Imports tensorflow-cpu, flagging a common error caused by not READMEing the eponymous
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ModuleNotFoundError as e:
        if "tensorflow" in str(e):
            raise ModuleNotFoundError(
                "\n\nIt seems TensorFlow was not installed before --train was run. Please download the training requirements as specified in the README: \n"
                " > pip install -r requirements-train.txt\n"
            ) from e
        else:
            raise

    # Shallow autoencoder
    input_dim = train_data.shape[1] # Explicit '2' not used to reflect future vitals addition
    input_layer = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(1, activation='relu')(input_layer)
    decoded = layers.Dense(input_dim, activation='linear')(encoded)
    demo_autoencoder = keras.Model(input_layer, decoded)
    demo_autoencoder.compile(optimizer='adam', loss='mse')
    # demo_autoencoder.summary()
    demo_autoencoder.fit(train_data, train_data, epochs=50, batch_size=32, shuffle=True,
    verbose=0)

    # Convert and save to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(demo_autoencoder)
    tflite_model = converter.convert()
    with open("AquaSentinel.tflite", 'wb') as f:
        f.write(tflite_model)

    print("Model trained and saved as 'AquaSentinel.tflite'")


# Inference mode (always runs)
def demo_pipeline(train_data, test_data):
    interpreter = tflite.Interpreter(model_path="AquaSentinel.tflite")
    interpreter.allocate_tensors()
    inp_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    # Samples a single data point and returns its reconstruction error
    def sample_err(s):
        # Add dim so sample is treated as batch (of 1)
        sample = np.expand_dims(s, axis=0).astype(np.float32)
        interpreter.set_tensor(inp_details[0]["index"], sample)
        interpreter.invoke()

        # Calc avg MSE of original and reconstructed output
        rec = interpreter.get_tensor(out_details[0]["index"])
        return np.mean((sample - rec) ** 2)

    # Establish normal/anomaly threshold and perform inference on test data
    # Std dev is very low to ensure precision - will need tuning for real world data
    train_errors = np.array([sample_err(s) for s in train_data])
    threshold = np.mean(train_errors) + 0.5 * np.std(train_errors)
    reconstruction_errors = np.array([sample_err(s) for s in test_data])
    predicted_labels = reconstruction_errors > threshold

    return reconstruction_errors, predicted_labels, threshold


def show_results(reconstruction_errors, test_labels, predicted_labels, threshold):
    print("\nTest results:\n")
    for i, (error, label, pred) in enumerate(zip(reconstruction_errors, test_labels, predicted_labels)):
        status = "Anomaly" if pred else "Normal"
        actual = "Anomaly" if label == 1 else "Normal"
        print(f"Sample {i+1:03d}: Reconstruction error = {error:.4f}; Predicted: {status}; Actual: {actual}")

    # Plot errors by anomaly threshold
    plt.figure(figsize=(10, 6))
    plt.plot(reconstruction_errors, 'bo-', label = 'Reconstruction error')
    plt.axhline(y = threshold, color= 'r', linestyle = '--', label = 'Anomaly threshold')
    plt.title("Reconstruction Error For Test Data")
    plt.xlabel("Test data index")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

    # Metrics
    num_norm = np.sum(test_labels == 0)
    num_anom = np.sum(test_labels == 1)

    correct_norm = np.sum((test_labels == 0) & (predicted_labels == False))
    correct_anom = np.sum((test_labels == 1) & (predicted_labels == True))

    normal_accuracy = correct_norm / num_norm * 100
    anomaly_accuracy = correct_anom / num_anom * 100
    overall_accuracy = (correct_norm + correct_anom) / len(test_labels) * 100

    print("\nAccuracy results:")
    print(f"Normal accuracy: {normal_accuracy:.2f}%")
    print(f"Anomaly accuracy: {anomaly_accuracy:.2f}%")
    print(f"Overall: {overall_accuracy:.2f}%")



def main():
    # CLI for training mode
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true", help="Train the model")
    args = p.parse_args()

    train_data, test_data, test_labels = create_data()
    if args.train:
        print("Training the model...this may take a while")
        train(train_data)

    reconstruction_errors, predicted_labels, threshold = demo_pipeline(train_data, test_data)
    show_results(reconstruction_errors, test_labels, predicted_labels, threshold)


if __name__ == "__main__":
    main()




# Reproducibility note:
# Reproducibility was attempted through random seeds and disabling oneDNN optimisations
# but minor non-determinism remains in TensorFlow training ops, leading to small 
# run-to-run accuracy fluctuations. This level of variance is expected for shallow 
# models and does not affect the validity of the demo.
"""
import os
# Setting up reproducibility during eval
np.random.seed(19)
tf.random.set_seed(19)
# Removes floating-point variations for full determinism
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
"""