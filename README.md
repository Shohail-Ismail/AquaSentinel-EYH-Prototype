# AquaSentintel-EYH-Demo

   * [Background information](#background-information)
   * [Core features](#core-features)
   * [Limitations](#limitations)
   * [Running the program](#running-the-program)


## Background information

AquaSentintel is a machine learning-driven anomaly-detection system designed for deep-sea diving safety, developed as part of a 6-person multidisciplinary team for the 'Engineering: You're Hired!' university-wide competition. AquaSentinel is a component of our project 'VitaLink': a wearable armband that continuously monitors a diver’s vital signs and environmental conditions, leveraging AI-based anomaly detection to identify potential medical emergencies, and alerting both the diver and the surface team in case of one.

An autoencoder Artificial Neural Network (ANN) architecture has been used to learn normal physiological patterns and detect deviations that could indicate distress. The model has been optimised for deployment on embedded devices with low power consumption and real-time inference capabilities. 

## Core Features

- **ML anomaly detection**  
  - Uses an **autoencoder** trained on synthetic normal physiological data to detect anomalies.
  - Identifies hypoxia-like conditions in divers based on heart rate and blood oxygen levels (anomalous data is also synthetic).
  - Chose optimiser, loss function, activation functions for encoding and decoding using comparative experiments between configurations (as shown in the training/validation loss graph below) prioritising lower validation loss and stability across epochs.
  - Model is slightly aggressive and, when presented with values at the threshold, errs on the side of anomalous in order to prioritise diver safety over false positives.

### Training/validation loss experiments for 2 different autoencoder configurations
![Training/Validation Loss Experiments for 2 Autoencoder Configs](Training-and-validation-loss-experiments-for-2-autoencoder-configs)


- **Optimised for edge computing**  
  - Converts the trained model to **TensorFlow Lite (TFLite)** for deployment on microcontrollers.
  - Lightweight architecture with minimal computational overhead.
  - Uses in-memory execution rather than persistent storage (done for evaluation purposes; in a real-world deployment, persistent storage is needed to save the *.tflite model).

- **Evaluation and visualisation**
  - The autoencoder ANN was chosen after considering the input data and the accuracy of the classification needed, weighed against other models such as decision trees, support vector machines, and traditional statistical anomaly detection methods.
  - The autoencoder reconstructs an input sample, after which the reconstruction error (MSE between the original and reconstructed data) is calculated, with a pre-computed threshold used to classify the sample.

- **Using in-memory TFLite model**
  - Uses in-memory TFLite model instead of saving the model, making the program more efficient by skipping the disk I/O operation.
  - Persistent storage is not required (at this stage) since the model is loaded dynamically during execution.

## Limitations

- **Reproducibility constraints**  
  - Due to timing constraints, full reproducibility was not able to be achieved (the reason for this was not fully understood as despite setting random seeds, forcing TensorFlow to behave deterministically, and disabling `oneDNN` (which was causing nondeterminism due to floating-point round-off errors), the program was still giving different accuracies with each run).
  - However, overall accuracy fluctuations were observed to only be in the range of (96±3)%, with 85% of tests achieving **100% anomaly detection accuracy**.

- **Limited feature set and simplified model**  
  - Currently detects only heart rate and blood oxygen anomalies, however this is expected for a prototype demonstration.
  - Later, the model would need to be expanded to encompass more vital signs (e.g., breathing patterns, temperature).
  - Further optimisations are required for real world deep-sea conditions.

## Running The Program

- **Required Python version**: Python 3.10.9
- **Libraries**:
  - `numpy`
  - `tensorflow`
  - `tensorflow.keras`
  - `maptplotlib`
 
- **Running the program**:

```bash
# Imports
pip install -r requirements.txt

# To run the analysis
python '.\VitaLink ML demo.py'
```

- **Expected outputs**:
  - Autoencoder summary (layers, output shapes and number of trainable/non-trainable parameters
  - Saved `*.tflite` model
  - Anomaly threshold value and reconstruction error values, and predicted/actual labels
  - Graph showing results of inference with anomaly threshold and all samples values' reconstruction errors plotted 3 successive program runs given below)
  - Accuracy results (overall, labelling normal values, labelling anomalous values)

### Example run 1: 
#### 100% overall accuracy (100% anomaly detection, 100% normal classification)

![Sample plot 1](sample-plot-example-1)

### Example run 2: 
#### 99.5% overall accuracy (100% anomaly detection, 99% normal classification)
![Sample plot 2](sample-plot-example-2)

### Example run 3: 
#### 97.27%% overall accuracy (90.0%% anomaly detection, 98% normal classification)
![Sample plot 3](sample-plot-example-3)

