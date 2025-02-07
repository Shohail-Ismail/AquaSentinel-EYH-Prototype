# AquaSentintel-EYH-Demo

   * [Background information](#background-information)
   * [Core features](#core-features)
   * [Limitations](#limitations)
   * [Running the program](#running-the-program)


## Background information

AquaSentintel is a machine learning driven anomaly detection system designed for deep-sea diving safety. It is a component of VitaLink: a wearable armband that continuously monitors a diver’s vital signs and environmental conditions, leveraging AI-based anomaly detection to identify potential medical emergencies, and alerting both the diver and the surface team in case of one.

An autoencoder Artificial Neural Network (ANN) architecture to learn normal physiological patterns and detect deviations that could indicate distress. The model has been optimised for deployment on embedded devices with low power consumption and real-time inference capabilities. 

## Core Features

- **ML anomaly detection**  
  - Uses an **autoencoder** trained on synthetic normal physiological data to detect anomalies.
  - Identifies hypoxia-like conditions in divers based on heart rate and blood oxygen levels (anomalous data is also synthetic).
  - Model is slightly aggressive and, when presented with features at the threshold errs on the side of anomalous in order to prioritise diver safety over false positives.

- **Optimised for edge computing**  
  - Converts the trained model to **TensorFlow Lite (TFLite)** for deployment on microcontrollers.
  - Lightweight architecture with minimal computational overhead.
  - Uses in-memory execution rather than persistent storage (for evaluation purposes - in a real-world deployment, persistent storage is needed to save the *.tflite model).

- **Evaluation and visualisation**
  - The autoencoder ANN was chosen after considering the requirements of the project and the necessary processing power and efficacy needed.
  - Provides **reconstruction error analysis** for anomaly detection.
  - Outputs classification results with performance metrics.
  - Displays an **MSE (Mean Squared Error) threshold** for anomaly classification.

- **Automatic cleanup using `tempfile`**  
  - The use of **`tempfile`** ensures automatic cleanup of temporary model files, preventing conflicts in the filesystem.
  - Persistent storage is not required at this stage since the model is loaded dynamically during execution.

## Limitations

- **Reproducibility constraints**  
  - Due to timing constraints, full reproducibility was not able to be achieved.
  - However, accuracy fluctuations of only ±3% were observed in different test runs, with 85% of tests achieving **100% anomaly detection accuracy**.

- **Limited feature set and simplified model**  
  - Currently detects only heart rate and blood oxygen anomalies, however this is expected for a prototype demonstration.
  - Later, the model would need to be expanded to encompass more vital signs (e.g., breathing patterns, temperature).
  - Further optimisations are required for real world deep-sea conditions.

## Running The Program

