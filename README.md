# AquaSentinel Demo

   * [Background information](#background-information)
   * [Run-demo](#run-demo)
   * [Core features](#core-features)
   * [Limitations](#limitations)
   * [Accuracy plots](#accuracy-plots)


## Background information

AquaSentinel is a machine learning-driven prototype to detect anomalous vital signs in deep-sea divers. Developed as part of a 6-person multidisciplinary team for the [Engineering: You're Hired!](https://sheffield.ac.uk/engineering/study/youre-hired) university-wide competition. This project won Distinction, placing in the top 5% of students out of 1000+ from 200 teams.

AquaSentinel is a component of our project 'VitaLink': a wearable armband that continuously monitors a diver’s vital signs and environmental conditions, using AI-based anomaly detection to identify potential medical emergencies, and alerting both the diver and the surface team in case of one. A shallow autoencoder Artificial Neural Network (ANN) architecture has been used to learn normal physiological patterns and detect deviations that could indicate distress. The model has been optimised for deployment on embedded devices with low power consumption and real-time inference capabilities. 

---
## Run demo

- **Requires**
    - Python 3.10.9
    - numpy, matplotlib, tflite-runtime, tensorflow-cpu (if retraining)

```bash
pip install -r requirements.txt
```
 
- **Running the program (inference)**:

```python
# To run the demo (inference)
python VitaLink-AquaSentinel.py
```

- **Running the program (retraining)**:
```python
# To retrain the model
pip install tensorflow-cpu
python VitaLink-AquaSentinel.py --train
```

- **Expected outputs**:
  - If training: saved `AquaSentinel.tflite` model rewriting repo one
  - Anomaly threshold value and reconstruction error values, and predicted/actual labels
  - Graph showing results of inference with anomaly threshold and all values' reconstruction errors plotted (3 successive program runs given at bottom of page)
  - Accuracy results (overall percentage, normal accuracy, anomaly accuracy)

---

## Core features

- **ML anomaly detection**  
  - Uses an autoencoder trained on synthetic normal physiological data to detect anomalies.
  - Tried different losses/activations and kept the one that gave lowest validation loss.
  - Identifies hypoxia-like conditions in divers based on heart rate and blood oxygen levels (anomalous data is also synthetic).
  - Model is slightly aggressive and, when presented with values at the threshold, errs on the side of anomalous in order to prioritise diver safety over false positives.
       - **THIS FEATURE HAS BROKEN DUE TO RECENT CODE UPDATES. INVESTIGATING REASONs. (28/09)**

- **Optimised for edge computing**  
  - Converts the trained model to TFLite for deployment on microcontrollers.
  - Lightweight architecture with minimal computational overhead.
  - Uses in-memory execution rather than persistent storage (done for evaluation purposes; in a real-world deployment, persistent storage is needed to save the *.tflite model).

- **Evaluation and visualisation**
  - The autoencoder ANN was chosen after considering the input data and the accuracy of the classification needed, weighed against other models such as decision trees, support vector machines, and traditional statistical anomaly detection methods.
  - The autoencoder reconstructs an input sample, after which the reconstruction error (MSE between the original and reconstructed data) is calculated, with a pre-computed threshold used to classify the sample.

- **Using in-memory TFLite model**
  - Uses in-memory TFLite model instead of saving the model, making the program more efficient by skipping the disk I/O operation.
  - Persistent storage is not required (at this stage) since the model is loaded dynamically during execution.

---

## Limitations

- **Reproducibility constraints**
  - Reproducibility was attempted through random seeds and disabling oneDNN (which was causing nondeterminism due to floating-point round-off errors) but minor non-determinism still remains in TensorFlow's training operations. However, due to timing constraints, scope of the problem, and the relatively small run-to-run accuracy fluctuations, this was not addressed further. The hackathon Hub Leader further confirmed that this level of variance is expected for shallow models and does not affect the validity of the demo.
  - Overall accuracy fluctuations were observed to only be in the range of (96±3)%, with 85% of tests achieving 100% anomaly detection accuracy.
    - Of course, instances of 100% are only due to the simplicity of the data and model (more realistic readings/fluctuations would give lower metrics), though the current accuracy readings serve as a useful benchmark on the model's ease of improvement.
   
- **Reproducibility constraints**
  - We attempted reproducibility through random seeds and by disabling oneDNN (which caused nondeterminism due to floating-point round-off errors). Minor run-to-run differences remain in TensorFlow’s training, but the fluctuations are small for a shallow autoencoder.
  - Across multiple runs, overall accuracy varied between ~95% and ~99%. However, anomaly detection accuracy was much lower (around 50–60%), while normal classification exceeded 95%. This gap reflects the simplicity of our anomaly dataset: anomalies were synthetic, with only two features (HR, SpO₂), so the model overfit more easily to the “normal” class.
  - For a 5-day hackathon prototype, these results provided a reasonable proof-of-concept baseline, but further data realism and feature diversity would be needed for stronger anomaly recall.


- **Limited feature set and simplified model**  
  - Currently detects only heart rate and blood oxygen anomalies, however this is expected for a prototype demonstration.
  - Later, the model would need to be expanded to encompass more vital signs (e.g., breathing patterns, temperature).
  - Further optimisations are required for real world deep-sea conditions.

---

## Accuracy plots

- Trained on 10,000 samples and tested on 2000.
- Currently investigating why anomaly detection accuracy has dipped (28/09)

### Example run 1: 
#### 95.42% overall accuracy (60% anomaly detection, 95.6% normal classification)
![Sample plot 1](docs/assets/sample-plot-example-1.png)

### Example run 2: 
#### 99.55% overall accuracy (50% anomaly detection, 99.8% normal classification)

![Sample plot 3](docs/assets/sample-plot-example-3.png)

### Example run 3: 
#### 96.37%% overall accuracy (50%% anomaly detection, 96.60% normal classification)
![Sample plot 2](docs/assets/sample-plot-example-2.png)
