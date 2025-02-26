# edge-ai-utilities

## Description
The **edge-ai-utilities** repository contains foundational AI utilities designed to support robust AI-driven applications. This repository includes tools for both training machine learning models and monitoring data pipeline quality. It demonstrates essential capabilities such as model training with hyperparameter tuning and anomaly detection using the Isolation Forest algorithm.

## Files
- **model_training.py**  
  A versatile utility for training machine learning models. It supports both linear regression and a hyperparameter-tuned random forest model, complete with cross-validation, detailed logging, and model persistence.
- **anomaly_detector.py**  
  A data pipeline anomaly detection tool that uses the Isolation Forest algorithm to identify outliers in numerical data—serving as an early warning system before data is loaded into systems like Snowflake.

## Prerequisites
- Python 3.7+
- Required libraries: `pandas`, `scikit-learn`, `joblib`
- (Optional) Use a virtual environment and install dependencies via a `requirements.txt` file.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/edge-ai-utilities.git
   cd edge-ai-utilities
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
* To train a machine learning model (for example, using the random forest model):
  ```bash
  python model_training.py --model rf --save trained_model.pkl
  ```

* To detect anomalies in a CSV file:
  ```bash
  python anomaly_detector.py --input data.csv --column numeric_column_name --contamination 0.05
  ```

* For detailed usage instructions, use the `--help` flag:
  ```bash
  python model_training.py --help
  ```

## License
This project is licensed under the MIT License.