#!/usr/bin/env python3
import pandas as pd
from sklearn.ensemble import IsolationForest
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_anomalies(df, column, contamination=0.05):
    """
    Detect anomalies in a specified column using Isolation Forest.
    """
    try:
        X = df[[column]].dropna()
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X)
        df['anomaly'] = model.predict(df[[column]])
        anomalies = df[df['anomaly'] == -1]
        logging.info(f"Anomaly detection complete. Found {len(anomalies)} anomalies.")
        return anomalies
    except Exception as e:
        logging.error(f"Error detecting anomalies: {e}")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection Utility for Data Pipeline")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--column", required=True, help="Column name to analyze for anomalies")
    parser.add_argument("--contamination", type=float, default=0.05, help="Contamination rate (default: 0.05)")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
        logging.info(f"Input data loaded from {args.input}.")
    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        return

    anomalies = detect_anomalies(df, args.column, args.contamination)
    if not anomalies.empty:
        logging.info("Anomalies detected:")
        print(anomalies)
    else:
        logging.info("No anomalies detected.")

if __name__ == "__main__":
    main()
