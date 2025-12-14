# -*- coding: utf-8 -*-
"""
Modelling Script - Drug Classification (Razif)
Kriteria 2: Membangun Model Machine Learning (Basic Level)
Dataset: drug200 (hasil preprocessing: X_train/X_test/y_train/y_test)
"""

import os
import warnings
import joblib 
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# === IMPORTANT: base directory untuk file CSV & model ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# CONFIG MLFLOW (LOCAL DEFAULT + OPSIONAL DAGSHUB)
# =====================================================================
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD")

if MLFLOW_TRACKING_URI and MLFLOW_USERNAME and MLFLOW_PASSWORD:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("✓ Using REMOTE MLflow tracking server")
    print(f"  URI : {MLFLOW_TRACKING_URI}")
else:
    LOCAL_URI = "file:./mlruns"
    mlflow.set_tracking_uri(LOCAL_URI)
    print("⚠️ Remote MLflow not fully configured, using LOCAL tracking")
    print(f"  URI : {LOCAL_URI}")

EXPERIMENT_NAME = "Drug_Classification_MSML_Razif"
mlflow.set_experiment(EXPERIMENT_NAME)

# =====================================================================
# LOAD DATA
# =====================================================================
def load_data():
    """Load preprocessed data (X_train/X_test/y_train/y_test)"""

    print("=" * 60)
    print("LOADING PREPROCESSED DATA")
    print("=" * 60)

    X_train_path = os.path.join(BASE_DIR, "X_train.csv")
    X_test_path = os.path.join(BASE_DIR, "X_test.csv")
    y_train_path = os.path.join(BASE_DIR, "y_train.csv")
    y_test_path = os.path.join(BASE_DIR, "y_test.csv")

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()

    print(f"✓ X_train shape: {X_train.shape}")
    print(f"✓ X_test shape:  {X_test.shape}")
    print(f"✓ y_train shape: {y_train.shape}")
    print(f"✓ y_test shape:  {y_test.shape}")

    return X_train, X_test, y_train, y_test


# =====================================================================
# TRAIN MODEL
# =====================================================================
def train_model(X_train, X_test, y_train, y_test, model_name="DecisionTree"):
    """Train model dan log ke MLflow (autolog + manual metric logging)"""

    print("\n" + "=" * 60)
    print(f"TRAINING MODEL: {model_name}")
    print("=" * 60)

    with mlflow.start_run(run_name=f"{model_name}_Model"):

        # Autolog harus di dalam start_run
        mlflow.sklearn.autolog(log_models=True)

        # Pilih model
        if model_name == "DecisionTree":
            model = DecisionTreeClassifier(
                random_state=42,
                max_depth=10,
                min_samples_split=5,
            )
        elif model_name == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Train
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Hitung metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Print hasil
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE")
        print("=" * 60)
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")

        # Logging manual ke MLflow (agar rapi di UI)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)

        mlflow.log_param("dataset_total", len(X_train) + len(X_test))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        # =======================
        # SIMPAN MODEL UNTUK API
        # =======================
        if model_name == "RandomForest":
            serve_path = os.path.join(BASE_DIR, "model.pkl")
            joblib.dump(model, serve_path)
            print(f"\n✓ Final serving model disimpan di: {serve_path}")

        print("\n✓ Model logged to MLflow!")
        print(f"✓ Run name  : {model_name}_Model")
        print(f"✓ Experiment: {EXPERIMENT_NAME}")

        return model


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("\n" + "=" * 60)
    print("DRUG CLASSIFICATION - MLFLOW TRAINING (RAZIF)")
    print("=" * 60)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_data()

    # 2. Train beberapa model
    models = ["DecisionTree", "RandomForest"]

    for model_name in models:
        train_model(X_train, X_test, y_train, y_test, model_name)
        print("\n")

    print("=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)

    print("\nUntuk melihat hasil di MLflow UI:")
    print("1. Di terminal (local tracking) jalankan:")
    print("   mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns")
    print("2. Buka browser: http://127.0.0.1:5000")
    print("3. Kalau pakai DagsHub, buka halaman MLflow di repo DagsHub-mu.")


if __name__ == "__main__":
    main()
