import pandas as pd
import pickle
import json
import os
import yaml
import mlflow
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report

def log_evaluation_to_mlflow(run_id, metrics):
    """Registra las métricas de evaluación en el mismo run de entrenamiento"""
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("test_accuracy", metrics["accuracy"])
        mlflow.log_metric("test_precision", metrics["precision"])
        mlflow.log_metric("test_recall", metrics["recall"])
        mlflow.log_metric("test_f1_score", metrics["f1_score"])

if __name__ == "__main__":
    # Cargar parámetros
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Configurar MLflow
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    
    # Cargar modelo
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Cargar metadata del modelo para obtener el run_id
    with open("models/model_metadata.json", "r") as f:
        metadata = json.load(f)
        run_id = metadata["mlflow_run_id"]
    
    # Cargar datos de test separados
    df_test = pd.read_csv("data/raw/test.csv")
    X_test = df_test.drop(['subscribed', 'id'], axis=1)
    y_test = df_test['subscribed']
    
    accuracy, report = evaluate_model(model, X_test, y_test)
    
    # Organizar métricas de forma ordenada
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(report['weighted avg']['precision'], 4),
        "recall": round(report['weighted avg']['recall'], 4),
        "f1_score": round(report['weighted avg']['f1-score'], 4),
        "support": report['weighted avg']['support'],
        "class_metrics": {
            "no": {
                "precision": round(report['no']['precision'], 4),
                "recall": round(report['no']['recall'], 4),
                "f1_score": round(report['no']['f1-score'], 4),
                "support": report['no']['support']
            },
            "yes": {
                "precision": round(report['yes']['precision'], 4),
                "recall": round(report['yes']['recall'], 4),
                "f1_score": round(report['yes']['f1-score'], 4),
                "support": report['yes']['support']
            }
        }
    }
    
    # Crear directorio para métricas
    os.makedirs("metrics", exist_ok=True)
    
    # Guardar métricas en JSON (para DVC)
    with open("metrics/evaluation.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Registrar métricas de evaluación en MLflow
    log_evaluation_to_mlflow(run_id, metrics)
    
    print("✓ Evaluación completada")
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ Precision: {metrics['precision']}")
    print(f"✓ Recall: {metrics['recall']}")
    print(f"✓ F1-Score: {metrics['f1_score']}")
    print("✓ Métricas guardadas en metrics/evaluation.json")
    print(f"✓ Métricas registradas en MLflow (Run ID: {run_id})")