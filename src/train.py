import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import yaml

def train_model(df):
    X = df.drop(['subscribed', 'id'], axis=1)
    y = df['subscribed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        clf = RandomForestClassifier(random_state=42, criterion='entropy')
        clf.fit(X_train, y_train)
        
        # Calcular métricas de entrenamiento
        train_accuracy = clf.score(X_train, y_train)
        val_accuracy = clf.score(X_test, y_test)
        
        # Registrar parámetros y métricas en MLflow
        mlflow.log_param("n_estimators", clf.n_estimators)
        mlflow.log_param("random_state", clf.random_state)
        mlflow.log_param("criterion", clf.criterion)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        
        # Crear input example para la signatura
        input_example = X_train.iloc[:5]
        
        # Registrar el modelo en MLflow (sin registered_model_name para evitar warnings)
        mlflow.sklearn.log_model(
            clf, 
            name="model",
            input_example=input_example.astype('float64'),  # Añadir ejemplo para inferir signatura
            registered_model_name="SubscriptionPredictor"
        )
        
        # Obtener run_id para metadata
        run_id = mlflow.active_run().info.run_id
        
        return clf, X_test, y_test, run_id

if __name__ == "__main__":
    # Cargar parámetros
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Configurar MLflow
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    
    # Crear experimento si no existe (evita el warning INFO)
    try:
        experiment = mlflow.get_experiment_by_name(params['mlflow']['experiment_name'])
        if experiment is None:
            mlflow.create_experiment(params['mlflow']['experiment_name'])
    except Exception:
        pass
        
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    
    # Cargar datos procesados
    df = pd.read_csv("data/processed/clean.csv")
    
    # Entrenar modelo
    model, X_test, y_test, run_id = train_model(df)
    
    # Crear directorio si no existe
    os.makedirs("models", exist_ok=True)
    
    # Guardar modelo para DVC
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Guardar metadata del experimento
    metadata = {
        "mlflow_run_id": run_id,
        "experiment_name": params['mlflow']['experiment_name'],
        "model_name": "SubscriptionPredictor"
    }
    
    with open("models/model_metadata.json", "w") as f:
        import json
        json.dump(metadata, f, indent=2)
    
    print("✓ Modelo entrenado y guardado")
    print(f"✓ MLflow Run ID: {run_id}")
    print("✓ Modelo registrado en MLflow como 'SubscriptionPredictor'")