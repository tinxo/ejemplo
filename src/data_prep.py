import pandas as pd
import os
import yaml

INPUT_PATH = "data/raw/sample.csv"
OUTPUT_PATH = "data/processed/clean.csv"

df = pd.read_csv(INPUT_PATH)

with open("params.yaml") as f:  # <- mejor no usar ../ si corrés desde raíz del repo
    params = yaml.safe_load(f)

df_clean = df.copy()

if params["limpieza_param"].get("eliminar_duplicados", True):
    df_clean = df_clean.drop_duplicates()

if params["limpieza_param"].get("eliminar_nulos", True):
    df_clean = df_clean.dropna()

# Guardar resultado
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_clean.to_csv(OUTPUT_PATH, index=False)
