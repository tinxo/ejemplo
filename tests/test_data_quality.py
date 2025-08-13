import sys
import pandas as pd
from pandera.pandas import Column, DataFrameSchema

def test_schema():
    schema = DataFrameSchema({
        "age": Column(int),
        "income": Column(int),
        "subscribed": Column(str),
    })
    df = pd.read_csv("data/raw/sample.csv")
    schema.validate(df)

def test_data_quality():
    df = pd.read_csv("data/raw/sample.csv")
    assert df.isnull().sum().sum() == 0, "Data contains null values"
    assert df.duplicated().sum() == 0, "Data contains duplicate rows"
    assert df.shape[1] == 4, "Data does not have the expected number of columns"

def test_data_integrity():
    df = pd.read_csv("data/raw/sample.csv")
    assert df['age'].min() >= 0, "Age cannot be negative"
    assert df['income'].min() >= 0, "Income cannot be negative"
    assert df['subscribed'].isin(['yes', 'no']).all(), "Subscribed column contains invalid values"

if __name__ == "__main__":
    try:
        test_schema()
        test_data_quality()
        test_data_integrity()
        print("✓ All data validation checks passed")
        # Crear un archivo de salida para indicar que la validación fue exitosa
        with open("data/validation_passed.txt", "w") as f:
            f.write("Data validation completed successfully")
    except Exception as e:
        print(f"✗ Data validation failed: {e}")
        sys.exit(1)  # Esto hará que DVC falle el stage