import numpy as np
import pandas as pd
import gc
from sklearn.impute import SimpleImputer

def impute_numeric_only(new_df):
    """
    Impute uniquement les colonnes numériques (catégorielles déjà encodées).
    - Imputation : médiane
    """
    print(" Début de l’imputation (numérique uniquement)...")

    num_cols = [c for c in new_df.columns if new_df[c].dtype in ['int64', 'float64']
                and c not in ['SK_ID_CURR', 'TARGET']]

    imputer = SimpleImputer(strategy='median')

    new_df[num_cols] = imputer.fit_transform(new_df[num_cols])

    print(" Imputation terminée avec succès")
    print(f"   → TRAIN shape : {new_df.shape}")

    return new_df