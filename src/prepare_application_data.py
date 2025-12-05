import numpy as np
import pandas as pd
import gc
from src.data_cleaning import cleaning_application_test_train


def prepare_application_data(app_train, app_test):
    """
    Fusionne application_train et application_test pour un nettoyage cohérent,
    puis renvoie un DataFrame unique prêt pour enrichissement.

    - Concatène train et test (alignement des colonnes)
    - Applique le nettoyage et le feature engineering
    - Ne sépare PAS à la fin (on garde tout pour build_dataset)
    """

    print("=== Préparation du dataset complet (train + test) ===")

    # Ajout d'un repère pour pouvoir filtrer plus tard
    app_train["is_train"] = 1
    app_test["is_train"] = 0
    app_test["TARGET"] = np.nan  # pour aligner les colonnes

    # Fusion temporaire
    full_app = pd.concat([app_train, app_test], axis=0, ignore_index=True)
    print(f"→ Fusion train/test : {full_app.shape}")

    # Nettoyage et feature engineering
    full_app = cleaning_application_test_train(full_app)
    print(f"→ Nettoyage terminé : {full_app.shape}")

    del app_train, app_test
    gc.collect()

    return full_app
