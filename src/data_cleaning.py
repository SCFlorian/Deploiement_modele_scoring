import numpy as np
import pandas as pd

def cleaning_application_test_train(df):
    df = df.copy()

    # === Nettoyage des valeurs aberrantes ===
    df = df[df["CODE_GENDER"] != "XNA"]

    # === Conversion des jours en années ===
    for col in ["DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"]:
        df[col] = df[col] / -365

    # === DAYS_EMPLOYED : transformation + valeurs spéciales ===
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"] / -365
    # === Suppression d'une corrélation forte ===
    df = df.drop(columns = 'REGION_RATING_CLIENT')
    # === Nettoyage des chaînes ===
    to_clean = [
        "NAME_CONTRACT_TYPE", "NAME_TYPE_SUITE", "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE", "WALLSMATERIAL_MODE"
    ]
    for col in to_clean:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).replace(" ", "_").replace("/", "").replace(":", "") if isinstance(x, str) else x)

    df["EMERGENCYSTATE_MODE"] = df["EMERGENCYSTATE_MODE"].replace({"Yes": "Y", "No": "N"})

    # === Feature engineering de base ===
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    # === EXT_SOURCE features ===
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCES_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCES_STD"] = df[ext_cols].std(axis=1)
    df["SCORE_PRODUCT"] = df[ext_cols[0]] * df[ext_cols[1]] * df[ext_cols[2]]

    # === Encodage binaire simple ===
    for col in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[col], _ = pd.factorize(df[col])

    # === Encodage catégoriel (propre, cohérent avec les sous-tables) ===
    cat_cols = [col for col in df.columns if df[col].dtype == "object"]

    # Remplir les NaN par "MISSING" avant one-hot
    df[cat_cols] = df[cat_cols].fillna("MISSING")

    #  One-hot encoding propre (dtype = uint8 pour éviter les bool)
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False, dtype=np.uint8)

    # === Vérification finale ===
    bool_cols = df.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(np.uint8)
        print(f" {len(bool_cols)} colonnes booléennes converties en uint8")

    print(f" Nettoyage terminé → shape : {df.shape}")

    return df
