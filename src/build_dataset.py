import numpy as np
import pandas as pd
import gc

def build_dataset(full_data,
                  bureau_agg, previous_agg, pos_agg,
                  install_agg, credit_agg):
    """
    Fusionne toutes les sous-tables agrégées avec les fichiers application_train/test,
    puis harmonise les types pour un dataset propre et exploitable.
    """

    print(" Fusion des sous-tables...")

    def safe_merge(base_df, add_df, name):
        before = base_df.shape
        base_df = base_df.merge(add_df, on='SK_ID_CURR', how='left')
        after = base_df.shape
        print(f"    Fusion {name:15s} | avant: {before}, après: {after}")
        return base_df

    # --- FUSIONS ---
    full_data = safe_merge(full_data, bureau_agg, 'bureau')

    full_data = safe_merge(full_data, previous_agg, 'previous')

    full_data = safe_merge(full_data, pos_agg, 'pos_cash')

    full_data = safe_merge(full_data, install_agg, 'installments')

    full_data = safe_merge(full_data, credit_agg, 'credit_card')

    print("\n Fusion terminée avec succès !")
    print(f"   → TRAIN shape : {full_data.shape}")

    full_data.to_csv("raw_fused_data.csv", index=False)
    print("   → Fichier brut fusionné sauvegardé : raw_fused_data.csv")
    
    # -------------------------------------------------------------------------
    #  Harmonisation des types après fusion
    # -------------------------------------------------------------------------
    print("\n Harmonisation post-fusion...")

    def harmonize_types(df, df_name):
        # Conversion bool → int
        bool_cols = df.select_dtypes(include="bool").columns
        if len(bool_cols):
            df[bool_cols] = df[bool_cols].astype(np.uint8)
            print(f"    {len(bool_cols)} colonnes booléennes converties en uint8 ({df_name})")

        # Conversion des chaînes 'True'/'False' → 1.0/0.0
        obj_cols = df.select_dtypes(include="object").columns
        converted = 0
        for c in obj_cols:
            unique_vals = set(df[c].dropna().unique())
            if unique_vals.issubset({'True', 'False'}):
                df[c] = df[c].map({'True': 1.0, 'False': 0.0})
                converted += 1
        if converted:
            print(f"    {converted} colonnes object converties en 0/1 ({df_name})")

        return df

    full_data = harmonize_types(full_data, "TRAIN")

    # -------------------------------------------------------------------------
    # Correction finale : suppression des 'False' résiduels
    # -------------------------------------------------------------------------
    print("\n Vérification et correction finale des valeurs 'False'...")

    def fix_false_values(df, df_name):
        mask = df.apply(lambda col: col.isin([False, 'False']).any())
        cols = mask[mask].index.tolist()

        if len(cols) > 0:
            for c in cols:
                df[c] = np.where(df[c].isin([False, 'False']), 0.0, df[c])
                try:
                    df[c] = df[c].astype(float)
                except Exception:
                    pass
            print(f"    {len(cols)} colonnes corrigées dans {df_name}")
        else:
            print(f"    Aucune valeur 'False' détectée dans {df_name}")
        return df

    full_data = fix_false_values(full_data, "TRAIN")

    # -------------------------------------------------------------------------
    # Vérifications finales
    # -------------------------------------------------------------------------
    print("\n Vérification finale :")
    print(f"   → Colonnes bool restantes TRAIN : {len(full_data.select_dtypes(include='bool').columns)}")

    print(f"\n Nettoyage complet — datasets prêts pour imputation et modélisation !")

    gc.collect()
    
    return full_data