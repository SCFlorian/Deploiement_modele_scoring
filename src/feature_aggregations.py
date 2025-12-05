import numpy as np
import pandas as pd
import gc

# =========================================================
#  ONE-HOT ENCODER UTILITAIRE (corrigé pour éviter les bool)
# =========================================================
def one_hot_encoder(df, nan_as_category=True):
    """Encodage one-hot propre (uint8, pas de booléens)."""
    df = df.copy()
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == "object"]

    #  Correction clé : dtype = uint8 pour avoir des 0/1 numériques
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category, dtype=np.uint8)

    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# =========================================================
#  1. BUREAU + BUREAU BALANCE
# =========================================================
def process_bureau_and_balance(bureau, bureau_balance):
    bureau = bureau.copy()
    bureau_balance = bureau_balance.copy()

    bb, bb_cat = one_hot_encoder(bureau_balance)
    bureau, bureau_cat = one_hot_encoder(bureau)

    # --- Bureau_balance aggregations
    bb_aggregations = {"MONTHS_BALANCE": ["min", "max", "size"]}
    for col in bb_cat:
        bb_aggregations[col] = ["mean"]

    bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_aggregations)
    bb_agg.columns = pd.Index([f"{e[0]}_{e[1].upper()}" for e in bb_agg.columns.tolist()])

    bureau = bureau.join(bb_agg, on="SK_ID_BUREAU", how="left")
    bureau.drop(["SK_ID_BUREAU"], axis=1, inplace=True)

    # --- Agrégations principales
    num_aggregations = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
    }
    cat_aggregations = {cat: ["mean"] for cat in bureau_cat}
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ["mean"]

    bureau_agg = bureau.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index([f"BURO_{e[0]}_{e[1].upper()}" for e in bureau_agg.columns.tolist()])

    # --- Sous-ensembles Active / Closed
    for status, prefix in [("CREDIT_ACTIVE_Active", "ACTIVE"), ("CREDIT_ACTIVE_Closed", "CLOSED")]:
        subset = bureau[bureau.get(status, 0) == 1]
        agg = subset.groupby("SK_ID_CURR").agg(num_aggregations)
        agg.columns = pd.Index([f"{prefix}_{e[0]}_{e[1].upper()}" for e in agg.columns.tolist()])
        bureau_agg = bureau_agg.join(agg, on="SK_ID_CURR", how="left")

    #  Conversion bool → uint8 pour éviter les False
    for df in [bureau_agg]:
        bool_cols = df.select_dtypes(include="bool").columns
        if len(bool_cols) > 0:
            df[bool_cols] = df[bool_cols].astype(np.uint8)

    del bureau, bb, bb_agg, subset, agg
    gc.collect()
    return bureau_agg.reset_index()


# =========================================================
#  2. PREVIOUS APPLICATIONS
# =========================================================
def process_previous_applications(prev):
    prev = prev.copy()

    # Suppression de colonnes avec 99% de valeurs manquantes
    prev = prev.drop(columns = 'RATE_INTEREST_PRIMARY')
    prev = prev.drop(columns = 'RATE_INTEREST_PRIVILEGED')
    prev, cat_cols = one_hot_encoder(prev)

    cols_365 = [
        "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE", "DAYS_TERMINATION"
    ]
    prev[cols_365] = prev[cols_365].replace(365243, np.nan)

    prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / (prev["AMT_CREDIT"] + 1e-5)

    num_aggregations = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
        "DAYS_DECISION": ["min", "max", "mean"],
    }
    cat_aggregations = {cat: ["mean"] for cat in cat_cols}

    prev_agg = prev.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index([f"PREV_{e[0]}_{e[1].upper()}" for e in prev_agg.columns.tolist()])

    for status, prefix in [("NAME_CONTRACT_STATUS_Approved", "APPROVED"), ("NAME_CONTRACT_STATUS_Refused", "REFUSED")]:
        subset = prev[prev.get(status, 0) == 1]
        agg = subset.groupby("SK_ID_CURR").agg(num_aggregations)
        agg.columns = pd.Index([f"{prefix}_{e[0]}_{e[1].upper()}" for e in agg.columns.tolist()])
        prev_agg = prev_agg.join(agg, on="SK_ID_CURR", how="left")

    #  Conversion bool → uint8
    bool_cols = prev_agg.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        prev_agg[bool_cols] = prev_agg[bool_cols].astype(np.uint8)

    del prev, subset, agg
    gc.collect()
    return prev_agg.reset_index()


# =========================================================
#  3. POS CASH BALANCE
# =========================================================
def process_pos_cash(pos):
    pos = pos.copy()
    pos, cat_cols = one_hot_encoder(pos)

    aggregations = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
        **{cat: ["mean"] for cat in cat_cols},
    }

    pos_agg = pos.groupby("SK_ID_CURR").agg(aggregations)
    pos_agg.columns = pd.Index([f"POS_{e[0]}_{e[1].upper()}" for e in pos_agg.columns.tolist()])
    pos_agg["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()

    #  Conversion bool → uint8
    bool_cols = pos_agg.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        pos_agg[bool_cols] = pos_agg[bool_cols].astype(np.uint8)

    del pos
    gc.collect()
    return pos_agg.reset_index()


# =========================================================
#  4. INSTALLMENTS PAYMENTS
# =========================================================
def process_installments(ins):
    ins = ins.copy()
    ins, cat_cols = one_hot_encoder(ins)

    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / (ins["AMT_INSTALMENT"] + 1e-5)
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)

    aggregations = {
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["mean", "var"],
        "PAYMENT_DIFF": ["mean", "var"],
        **{cat: ["mean"] for cat in cat_cols},
    }

    ins_agg = ins.groupby("SK_ID_CURR").agg(aggregations)
    ins_agg.columns = pd.Index([f"INSTAL_{e[0]}_{e[1].upper()}" for e in ins_agg.columns.tolist()])
    ins_agg["INSTAL_COUNT"] = ins.groupby("SK_ID_CURR").size()

    #  Conversion bool → uint8
    bool_cols = ins_agg.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        ins_agg[bool_cols] = ins_agg[bool_cols].astype(np.uint8)

    del ins
    gc.collect()
    return ins_agg.reset_index()


# =========================================================
#  5. CREDIT CARD BALANCE
# =========================================================
def process_credit_card(cc):
    cc = cc.copy()
    cc, cat_cols = one_hot_encoder(cc)
    cc.drop(["SK_ID_PREV"], axis=1, inplace=True, errors="ignore")

    cc_agg = cc.groupby("SK_ID_CURR").agg(["min", "max", "mean", "sum", "var"])
    cc_agg.columns = pd.Index([f"CC_{e[0]}_{e[1].upper()}" for e in cc_agg.columns.tolist()])
    cc_agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()

    #  Conversion bool → uint8
    bool_cols = cc_agg.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        cc_agg[bool_cols] = cc_agg[bool_cols].astype(np.uint8)

    del cc
    gc.collect()
    return cc_agg.reset_index()