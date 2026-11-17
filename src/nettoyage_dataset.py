import pandas as pd
import numpy as np

TARGET = "Yield"

# Production supprimée des variables explicatives
NUMERIC = [
    "Crop_Year",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

CATEGORICAL = [
    "Crop",
    "Season",
    "State"
]

FEATURES = CATEGORICAL + NUMERIC
REQUIRED = FEATURES + [TARGET]


def nettoyer_features(df):
    '''
    Règles communes Train / Production.
    Les valeurs non conformes deviennent NaN.
    Aucune observation supprimée ici.
    La variable Production n'est pas utilisée.
    '''

    d = df.copy()

    for c in FEATURES:
        if c not in d.columns:
            d[c] = pd.NA

    # Variables catégorielles
    for c in CATEGORICAL:
        d[c] = (
            d[c]
            .astype("string")
            .str.strip()
            .str.title()
            .replace("", pd.NA)
        )

    # Variables numériques
    for c in NUMERIC:
        d[c] = pd.to_numeric(
            d[c], errors="coerce"
        )

        d[c] = d[c].replace(
            [np.inf, -np.inf], np.nan
        )

    # Valeurs négatives interdites
    for c in [
        "Area",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide"
    ]:
        d.loc[d[c] < 0, c] = np.nan

    # Année entière entre 1900 et 2100
    annee = d["Crop_Year"]

    d.loc[
        (annee < 1900) |
        (annee > 2100) |
        ((annee % 1) != 0),
        "Crop_Year"
    ] = np.nan

    return d[FEATURES].copy()


def nettoyer_dataset(df):
    '''
    Nettoyage historique du dataset avec Yield.
    Production est exclue des données finales.
    '''

    d = df.copy()

    if TARGET not in d.columns:
        raise ValueError(
            "La colonne Yield est absente."
        )

    # Supprimer les doublons historiques
    d = d.drop_duplicates().copy()

    # Nettoyer les 8 variables explicatives
    x = nettoyer_features(d)

    # Nettoyer la variable cible
    y = pd.to_numeric(
        d[TARGET], errors="coerce"
    )

    y = y.replace(
        [np.inf, -np.inf], np.nan
    )

    # Conserver les observations valides
    ok = (
        y.notna() &
        (y >= 0) &
        x.notna().any(axis=1)
    )

    result = x.loc[ok].copy()
    result[TARGET] = y.loc[ok].astype(float)

    return result.reset_index(drop=True)
