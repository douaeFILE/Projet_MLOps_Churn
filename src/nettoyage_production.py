import pandas as pd
import numpy as np

from numbers import Real

from nettoyage_dataset import (
    FEATURES,
    CATEGORICAL,
    NUMERIC,
    nettoyer_features
)

MAX_MANQUANTS = 3

FORMATS_ATTENDUS = {
    "Crop": "Texte non vide, ex. Rice",
    "Season": "Texte non vide, ex. Kharif",
    "State": "Texte non vide, ex. Assam",
    "Crop_Year": "Année entière entre 1900 et 2100",
    "Area": "Nombre positif ou nul, ex. 150",
    "Annual_Rainfall": "Nombre positif ou nul, ex. 1200",
    "Fertilizer": "Nombre positif ou nul, ex. 200",
    "Pesticide": "Nombre positif ou nul, ex. 15"
}


def est_manquant(valeur):
    '''
    Détecter un champ réellement absent.
    '''

    if valeur is None:
        return True

    if isinstance(valeur, str):
        return valeur.strip() == ""

    if isinstance(valeur, (list, dict, tuple, set)):
        return False

    try:
        resultat = pd.isna(valeur)

        if isinstance(resultat, (bool, np.bool_)):
            return bool(resultat)

    except (TypeError, ValueError):
        pass

    return False


def valider_champ(champ, valeur):
    '''
    Retourne :
    - manquant : champ absent
    - valide : valeur conforme
    - invalide : mauvais type ou mauvaise valeur
    '''

    if est_manquant(valeur):
        return "manquant"

    # Variables catégorielles
    if champ in CATEGORICAL:

        if not isinstance(valeur, str):
            return "invalide"

        return "valide"

    # Variables numériques
    if isinstance(valeur, (bool, np.bool_)):
        return "invalide"

    if not isinstance(valeur, (str, Real, np.number)):
        return "invalide"

    try:
        nombre = float(valeur)

    except (ValueError, TypeError, OverflowError):
        return "invalide"

    if not np.isfinite(nombre):
        return "invalide"

    # Vérification de l'année
    if champ == "Crop_Year":

        if not nombre.is_integer():
            return "invalide"

        if not (1900 <= nombre <= 2100):
            return "invalide"

        return "valide"

    # Autres variables numériques positives ou nulles
    if champ in NUMERIC:

        if nombre < 0:
            return "invalide"

        return "valide"

    return "invalide"


def nettoyer_production(df):
    '''
    Validation et nettoyage des données de production.

    Une observation est acceptée si :
    - aucun champ renseigné n'est invalide
    - au maximum 3 champs sont manquants

    Production n'est pas une variable attendue.

    Retour :
        donnees_valides, avertissements
    '''

    donnees_brutes = df.copy()

    # Compléter les colonnes absentes
    for champ in FEATURES:

        if champ not in donnees_brutes.columns:
            donnees_brutes[champ] = pd.NA

    # Nettoyage commun Train / Production
    donnees_nettoyees = nettoyer_features(
        donnees_brutes
    )

    lignes_acceptees = []
    avertissements = []

    for position in range(len(donnees_brutes)):

        ligne = donnees_brutes.iloc[position]

        champs_manquants = []
        champs_invalides = []

        # Vérifier chaque champ
        for champ in FEATURES:

            statut = valider_champ(
                champ,
                ligne[champ]
            )

            if statut == "manquant":
                champs_manquants.append(champ)

            elif statut == "invalide":
                champs_invalides.append(champ)

        nombre_manquants = len(champs_manquants)

        # Observation acceptée
        if (
            len(champs_invalides) == 0
            and nombre_manquants <= MAX_MANQUANTS
        ):
            lignes_acceptees.append(position)
            continue

        # Observation refusée
        formats_a_corriger = {
            champ: FORMATS_ATTENDUS[champ]
            for champ in champs_invalides
        }

        if nombre_manquants > MAX_MANQUANTS:

            for champ in champs_manquants:
                formats_a_corriger[champ] = (
                    FORMATS_ATTENDUS[champ]
                )

        motifs = []

        if champs_invalides:
            motifs.append(
                "Certains champs ne respectent pas "
                "le type ou le format attendu."
            )

        if nombre_manquants > MAX_MANQUANTS:
            motifs.append(
                "Plus de 3 champs sont manquants."
            )

        avertissements.append({
            "ligne": position,
            "nombre_manquants": nombre_manquants,
            "champs_manquants": champs_manquants,
            "champs_invalides": champs_invalides,
            "message": (
                " ".join(motifs) +
                " Veuillez corriger votre saisie."
            ),
            "formats_attendus": formats_a_corriger
        })

    # Conserver les observations acceptées
    donnees_valides = donnees_nettoyees.iloc[
        lignes_acceptees
    ].copy()

    return donnees_valides, avertissements
