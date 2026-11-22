"""Calibration empirique des seuils Data Drift et performance pour AgriYield MLOps.

Exemple :
python src/calibrer_seuils.py
python src/calibrer_seuils.py --model models/model_v1.pkl --preprocessor models/preprocessing.pkl

Le modèle doit être soit un Pipeline sklearn complet, soit un estimateur accompagné
par --preprocessor. Les fichiers sont chargés avec joblib : ne chargez que des
artefacts de confiance.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
CATEGORICAL = ["Crop", "Season", "State"]
FEATURES = CATEGORICAL + NUMERIC
METRICS = ["MAE", "RMSE", "R2", "MedAE"]


def section(n: int, title: str) -> None:
    print(f"\n{'=' * 72}\nÉTAPE {n} — {title}\n{'=' * 72}")


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier {name} introuvable : {path}")
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + ["Yield"] if c not in df.columns]
    if missing:
        raise ValueError(f"{name} : colonnes manquantes : {missing}")
    df = df.copy()
    for col in NUMERIC + ["Yield"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL:
        df[col] = df[col].astype("string").fillna("<MANQUANT>").str.strip().replace("", "<MANQUANT>").astype(str)
    if df[NUMERIC + ["Yield"]].isna().any().any():
        raise ValueError(f"{name} : valeurs numériques manquantes/non valides. Nettoyer avant calibration.")
    print(f"  ✓ {name:<12} : {len(df):>6} lignes | {path}")
    return df


def ks_stat(a: pd.Series, b: pd.Series) -> float:
    return float(ks_2samp(a.to_numpy(), b.to_numpy()).statistic)


def chi2_stat(a: pd.Series, b: pd.Series) -> float:
    """Chi-deux d'homogénéité : mêmes catégories, tailles des groupes égales."""
    cats = sorted(set(a.unique()) | set(b.unique()))
    counts_a = a.value_counts().reindex(cats, fill_value=0).to_numpy()
    counts_b = b.value_counts().reindex(cats, fill_value=0).to_numpy()
    table = np.vstack([counts_a, counts_b])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 0.0
    # On utilise uniquement la statistique comme score calibré empiriquement.
    # Si certains effectifs attendus sont faibles, ne PAS interpréter la p-value
    # du Chi-deux classique. Les catégories rares sont regroupées en amont.
    return float(chi2_contingency(table, correction=False).statistic)


def group_rare_categories(train: pd.DataFrame, min_count: int) -> dict[str, list[str]]:
    # Règle fixe apprise sur Train, réutilisée sur toutes les données.
    kept = {}
    for col in CATEGORICAL:
        counts = train[col].value_counts()
        kept[col] = counts[counts >= min_count].index.tolist()
        print(f"  {col:<16} : {len(kept[col])} catégories conservées ; autres → <AUTRES>")
    return kept


def apply_categories(df: pd.DataFrame, kept: dict[str, list[str]]) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL:
        df[col] = df[col].where(df[col].isin(kept[col]), "<AUTRES>")
    return df


def drift_stats(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float]:
    result = {f"KS::{c}": ks_stat(a[c], b[c]) for c in NUMERIC}
    for col in CATEGORICAL:
        result[f"CHI2::{col}"] = chi2_stat(a[col], b[col])
    return result


def predict(model, preprocessor, df: pd.DataFrame) -> np.ndarray:
    x = df[FEATURES]
    if preprocessor is not None:
        x = preprocessor.transform(x)
    y = np.asarray(model.predict(x), dtype=float).reshape(-1)
    if len(y) != len(df) or not np.isfinite(y).all():
        raise ValueError("Prédictions non valides du modèle V1.")
    return y


def performance_stats(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "MedAE": float(median_absolute_error(actual, predicted)),
    }


def decide(stats: dict[str, float], thresholds: dict[str, dict]) -> dict[str, bool]:
    return {
        name: (value < thresholds[name]["seuil"] if thresholds[name]["sens"] == "inferieur" else value > thresholds[name]["seuil"])
        for name, value in stats.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration et validation des seuils MLOps")
    parser.add_argument("--model", type=Path, default=Path("models/model_v1.pkl"), help="Chemin du modèle V1 (.pkl)")
    parser.add_argument("--preprocessor", type=Path, default=Path("models/preprocessing.pkl"), help="Chemin du prétraitement V1 (.pkl)")
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--taille-lot", type=int, default=200)
    parser.add_argument("--min-categorie", type=int, default=100, help="Effectif minimum Train pour garder une catégorie")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repetitions < 20 or args.taille_lot < 30:
        parser.error("Choisir au moins 20 répétitions et des lots d'au moins 30 observations.")
    rng = np.random.default_rng(args.seed)

    section(1, "CHARGEMENT DES DONNÉES ET DU MODÈLE V1")
    train = load_csv(ROOT / "data/train.csv", "Train")
    val = load_csv(ROOT / "data/validation.csv", "Validation")
    sim = load_csv(ROOT / "data/simulation.csv", "Simulation")
    if len(train) < 2 * args.taille_lot or len(val) < args.taille_lot or len(sim) < args.taille_lot:
        raise ValueError("Taille de lot trop élevée par rapport aux données disponibles.")
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    pre_path = (args.preprocessor if args.preprocessor.is_absolute() else ROOT / args.preprocessor) if args.preprocessor else None
    if not model_path.exists() or (pre_path and not pre_path.exists()):
        raise FileNotFoundError(f"Fichiers introuvables : modèle={model_path} (existe={model_path.exists()}), prétraitement={pre_path} (existe={pre_path.exists() if pre_path else None})")
    model = joblib.load(model_path)
    preprocessor = joblib.load(pre_path) if pre_path else None
    if not hasattr(model, "predict"):
        raise TypeError("Le fichier modèle ne contient pas un estimateur avec méthode predict().")
    if preprocessor is not None and not hasattr(preprocessor, "transform"):
        raise TypeError("Le fichier de prétraitement ne contient pas de méthode transform().")
    print(f"  ✓ Modèle V1 : {model_path}")
    print(f"  ✓ Prétraitement : {pre_path or 'intégré au modèle (Pipeline attendu)'}")
    print("  ℹ Test final : volontairement non utilisé.")

    # Pour le Chi-deux, les catégories rares sont regroupées avec une règle fixe.
    # Si les effectifs attendus restent < 5 sur un lot, réduire le nombre de
    # catégories conservées (augmenter --min-categorie) ou augmenter --taille-lot.
    print("\n  Préparation des catégories pour le Chi-deux :")
    kept = group_rare_categories(train, args.min_categorie)
    train_drift = apply_categories(train, kept)
    sim_drift = apply_categories(sim, kept)

    section(2, f"{args.repetitions} SIMULATIONS DE FONCTIONNEMENT NORMAL")
    records = []
    try:
        val_predictions = predict(model, preprocessor, val)
    except Exception as exc:
        raise RuntimeError("Échec des prédictions V1 sur Validation : vérifier l’ordre des colonnes et la compatibilité du prétraitement enregistré.") from exc
    for i in range(args.repetitions):
        # Deux groupes distincts de Train (pas de doublons dans la répétition).
        idx = rng.choice(len(train_drift), size=2 * args.taille_lot, replace=False)
        a = train_drift.iloc[idx[:args.taille_lot]]
        b = train_drift.iloc[idx[args.taille_lot:]]
        drift = drift_stats(a, b)
        v_idx = rng.choice(len(val), size=args.taille_lot, replace=False)
        perf = performance_stats(val["Yield"].to_numpy()[v_idx], val_predictions[v_idx])
        records.append({"repetition": i + 1, **drift, **perf})
        if (i + 1) % 10 == 0:
            print(f"  ✓ {i + 1:>3}/{args.repetitions} répétitions terminées")
    history = pd.DataFrame(records)
    print(f"  ✓ {args.repetitions} valeurs par statistique et métrique calculées.")

    section(3, "CALCUL DES SEUILS (P95 ET P5)")
    thresholds = {}
    for name in [*([f"KS::{c}" for c in NUMERIC]), *([f"CHI2::{c}" for c in CATEGORICAL]), *METRICS]:
        is_r2 = name == "R2"
        percentile = 5 if is_r2 else 95
        threshold = float(np.percentile(history[name], percentile))
        thresholds[name] = {
            "seuil": threshold,
            "percentile": percentile,
            "sens": "inferieur" if is_r2 else "superieur",
            "moyenne_simulations": float(history[name].mean()),
        }
        print(f"  {name:<22} | P{percentile:<2} = {threshold:>13.6f} | alerte si {'<' if is_r2 else '>'} seuil")

    section(4, "TEST DES TROIS SCÉNARIOS SUR SIMULATION")
    # Les données de Simulation ne servent pas à calculer les seuils.
    # 20 lots de test par scénario, avec échantillonnage indépendant par tirage.
    nb_tests = 20
    results = {"normal": [], "data_drift": [], "baisse_performance": []}
    sim_predictions = predict(model, preprocessor, sim)
    train_reference_idx = rng.choice(len(train_drift), size=args.taille_lot, replace=False)
    reference_batch = train_drift.iloc[train_reference_idx]
    for i in range(nb_tests):
        ix = rng.choice(len(sim), size=args.taille_lot, replace=False)
        current = sim_drift.iloc[ix].copy()
        real = sim["Yield"].to_numpy()[ix]
        pred = sim_predictions[ix]

        # Scénario 1 : aucun changement artificiel.
        drift_normal = decide(drift_stats(reference_batch, current), thresholds)
        perf_normal = decide(performance_stats(real, pred), thresholds)
        results["normal"].append({"drift": drift_normal, "performance": perf_normal})

        # Scénario 2 : décalage numérique + changement catégoriel.
        modified = current.copy()
        modified["Annual_Rainfall"] = modified["Annual_Rainfall"] * 1.8
        # Changement de distribution des saisons, sans modifier la référence.
        modified["Season"] = "Kharif" if "Kharif" in kept["Season"] else "<AUTRES>"
        drift_changed = decide(drift_stats(reference_batch, modified), thresholds)
        results["data_drift"].append({"drift": drift_changed})

        # Scénario 3 : dégradation artificielle des prédictions uniquement.
        scale = max(float(np.std(real)), 1.0)
        degraded = pred + rng.normal(0, 2.0 * scale, size=len(pred))
        perf_changed = decide(performance_stats(real, degraded), thresholds)
        results["baisse_performance"].append({"performance": perf_changed})

    section(5, "MESURE DE L'EFFICACITÉ ET SAUVEGARDE")
    normal = results["normal"]
    drifted = results["data_drift"]
    degraded = results["baisse_performance"]
    any_drift = lambda row: any(row["drift"].values())
    any_perf = lambda row: any(row["performance"].values())
    rates = {
        "fausses_alertes_drift_pct": 100 * np.mean([any_drift(r) for r in normal]),
        "fausses_alertes_performance_pct": 100 * np.mean([any_perf(r) for r in normal]),
        "detection_drift_pct": 100 * np.mean([any_drift(r) for r in drifted]),
        "detection_baisse_performance_pct": 100 * np.mean([any_perf(r) for r in degraded]),
        "detection_drift_Annual_Rainfall_pct": 100 * np.mean([r["drift"]["KS::Annual_Rainfall"] for r in drifted]),
        "detection_drift_Season_pct": 100 * np.mean([r["drift"]["CHI2::Season"] for r in drifted]),
    }
    for key, value in rates.items():
        print(f"  {key:<43} : {value:>5.1f} %")

    config_dir = ROOT / "config"
    report_dir = ROOT / "reports/calibration"
    config_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "version_modele": "V1",
        "date_calibration_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "repetitions": args.repetitions,
            "taille_lot": args.taille_lot,
            "seed": args.seed,
            "train": "data/train.csv",
            "validation": "data/validation.csv",
            "simulation": "data/simulation.csv",
            "modele": str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else str(model_path),
            "min_categorie": args.min_categorie,
            "categories_conservees": kept,
            "regle_alerte_globale": "au_moins_une_metrique_en_alerte",
        },
        "data_drift": {
            "numeriques": {col: {"test": "KS", **thresholds[f"KS::{col}"]} for col in NUMERIC},
            "categorielles": {col: {"test": "CHI2", **thresholds[f"CHI2::{col}"]} for col in CATEGORICAL},
        },
        "performance": {name: thresholds[name] for name in METRICS},
        "statut": "provisoire_a_valider_selon_resultats_scenarios",
    }
    (config_dir / "seuils.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    history.to_csv(report_dir / "historique_calibration.csv", index=False)
    evaluation = {
        "nombre_lots_par_scenario": nb_tests,
        "taille_lot": args.taille_lot,
        "resultats_pct": {k: float(v) for k, v in rates.items()},
        "note": "Les scénarios sont artificiels ; les résultats ne garantissent pas la performance en production.",
    }
    (report_dir / "evaluation_scenarios.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n  Fichiers générés :")
    for path in [config_dir / "seuils.json", report_dir / "historique_calibration.csv", report_dir / "evaluation_scenarios.json"]:
        print(f"  ✓ {path.relative_to(ROOT)}")
    print("\n  IMPORTANT : vérifier les taux de fausses alertes avant activation en production.")
    print("  IMPORTANT : pour un Chi-deux stable, utiliser les mêmes tailles de lots et regroupements de catégories.")
    print("  IMPORTANT : Crop_Year est discret ; le KS est calibré empiriquement.")


if __name__ == "__main__":
    main()
