---
noteId: "3e58d160e03611f1a3c417ce6268340c"
tags: []

---

# Dossier tests/

Ce dossier contient les tests automatiques permettant de vérifier le bon fonctionnement des différentes composantes du projet MLOps.

| Fichier | Description |
|---|---|
| `test_preprocessing.py` | Vérifie le nettoyage, la transformation et la préparation des données. |
| `test_model.py` | Vérifie le chargement du modèle et la validité de ses prédictions. |
| `test_api.py` | Vérifie les réponses et le fonctionnement des routes FastAPI. |
| `test_monitoring.py` | Vérifie les règles de détection de dérive et de génération des alertes. |

Les tests sont exécutés avec `pytest`.

**Objectif :** détecter les erreurs avant le déploiement et garantir la fiabilité du système au cours de son évolution.