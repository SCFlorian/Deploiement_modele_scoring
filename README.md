# Mise en production d'un modèle de scoring

---

## Description

Lors de ce projet nous allons mettre en production le meilleur modèle du projet **'Modele_scoring_MLFlow"** (disponible également dans un des repositories).

Les objectifs principaux :
1. Création d'une API robuste
2. Conteneurisation pour un déploiement fluide
3. Mise en place d'un monitoring fluide

---

## Structure du projet (en cours de construction)
```
Deploiement_modele_scoring/
 ├── .github/workflows/      # Configuration du pipeline CI/CD
 ├── data/                   # Échantillon du jeu de données de test
 ├── database/               # Création et gestion des bases (PostgreSQL / SQLite)
 ├── models/                 # Modèle sauvegardé (model.pkl)
 ├── src/                    # Modules de preprocessing, scaling, prédiction
 ├── tests/                  # Tests unitaires et fonctionnels (Pytest)
 ├── .gitignore
 ├── app.py                  # Application principale FastAPI + interface Gradio
 ├── Dockerfile
 ├── README_CI_CD.md
 ├── README.md
 └── pyproject.toml
```

---

## Organisation Gitflow

Le projet suit le workflow **Gitflow** avec **Pull Requests** :
- **feature/** → développement d’une nouvelle fonctionnalité  
- **develop** → intégration et validation des fonctionnalités terminées  
- **Pull Request** → ouverture d’une PR depuis `develop` vers `main`  
  - Exécution automatique de la CI/CD (tests Pytest + couverture)  
  - Merge uniquement si les tests passent  
- **release/vX.X.X** → stabilisation avant mise en production et création du tag de version  
- **main** → branche de production, déployée automatiquement

---

## Installation
1. Cloner le projet :
``` 
git clone https://github.com/SCFlorian/Modele_scoring_MLFlow
cd Modele_scoring_MLFlow
```
2. Installer les dépendances :
Le projet utilise pyproject.toml pour la gestion des dépendances.
```
poetry install
```
3. Ouvrir le projet dans VS Code
```
code .
```
4. Configurer l’environnement Python dans VS Code
	1.	Installez l’extension Python (si ce n’est pas déjà fait).
	2.	Appuyez sur Ctrl+Shift+P (Windows/Linux) ou Cmd+Shift+P (Mac).
	4.	Recherchez “Python: Select Interpreter”.
	5.	Sélectionnez l’environnement créé par Poetry ou celui dans lequel tu as installé le projet.