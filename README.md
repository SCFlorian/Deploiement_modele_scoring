---
title: Deploiement Modele Scoring
emoji: 💻
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
---
# 📊 Déploiement d'un modèle de scoring
---
## Description
Précédemment nous avons réalisé un modèle de scoring en partant du projet Home Credit Default Risk de Kaggle. Nous allons reprendre le meilleur modèle de ce projet afin de le déployer.

## Objectifs 

Les objectifs sont les suivants :
- un historique des versions
- une API fonctionnelle -> FastAPI avec une interface réalisée avec Gradio
- des tests unintaires automatisés
- un dockerfile
- Une analyse du Data Drift -> Réalisée avec EvidentlyAI
- Un dashboard avec Streamlit
- une solution de stockage des données en production
- un pipeline CI/CD
- une documentation README

## Données utilisées
- Les données de base viennent du projet Kaggle :

https://www.kaggle.com/c/home-credit-default-risk/data

## Organisation du projet
### En local
```
├── .github/workflows
│   ├──cicd.yaml
├── data
│   ├── example_input.csv # Nouveaux clients à tester
│   └── train_df.csv      # Données d'entraînement / Fichier présent uniquement en local car volumineux
│
├── database      
│   ├── create_db.py     # Création des tables de la BDD
│
├── models/        
│   ├── expected_columns.json
│   ├── imputer_columns.json
│   ├── threshold.txt
│
├── monitoring     # Dossier présent uniquement en local car trop volumineux
│   ├── dashboard_streamlit.py        # Tableaux de bord des indicateurs opérationnels
│   ├── monitoring_evidentlyai.ipynb  # Anlayse du datadrift
│   ├── cprofile.ipynb                # Optimisation du code sur la fonction prepare_client
│   ├── cprofile_old.ipynb            # Fichier app avant optimisation du code
│
├── tests/        
│   ├── test_basic.py
│   ├── test_endpoints.py
│   ├── test_model_full_dataset.py
│ 
├── .env     # Présent uniquement en local car gère les "secrets"
├── .gitignore
├── app.py        # Fichier app avant transformation
├── app_old.py    # Fichier app après transformation
├── Dockerfile    # Conteneurisation
├── README.md     # Documentation du projet
├── poetry.lock
└── pyproject.toml    # Dépendances et configuration
```
### Dans un dépôt Hugging Face
https://huggingface.co/FlorianSC/homecredit-scoring-artifacts
```
homecredit-scoring-artifacts
├── .gitattributes
├── imputer_numeric.pkl
├── pipeline_lightgbm.pkl
```
---

## Installation et utilisation

### Installation
1. Cloner le projet :
``` 
git clone git@github.com:SCFlorian/Deploiement_modele_scoring.git
cd Deploiement_modele_scoring
```
2. Installer les dépendances :
Le projet utilise pyproject.toml pour la gestion des dépendances.
```
poetry install
```
3. Ouvrir le projet dans VS Code :
```
code .
```
4. Configurer l’environnement Python dans VS Code :
	1.	Installez l’extension Python (si ce n’est pas déjà fait).
	2.	Appuyez sur Ctrl+Shift+P (Windows/Linux) ou Cmd+Shift+P (Mac).
	4.	Recherchez “Python: Select Interpreter”.
	5.	Sélectionnez l’environnement créé par Poetry ou celui dans lequel tu as installé le projet.

5. Création d'une base de données PostreSQL en local :
    1. Bien penser à mettre dans ".env" l'URL de la base de données.
    2. La création des bases se réalise au lancement de l'API.
    3. Création de 4 tables.


### Utilisation de l'API en local

- Générer l'environnement virtuel
```
poetry run uvicorn app:app --reload
```
- Ajouter l'URL sur un navigateur
```
http://127.0.0.1:8000
```

- L'ensemble des inputs et outputs seront enregistrés dans une base PostreSQL.

### Utilisation de l'API via Hugging Face

- Bien penser à paramétrer un token pour faire lien entre le projet GitHub et HF.
- Au lancement d'un 'push' sur main, il y a un déploiement sur Hugging Face Spaces :
    - https://huggingface.co/spaces/FlorianSC/Deploiement_modele_scoring
- Pas de sauvegarde des informations avec HF, uniquement une interface opérationnelle.

#### Création d'un dashboard pour le monitoring opérationnel avec Streamlit
- Générer l'environnement virtuel :
```
poetry run streamlit run monitoring/dashboard_streamlit.py
```
- Dashboard Streamlit

<img width="766" height="726" alt="Capture d’écran 2025-12-23 à 11 16 51" src="https://github.com/user-attachments/assets/d684e0d7-566b-4c7f-8869-b7a19cda1dd2" />

On peut remarquer des temps plutôt cohérent avec notre type de modèle :
    - Un temps d'inférence rapide.
    - La latence global et le temps CPU -> cohérence avec l'objectif du projet.

#### Génération de l'analyse du datadrift
- Rapport EvidendlyAI

<img width="1274" height="645" alt="Capture d’écran 2025-12-23 à 15 37 08" src="https://github.com/user-attachments/assets/7d987f90-0a71-4432-b443-dab032b18879" />

Pas de présence d'un gros changement entre les données qui ont contribué à l'entraînement et celles utilisées pour les nouveaux clients.

