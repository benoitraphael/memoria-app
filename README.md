# MemorIA 🎤

MemorIA est une application web minimaliste conçue pour aider les utilisateurs à raconter leur vie, chapitre par chapitre, avec l'aide de l'intelligence artificielle pour la structuration et la rédaction.

## ✨ Fonctionnalités (MVP)

*   **Choix de modèles de livre :** Sélectionnez une structure prédéfinie (ex: "Ma vie en 5 temps").
*   **Ajout de souvenirs :** Racontez vos souvenirs par écrit ou via dictée vocale pour chaque chapitre.
*   **Transcription audio :** Utilise l'API Whisper d'OpenAI pour transcrire les enregistrements vocaux.
*   **Génération de chapitres par IA :** Combinez vos souvenirs et laissez GPT-4o rédiger un chapitre au style littéraire.
*   **Sauvegarde locale :** Les souvenirs et les chapitres générés sont sauvegardés en fichiers Markdown (`.md`).
*   **Export simple :** Téléchargez le livre complet ou des chapitres individuels.
*   **Clé API personnelle :** Utilisez votre propre clé API OpenAI pour les appels IA.

## 🛠️ Stack Technique

*   **Langage :** Python
*   **Framework Web :** Streamlit
*   **IA & Audio :** API OpenAI (GPT-4o, Whisper)
*   **Dépendances :** Voir `requirements.txt`

## 🚀 Installation et Lancement

1.  **Cloner le dépôt (si vous partez de zéro) :**
    ```bash
    git clone https://github.com/benoitraphael/memoria-app.git
    cd memoria-app
    ```
2.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
3.  **(Optionnel mais recommandé) Créer un environnement virtuel :**
    ```bash
    python -m venv venv
    source venv/bin/activate # Sur macOS/Linux
    # ou venv\\Scripts\\activate sur Windows
    pip install -r requirements.txt
    ```
4.  **Lancer l'application :**
    ```bash
    streamlit run app.py
    ```
5.  Ouvrez votre navigateur à l'adresse indiquée (généralement `http://localhost:8501`).
6.  Entrez votre clé API OpenAI dans l'application lorsque demandé.

## 📁 Structure des Dossiers

*   `/templates`: Contient les modèles de livre (JSON).
*   `/souvenirs`: Stocke les souvenirs bruts enregistrés par l'utilisateur (fichiers `.md`, organisés par chapitre). **(Non versionné par Git)**
*   `/chapitres`: Stocke les chapitres générés par l'IA (fichiers `.md`). **(Non versionné par Git)**
*   `/livres`: Stocke les exportations du livre complet. **(Non versionné par Git)**
*   `app.py`: Fichier principal de l'application Streamlit.
*   `requirements.txt`: Liste des dépendances Python.
*   `.gitignore`: Spécifie les fichiers et dossiers ignorés par Git.
*   `README.md`: Ce fichier.
