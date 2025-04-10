import streamlit as st
import os
import json
import time
import uuid
import datetime
import re
import openai
from pathlib import Path
import numpy as np
import tempfile
import io
import sys
import gc  # Garbage Collector pour libérer de la mémoire

# --- Logs de Démarrage Très Spécifiques ---
print("--- Script app.py démarré ---")
print(f"Répertoire de travail : {os.getcwd()}")
print(f"Version Python : {sys.version}")
# Essayer d'importer et afficher la version de streamlit
try:
    print(f"Version Streamlit : {st.__version__}")
except AttributeError:
    print("Impossible de récupérer la version de Streamlit (__version__ non trouvé)")

try:
    # --- Configuration de la page --- 
    # DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT
    print("--- AVANT st.set_page_config ---")
    st.set_page_config(
        page_title="MemorIA - Votre atelier d'écriture personnel",
        page_icon="📝",
        layout="wide"
    )
    print("--- APRÈS st.set_page_config ---")

    # --- Initialisation des variables de session ---
    print("--- AVANT initialisation st.session_state ---")
    if "selected_template_name" not in st.session_state:
        st.session_state.selected_template_name = None
        print("Initialisé: selected_template_name")
    if "chapters" not in st.session_state:
        st.session_state.chapters = []
        print("Initialisé: chapters")
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
        print("Initialisé: api_key")
    if "generated_chapters" not in st.session_state:
        st.session_state.generated_chapters = {}
        print("Initialisé: generated_chapters")
    if "user_message" not in st.session_state:
        st.session_state.user_message = None
        print("Initialisé: user_message")
    if 'active_expander' not in st.session_state:
        st.session_state.active_expander = None
        print("Initialisé: active_expander")
    if 'chapter_messages' not in st.session_state:
        st.session_state.chapter_messages = {}
        print("Initialisé: chapter_messages")
    print("--- APRÈS initialisation st.session_state ---")

    # Appel gc.collect() ici, après les premières opérations Streamlit
    print("--- Appel gc.collect() --- ")
    gc.collect()

    # --- Affichage du message utilisateur global (si existant) ---
    print("--- AVANT affichage message utilisateur ---")
    if st.session_state.user_message:
        msg_type = st.session_state.user_message["type"]
        msg_text = st.session_state.user_message["text"]
        if msg_type == "success":
            st.success(msg_text)
        elif msg_type == "error":
            st.error(msg_text)
        elif msg_type == "warning":
            st.warning(msg_text)
        elif msg_type == "info":
            st.info(msg_text)
        # Effacer le message pour qu'il ne s'affiche qu'une fois
        st.session_state.user_message = None
    print("--- APRÈS affichage message utilisateur ---")
    print("--- Fin du bloc d'initialisation principal ---")

except Exception as e:
    print(f"!!! ERREUR CRITIQUE PENDANT L'INITIALISATION : {str(e)} !!!")
    # Essayer d'afficher l'erreur dans Streamlit si possible
    try:
        st.error(f"Une erreur critique s'est produite lors du démarrage : {str(e)}")
    except Exception as inner_e:
        print(f"!!! Impossible d'afficher l'erreur dans Streamlit : {inner_e} !!!")

# --- La suite du code (fonctions, interface principale...) ---
print("--- Début de la définition des fonctions --- ")

# --- Fonctions Utilitaires ---
def normalize_name(name):
    """Normaliser les noms de fichiers/dossiers pour éviter les caractères invalides"""
    return re.sub(r'[^a-z0-9]', '_', name.lower())

def update_text_content(widget_session_key, content_session_key):
    """Met à jour la variable de session contenant le texte lorsque l'utilisateur tape."""
    st.session_state[content_session_key] = st.session_state[widget_session_key]

def load_templates():
    """Charger les templates de livre disponibles"""
    template_dir = os.path.join(os.getcwd(), "templates")
    templates = {}
    
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
        # Créer un template par défaut si aucun n'existe
        default_template = {
            "titre": "Mon Autobiographie",
            "chapitres": ["Enfance", "Adolescence", "Âge adulte", "Réflexions"]
        }
        with open(os.path.join(template_dir, "default.json"), 'w', encoding='utf-8') as f:
            json.dump(default_template, f, ensure_ascii=False, indent=2)
    
    for filename in os.listdir(template_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(template_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    template = json.load(f)
                    # Accepter soit "titre" (français) soit "title" (anglais)
                    template_name = template.get('titre', template.get('title', os.path.splitext(filename)[0]))
                    templates[template_name] = template
            except Exception as e:
                st.error(f"Erreur lors du chargement du template {filename}: {str(e)}")
    
    return templates

def load_memories(chapter):
    """Charger les souvenirs enregistrés pour un chapitre spécifique"""
    memories = []
    
    # Créer le chemin du dossier pour ce chapitre
    template_name = st.session_state.get('selected_template_name')
    if not template_name:
        return memories
        
    chapter_dir = os.path.join(os.getcwd(), "souvenirs", normalize_name(template_name), normalize_name(chapter))
    
    if not os.path.exists(chapter_dir):
        return memories
    
    # Charger tous les fichiers .md dans le dossier
    for filename in os.listdir(chapter_dir):
        if filename.endswith('.md'):
            file_path = os.path.join(chapter_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                memories.append({
                    'filename': filename,
                    'content': content,
                    'path': file_path
                })
            except Exception as e:
                st.error(f"Erreur lors du chargement du souvenir {filename}: {str(e)}")
    
    # Libérer la mémoire après le chargement (ajout pour optimisation)
    gc.collect()
    
    return memories

def save_memory(template_name, chapter, content):
    """Sauvegarder un souvenir pour un chapitre spécifique avec un nom de fichier unique."""
    if not content or not content.strip():
        return False, "Le contenu du souvenir est vide.", None
        
    try:
        # Créer les dossiers nécessaires s'ils n'existent pas
        base_dir = Path("souvenirs")
        template_dir = base_dir / normalize_name(template_name)
        chapter_dir = template_dir / normalize_name(chapter)
        chapter_dir.mkdir(parents=True, exist_ok=True)
        
        # Générer un nom de fichier unique
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8] # Court UUID pour unicité
        filename = f"{timestamp}_{unique_id}.md"
        file_path = chapter_dir / filename
        
        # Écrire le contenu dans le fichier
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return True, f"Souvenir sauvegardé avec succès sous `{filename}`", str(file_path)
        
    except Exception as e:
        error_msg = f"Erreur lors de la sauvegarde du souvenir : {str(e)}"
        st.error(error_msg)
        return False, error_msg, None

def load_chapter(template_name, chapter):
    """Charger un chapitre généré s'il existe"""
    chapter_file = os.path.join(os.getcwd(), "chapitres", normalize_name(template_name), f"{normalize_name(chapter)}.md")
    
    if not os.path.exists(chapter_file):
        return None
        
    try:
        with open(chapter_file, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        st.error(f"Erreur lors du chargement du chapitre: {str(e)}")
        return None

def save_chapter(template_name, chapter_name, content):
    """Sauvegarder un chapitre généré"""
    # Créer les dossiers nécessaires
    base_dir = os.path.join(os.getcwd(), "chapitres")
    template_dir = os.path.join(base_dir, normalize_name(template_name))
    
    os.makedirs(template_dir, exist_ok=True)
    
    file_path = os.path.join(template_dir, f"{normalize_name(chapter_name)}.md")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, f"Chapitre '{chapter_name}' sauvegardé avec succès", file_path
    except Exception as e:
        error_msg = f"Erreur lors de l'enregistrement du chapitre: {str(e)}"
        return False, error_msg, None

def load_generated_chapters():
    """Charger tous les chapitres générés pour le template actuel"""
    if 'selected_template_name' not in st.session_state:
        return {}
        
    template_name = st.session_state.selected_template_name
    chapters_dir = os.path.join(os.getcwd(), "chapitres", normalize_name(template_name))
    
    if not os.path.exists(chapters_dir):
        return {}
        
    chapters = {}
    for filename in os.listdir(chapters_dir):
        if filename.endswith('.md'):
            chapter_name = os.path.splitext(filename)[0]
            file_path = os.path.join(chapters_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                chapters[chapter_name] = content
            except Exception as e:
                st.error(f"Erreur lors du chargement du chapitre {filename}: {str(e)}")
    
    return chapters

def compile_book(template_name, ordered_chapters):
    """Compiler tous les chapitres en un seul document"""
    if not ordered_chapters:
        return None
        
    book_content = []
    
    # Ajouter le titre
    book_content.append(f"# {template_name}\n\n")
    
    # Ajouter chaque chapitre
    for chapter_name in ordered_chapters:
        norm_chapter = normalize_name(chapter_name)
        chapter_file = os.path.join(os.getcwd(), "chapitres", normalize_name(template_name), f"{norm_chapter}.md")
        
        if os.path.exists(chapter_file):
            try:
                with open(chapter_file, 'r', encoding='utf-8') as f:
                    chapter_content = f.read()
                
                # Ajouter le titre du chapitre si nécessaire
                if not chapter_content.strip().startswith('#'):
                    book_content.append(f"## {chapter_name}\n\n")
                
                book_content.append(f"{chapter_content}\n\n")
            except Exception as e:
                st.error(f"Erreur lors de la compilation du chapitre {chapter_name}: {str(e)}")
    
    # Créer le livre complet
    full_book = "\n".join(book_content)
    
    # Sauvegarder dans le dossier des livres
    books_dir = os.path.join(os.getcwd(), "livres")
    os.makedirs(books_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    book_filename = f"{normalize_name(template_name)}_{timestamp}.md"
    book_path = os.path.join(books_dir, book_filename)
    
    try:
        with open(book_path, 'w', encoding='utf-8') as f:
            f.write(full_book)
        return full_book, book_filename
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde du livre: {str(e)}")
        return full_book, None

def generate_chapter_with_ai(api_key, template_name, chapter_name, memories):
    """Générer un chapitre à partir des souvenirs avec l'API OpenAI"""
    if not api_key:
        return "Veuillez d'abord entrer votre clé API OpenAI", False
    
    if not memories:
        return "Aucun souvenir trouvé pour ce chapitre", False
    
    try:
        # Concaténer tous les souvenirs
        all_memories = "\n\n---\n\n".join([memory['content'] for memory in memories])
        
        # Définir le prompt
        prompt = f"""
En tant qu'écrivain biographe professionnel, tu vas rédiger un chapitre intitulé '{chapter_name}' pour le livre '{template_name}' basé sur les souvenirs suivants :

{all_memories}

Instructions :
1. Écris un texte captivant et cohérent qui intègre les souvenirs de façon fluide
2. Organise le contenu de manière chronologique ou thématique selon ce qui convient le mieux
3. Utilise un style littéraire élégant avec des descriptions sensorielles
4. Crée des transitions naturelles entre les différents souvenirs
5. Commence par un titre de niveau 2 avec le nom du chapitre
6. Divise le texte en sections avec des sous-titres si nécessaire
7. Longueur recommandée : 800-1500 mots

Le texte doit avoir un ton personnel, comme si l'auteur des souvenirs le racontait lui-même.
Utilise exclusivement les informations fournies dans les souvenirs, sans inventer de faits.
        """
        
        # Configurer l'API OpenAI
        openai.api_key = api_key
        
        # Appeler l'API
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Tu es un écrivain biographe professionnel qui transforme des souvenirs en chapitres de livre."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=16000
        )
        
        # Extraire le texte généré
        generated_content = response.choices[0].message.content
        
        # Nettoyage de la mémoire explicite (ajout pour optimisation)
        del response
        gc.collect()
        
        # Sauvegarder le chapitre
        success, _, _ = save_chapter(template_name, chapter_name, generated_content)
        
        if success:
            return generated_content, True
        else:
            return "Erreur lors de la sauvegarde du chapitre généré", False
            
    except Exception as e:
        error_msg = f"Erreur lors de la génération du chapitre: {str(e)}"
        print(error_msg)  # Garder le log d'erreur
        return error_msg, False

# Fonction simplifiée pour l'enregistrement audio
def simple_audio_recorder(chapter_name, text_key):
    """Création d'un enregistreur audio simple qui ressemble à celui de ChatGPT"""
    
    # Générer des clés uniques
    recording_key = f"recording_{normalize_name(chapter_name)}"
    api_key = st.session_state.get('api_key', '')
    
    # Vérifier si on a une clé API
    if not api_key:
        st.warning("⚠️ Veuillez d'abord entrer votre clé API OpenAI dans les paramètres.")
        return
    
    # Initialiser l'état d'enregistrement s'il n'existe pas
    if recording_key not in st.session_state:
        st.session_state[recording_key] = {
            "is_recording": False,
            "audio_processor": None,
            "webrtc_ctx_key": f"webrtc_{uuid.uuid4()}"
        }
    
    # Récupérer l'état
    state = st.session_state[recording_key]
    
    # Colonne pour le bouton micro et les contrôles
    col1, col2 = st.columns([1, 10])
    
    with col1:
        # Si on n'est pas en train d'enregistrer, montrer le bouton micro
        if not state["is_recording"] and not st.session_state.get(f"transcribing_{recording_key}", False):
            if st.button("🎤", key=f"mic_{recording_key}", help="Commencer l'enregistrement"):
                state["is_recording"] = True
                # Forcer la mise à jour de l'interface
                st.rerun()
        
        # Si on est en train d'enregistrer, montrer le bouton pour arrêter
        elif state["is_recording"]:
            if st.button("✓", key=f"stop_{recording_key}", help="Arrêter et transcrire"):
                state["is_recording"] = False
                st.session_state[f"transcribing_{recording_key}"] = True
                # Forcer la mise à jour de l'interface
                st.rerun()
        
        # Si on est en train de transcrire, montrer une animation de chargement
        elif st.session_state.get(f"transcribing_{recording_key}", False):
            st.spinner("...")
            # Ici on aurait normalement un spinner, mais on va utiliser un texte
            st.text("...")
            
            # Effectuer la transcription
            with st.spinner("Transcription en cours..."):
                try:
                    # Initialiser le client OpenAI avec la clé API
                    client = openai.OpenAI(api_key=api_key)
                    
                    # Créer un objet fichier-en-mémoire à partir des bytes audio
                    audio_bio = io.BytesIO(st.session_state[f"audio_bytes_{recording_key}"])
                    # IMPORTANT: Donner un nom à ce fichier virtuel, requis par OpenAI
                    audio_bio.name = "audio.wav" 
                    
                    # Appel à l'API Whisper en passant l'objet BytesIO directement
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_bio, # Passer l'objet BytesIO
                        language="fr",
                        response_format="text"
                    )
                        
                    # Stocker la transcription dans l'état de session
                    content_key = f"new_memory_content_{normalize_name(chapter_name)}"
                    st.session_state[content_key] = transcription

                    # Afficher l'interface texte et masquer l'interface audio
                    show_text_input_key = f"show_text_input_{normalize_name(chapter_name)}"
                    st.session_state[show_text_input_key] = True
                    st.session_state[f"show_audio_recorder_{recording_key}"] = False
                    st.session_state.active_expander = f"expander_{normalize_name(chapter_name)}"
                            
                    # Déclencher le rafraîchissement pour afficher le texte dans le bon champ
                    st.rerun()
                                
                except Exception as e:
                    st.error(f"Erreur lors de la transcription : {str(e)}")
                    # Optionnel: on pourrait aussi stocker l'erreur via user_message
                    # st.session_state.user_message = {"type": "error", "text": f"Erreur transcription: {e}"}
                    # st.rerun()

                # --- SUPPRIMER TOUTE LA SECTION SUIVANTE --- 
                # (Affichage de la zone de texte spécifique à l'audio et son bouton)
                # transcribed_text = st.text_area(
                #     "Texte transcrit (éditable) :", 
                #     value=st.session_state.get(transcribed_text_key, ""),
                #     height=150,
                #     key=f"transcribed_display_{normalize_name(chapter_name)}"
                # )

                # if st.button("", key=f"save_audio_{normalize_name(chapter_name)}"):
                #     if transcribed_text.strip():
                #         try:
                #             # Lire le texte potentiellement édité
                #             edited_text = st.session_state[f"transcribed_display_{normalize_name(chapter_name)}"]
                #             save_memory(st.session_state.selected_template_name, chapter_name, edited_text)
                #             st.success(f"Souvenir (issu de l'audio) sauvegardé pour le chapitre '{chapter_name}' !")
                #             # Réinitialiser et cacher
                #             st.session_state[f"show_audio_recorder_{recording_key}"] = False
                #             st.session_state[transcribed_text_key] = ""
                #             st.rerun()
                #         except Exception as e:
                #              st.error(f"Erreur sauvegarde audio: {e}")
                #     else:
                #         st.warning("Le champ transcrit est vide.")

            # Affichage des souvenirs existants pour le chapitre
            # memories = load_memories(chapter_name)
            # if memories:
            #     for i, memory in enumerate(memories):
            #         # Ne PAS utiliser d'expander ici pour éviter l'imbrication
            #         st.markdown(f"- **{memory['filename']}**") # Afficher le nom du fichier
            #         # Afficher le contenu directement en dessous
            #         st.text_area(
            #             f"Contenu_{memory['filename']}", 
            #             value=memory['content'], 
            #             height=100, 
            #             disabled=True, # Rendre non éditable ici
            #             label_visibility="collapsed" # Cacher le label par défaut
            #         ) 
            #         # Ajouter un petit séparateur
            #         st.divider()
            # else:
            #     st.caption("Aucun souvenir enregistré pour ce chapitre.")
            
            st.markdown("--- ")
            
            # --- Ajout d'un nouveau souvenir --- 
            st.markdown("**Ajouter un nouveau souvenir :**")
            add_text_key = f"add_text_{normalize_name(chapter_name)}"
            add_audio_key = f"add_audio_{normalize_name(chapter_name)}"
            show_text_input_key = f"show_text_input_{normalize_name(chapter_name)}"
            new_memory_text_key = f"new_memory_text_{normalize_name(chapter_name)}"
            show_audio_recorder_key = f"show_audio_recorder_{recording_key}"
            transcribed_text_key = f"transcribed_text_{normalize_name(chapter_name)}"

            # Initialiser les états si nécessaire
            if show_text_input_key not in st.session_state:
                st.session_state[show_text_input_key] = False
            if show_audio_recorder_key not in st.session_state:
                st.session_state[show_audio_recorder_key] = False
            if transcribed_text_key not in st.session_state:
                st.session_state[transcribed_text_key] = ""

            # Boutons pour choisir le mode d'ajout
            col1, col2, _ = st.columns([1,1,3])
            with col1:
                if st.button("💬 Ajouter via Texte", key=add_text_key):
                    st.session_state[show_text_input_key] = True
                    st.session_state[show_audio_recorder_key] = False # Assurer l'exclusivité
                    st.rerun()
            with col2:
                 if st.button("🎙️ Ajouter via Audio", key=add_audio_key):
                    st.session_state[show_audio_recorder_key] = True
                    st.session_state[show_text_input_key] = False # Assurer l'exclusivité
                    st.rerun()

            # --- Interface d'ajout Texte ---
            if st.session_state.get(show_text_input_key, False):
                st.markdown("#### Nouveau Souvenir Texte")
                widget_key = f"new_memory_text_widget_{normalize_name(chapter_name)}"
                content_key = f"new_memory_content_{normalize_name(chapter_name)}"

                if content_key not in st.session_state:
                    st.session_state[content_key] = ""

                st.text_area(
                    "Racontez votre souvenir ici :",
                    height=150,
                    value=st.session_state[content_key], 
                    key=widget_key, 
                    on_change=update_text_content, 
                    args=(widget_key, content_key) 
                )

                save_col, cancel_col = st.columns(2)
                with save_col:
                    if st.button("💾 Enregistrer ce Souvenir", key=f"save_new_text_{normalize_name(chapter_name)}"):
                        current_content = st.session_state.get(content_key, "").strip()
                        if current_content:
                            try:
                                save_memory(st.session_state.selected_template_name, chapter_name, current_content)
                                # Stocker le message de succès dans le dictionnaire des messages par chapitre
                                chapter_key = normalize_name(chapter_name)
                                st.session_state.chapter_messages[chapter_key] = {"type": "success", "text": f"Souvenir texte enregistré pour le chapitre '{chapter_name}' !"}
                                st.session_state[content_key] = ""
                                st.session_state.active_expander = f"expander_{normalize_name(chapter_name)}"
                                st.rerun()
                            except Exception as e:
                                # Stocker le message d'erreur dans le dictionnaire des messages par chapitre
                                chapter_key = normalize_name(chapter_name)
                                st.session_state.chapter_messages[chapter_key] = {"type": "error", "text": f"Erreur lors de l'enregistrement du souvenir texte : {e}"}
                                st.rerun()
                        else:
                            # Stocker le message d'avertissement dans le dictionnaire des messages par chapitre
                            chapter_key = normalize_name(chapter_name)
                            st.session_state.chapter_messages[chapter_key] = {"type": "warning", "text": "Le champ souvenir est vide."}
                            st.rerun()

                with cancel_col:
                    if st.button("Annuler", key=f"cancel_text_{normalize_name(chapter_name)}"):
                        st.session_state[show_text_input_key] = False
                        st.session_state[content_key] = ""
                        st.rerun()

            # --- Interface d'ajout Audio --- 
            if st.session_state.get(show_audio_recorder_key, False):
                st.markdown("#### Nouveau Souvenir Audio")
                st.info("Cliquez sur l'icône micro pour enregistrer, puis validez l'enregistrement.")

                # Utiliser le composant audio_input comme dans le tutoriel
                audio_data = st.audio_input(
                    label="Enregistrez votre souvenir ici :", 
                    key=f"audio_input_{normalize_name(chapter_name)}"
                )
                
                # Si l'audio a été enregistré OU uploadé (audio_data contient des données)
                if audio_data:
                    audio_bytes = audio_data.read() # Lire les bytes depuis l'objet retourné
                    
                    # Afficher le lecteur audio pour réécouter (facultatif mais utile)
                    st.audio(audio_bytes, format="audio/wav") # Assumer WAV pour l'instant
                    
                    # Préparation pour la transcription (méthode directe)
                    with st.spinner("Transcription en cours..."):
                        try:
                            # Initialiser le client OpenAI avec la clé API
                            client = openai.OpenAI(api_key=api_key)
                            
                            # Créer un objet fichier-en-mémoire à partir des bytes audio
                            audio_bio = io.BytesIO(audio_bytes)
                            # IMPORTANT: Donner un nom à ce fichier virtuel, requis par OpenAI
                            audio_bio.name = "audio.wav" 
                            
                            # Appel à l'API Whisper en passant l'objet BytesIO directement
                            transcription = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_bio, # Passer l'objet BytesIO
                                language="fr",
                                response_format="text"
                            )
                                
                            # Stocker la transcription dans l'état de session
                            content_key = f"new_memory_content_{normalize_name(chapter_name)}"
                            st.session_state[content_key] = transcription

                            # Afficher l'interface texte et masquer l'interface audio
                            show_text_input_key = f"show_text_input_{normalize_name(chapter_name)}"
                            st.session_state[show_text_input_key] = True
                            st.session_state[show_audio_recorder_key] = False
                            st.session_state.active_expander = f"expander_{normalize_name(chapter_name)}"
                            
                            # Déclencher le rafraîchissement pour afficher le texte dans le bon champ
                            st.rerun()
                                
                        except Exception as e:
                            st.error(f"Erreur lors de la transcription : {str(e)}")
                            # Optionnel: on pourrait aussi stocker l'erreur via user_message
                            # st.session_state.user_message = {"type": "error", "text": f"Erreur transcription: {e}"}
                            # st.rerun()

                # --- SUPPRIMER TOUTE LA SECTION SUIVANTE --- 
                # (Affichage de la zone de texte spécifique à l'audio et son bouton)
                # transcribed_text = st.text_area(
                #     "Texte transcrit (éditable) :", 
                #     value=st.session_state.get(transcribed_text_key, ""),
                #     height=150,
                #     key=f"transcribed_display_{normalize_name(chapter_name)}"
                # )

                # if st.button("", key=f"save_audio_{normalize_name(chapter_name)}"):
                #     if transcribed_text.strip():
                #         try:
                #             # Lire le texte potentiellement édité
                #             edited_text = st.session_state[f"transcribed_display_{normalize_name(chapter_name)}"]
                #             save_memory(st.session_state.selected_template_name, chapter_name, edited_text)
                #             st.success(f"Souvenir (issu de l'audio) sauvegardé pour le chapitre '{chapter_name}' !")
                #             # Réinitialiser et cacher
                #             st.session_state[show_audio_recorder_key] = False
                #             st.session_state[transcribed_text_key] = ""
                #             st.rerun()
                #         except Exception as e:
                #              st.error(f"Erreur sauvegarde audio: {e}")
                #     else:
                #         st.warning("Le champ transcrit est vide.")

            # Suppression de l'ancienne section 'souvenir rapide'
            # temp_memory_text = st.text_area("Écrire un souvenir rapide :", key=f"temp_text_{normalize_name(chapter_name)}")
            # if st.button(f" Sauvegarder souvenir rapide pour '{chapter_name}'", key=f"save_temp_{normalize_name(chapter_name)}"):
            #     if temp_memory_text.strip():
            #         # Utilisation de la fonction save_memory mise à jour
            #         success, message, file_path = save_memory(st.session_state.selected_template_name, chapter_name, temp_memory_text)
            #         if success:
            #             st.success(message)
            #             # Effacer le champ après sauvegarde et rafraîchir
            #             st.session_state[f"temp_text_{normalize_name(chapter_name)}"] = ""
            #             st.rerun()
            #         else:
            #             st.error(f"Erreur sauvegarde : {message}")
            #     else:
            #         st.warning("Le champ souvenir est vide.")
            
            st.markdown("--- ")
            
            # --- Génération du chapitre --- 
            st.markdown("**Générer le chapitre :**")
            chapter_filename = normalize_name(chapter_name) + ".md"
            chapter_path = os.path.join(os.getcwd(), "chapitres", normalize_name(st.session_state.selected_template_name), chapter_filename)
            
            # Charger le chapitre généré s'il existe déjà
            generated_content = None
            if os.path.exists(chapter_path):
                try:
                    with open(chapter_path, 'r', encoding='utf-8') as f:
                        generated_content = f.read()
                except Exception as e:
                    st.warning(f"Impossible de lire le chapitre généré: {e}")

            # Bouton pour générer
            if st.button(f"✨ Générer le chapitre '{chapter_name}'", key=f"generate_{normalize_name(chapter_name)}", disabled=not memories):
                if memories:
                    with st.spinner("🧠 L'IA réfléchit et rédige..."): 
                        # Appel réel à la fonction de génération
                        generated_text, success = generate_chapter_with_ai(
                            st.session_state.api_key, 
                            st.session_state.selected_template_name, 
                            chapter_name, 
                            memories
                        )
                        
                        # Vérifier si la génération a réussi
                        if success and generated_text:
                            # Sauvegarder le chapitre généré
                            save_success, save_message, save_path = save_chapter(
                                st.session_state.selected_template_name, 
                                chapter_name, 
                                generated_text
                            )
                            
                            if save_success:
                                # Mettre à jour l'état de session pour affichage immédiat
                                st.session_state.generated_chapters[normalize_name(chapter_name)] = generated_text
                                # Stocker le message de succès dans le dictionnaire des messages par chapitre
                                chapter_key = normalize_name(chapter_name)
                                st.session_state.chapter_messages[chapter_key] = {"type": "success", "text": f"Chapitre '{chapter_name}' généré et sauvegardé ! {save_message}"}
                                st.session_state.active_expander = f"expander_{normalize_name(chapter_name)}"
                                st.rerun() # Recharger pour afficher
                            else:
                                # Stocker le message d'erreur dans le dictionnaire des messages par chapitre
                                chapter_key = normalize_name(chapter_name)
                                st.session_state.chapter_messages[chapter_key] = {"type": "error", "text": f"Erreur lors de la sauvegarde du chapitre généré: {save_message}"}
                                st.session_state.active_expander = f"expander_{normalize_name(chapter_name)}"
                                st.rerun() # Recharger pour afficher l'erreur
                        else:
                            # Afficher l'erreur retournée par la fonction de génération
                            # Stocker le message d'erreur dans le dictionnaire des messages par chapitre
                            chapter_key = normalize_name(chapter_name)
                            st.session_state.chapter_messages[chapter_key] = {"type": "error", "text": f"La génération du chapitre a échoué : {generated_text}"}
                            st.session_state.active_expander = f"expander_{normalize_name(chapter_name)}"
                            st.rerun() # Recharger pour afficher l'erreur
                else:
                     st.warning("⚠️ Ajoutez au moins un souvenir avant de générer.")
                     st.session_state.active_expander = f"expander_{normalize_name(chapter_name)}"

            elif not memories:
                # Si aucun souvenir, on ne fait rien mais on reste sur le même expander
                st.caption("⚠️ Ajoutez des souvenirs pour pouvoir générer ce chapitre.") # Remettre le caption
                st.session_state.active_expander = f"expander_{normalize_name(chapter_name)}"

            # Afficher le chapitre généré (s'il existe dans l'état de session ou fichier)
            display_content = st.session_state.generated_chapters.get(normalize_name(chapter_name), generated_content)
            if display_content:
                st.markdown("**Texte généré :**")
                st.markdown(display_content)
            

# --- Export (à placer en dehors de la boucle des chapitres) ---
st.sidebar.markdown("---")
st.sidebar.header("Exporter votre livre")

if st.session_state.selected_template_name and st.session_state.generated_chapters:
    ordered_chapters = st.session_state.chapters.copy()  # Pour respecter l'ordre des chapitres
    
    if st.sidebar.button("📚 Compiler en un seul document"):
        with st.spinner("Compilation en cours..."):
            compiled_text, book_filename = compile_book(st.session_state.selected_template_name, ordered_chapters)
            st.session_state.user_message = {"type": "success", "text": "Livre compilé avec succès !"}
            st.download_button(
                label="📥 Télécharger le livre complet",
                data=compiled_text,
                file_name=book_filename,
                mime="text/markdown"
            )
            st.rerun()

# --- Interface Utilisateur Streamlit ---
st.title("MemorIA 📝")
st.markdown("Racontez vos souvenirs, l'IA rédige votre histoire.")

# --- Barre latérale ---
with st.sidebar:
    st.title("Configuration")
    st.markdown("Paramétrez votre expérience MemorIA.")
    
    # Saisie de la clé API OpenAI
    api_key = st.text_input(
        "Clé API OpenAI",
        type="password",
        value=st.session_state.get("api_key", ""),
        help="Votre clé API est nécessaire pour les fonctions IA. Elle reste locale."
    )
    
    # Mettre à jour la clé API dans l'état de session
    if api_key:
        st.session_state.api_key = api_key
        st.success("Clé API enregistrée.", icon="🔑")
    elif "api_key" in st.session_state and not api_key: # Si l'utilisateur efface la clé
        del st.session_state.api_key
        st.info("Clé API retirée.")
        st.rerun() # Recharger si la clé est enlevée pour désactiver les fonctions IA

    # Choix du modèle de livre (uniquement si la clé API est fournie)
    if st.session_state.get("api_key"):
        templates = load_templates()
        if templates:
            template_options = list(templates.keys())
            
            selected_template_index = 0
            if st.session_state.get("selected_template_name") in template_options:
                selected_template_index = template_options.index(st.session_state.selected_template_name)
                
            selected_template = st.selectbox(
                "Modèle de livre :",
                template_options,
                index=selected_template_index,
                key="template_selector_sidebar"
            )
            
            # Si le modèle sélectionné change
            if selected_template and selected_template != st.session_state.get("selected_template_name"):
                st.session_state.selected_template_name = selected_template
                st.session_state.chapters = templates[selected_template].get('chapitres', templates[selected_template].get('chapters', []))
                st.session_state.generated_chapters = {} # Réinitialiser si on change de modèle
                st.rerun()
        else:
            st.error("Aucun modèle trouvé !")
    else:
        st.warning("⚠️ Veuillez entrer votre clé API OpenAI pour choisir un modèle.")

    st.markdown("---")
    st.title("À propos")
    st.markdown("MemorIA transforme vos souvenirs bruts en chapitres de vie rédigés par l'IA.")

# --- Vérification initiale --- 
if not st.session_state.get("api_key"):
    st.warning("⚠️ Veuillez entrer votre clé API OpenAI dans la barre latérale pour commencer.")
    st.stop() # Arrêter l'exécution si pas de clé API

if not st.session_state.get("selected_template_name"):
    st.info("Veuillez choisir un modèle de livre dans la barre latérale.")
    st.stop() # Arrêter l'exécution si pas de modèle choisi

# --- Affichage des chapitres (Zone Principale) ---
st.header(f"Livre : {st.session_state.selected_template_name}")

if st.session_state.chapters:
    st.subheader("Vos Chapitres")
    
    # Affichage des chapitres avec expanders
    for chapter in st.session_state.chapters:
        # Clé unique pour l'expander basée sur le chapitre normalisé
        expander_key = f"expander_{normalize_name(chapter)}"
        chapter_key = normalize_name(chapter)
        
        with st.expander(f"Chapitre : {chapter}", expanded=(expander_key == st.session_state.get('active_expander'))):
            st.markdown(f"### Gérer le chapitre : {chapter}")
            
            # Afficher les messages spécifiques à ce chapitre, s'il y en a
            if chapter_key in st.session_state.chapter_messages:
                msg = st.session_state.chapter_messages[chapter_key]
                if msg["type"] == "success":
                    st.success(msg["text"])
                elif msg["type"] == "error":
                    st.error(msg["text"])
                elif msg["type"] == "warning":
                    st.warning(msg["text"])
                elif msg["type"] == "info":
                    st.info(msg["text"])
                # Effacer le message après l'avoir affiché
                del st.session_state.chapter_messages[chapter_key]
            
            # --- Affichage des souvenirs existants ---
            st.markdown("**Souvenirs enregistrés :**")
            memories = load_memories(chapter)
            if memories:
                for i, memory in enumerate(memories):
                    # Ne PAS utiliser d'expander ici pour éviter l'imbrication
                    st.markdown(f"- **{memory['filename']}**") # Afficher le nom du fichier
                    # Afficher le contenu directement en dessous
                    st.text_area(
                        f"Contenu_{memory['filename']}", 
                        value=memory['content'], 
                        height=100, 
                        disabled=True, # Rendre non éditable ici
                        label_visibility="collapsed" # Cacher le label par défaut
                    ) 
                    # Ajouter un petit séparateur
                    st.divider()
            else:
                st.caption("Aucun souvenir enregistré pour ce chapitre.")
            
            st.markdown("--- ")
            
            # --- Ajout d'un nouveau souvenir --- 
            st.markdown("**Ajouter un nouveau souvenir :**")
            add_text_key = f"add_text_{normalize_name(chapter)}"
            add_audio_key = f"add_audio_{normalize_name(chapter)}"
            show_text_input_key = f"show_text_input_{normalize_name(chapter)}"
            new_memory_text_key = f"new_memory_text_{normalize_name(chapter)}"
            show_audio_recorder_key = f"show_audio_recorder_{expander_key}"
            transcribed_text_key = f"transcribed_text_{normalize_name(chapter)}"

            # Initialiser les états si nécessaire
            if show_text_input_key not in st.session_state:
                st.session_state[show_text_input_key] = False
            if show_audio_recorder_key not in st.session_state:
                st.session_state[show_audio_recorder_key] = False
            if transcribed_text_key not in st.session_state:
                st.session_state[transcribed_text_key] = ""

            # Boutons pour choisir le mode d'ajout
            col1, col2, _ = st.columns([1,1,3])
            with col1:
                if st.button("💬 Ajouter via Texte", key=add_text_key):
                    st.session_state[show_text_input_key] = True
                    st.session_state[show_audio_recorder_key] = False # Assurer l'exclusivité
                    st.rerun()
            with col2:
                 if st.button("🎙️ Ajouter via Audio", key=add_audio_key):
                    st.session_state[show_audio_recorder_key] = True
                    st.session_state[show_text_input_key] = False # Assurer l'exclusivité
                    st.rerun()

            # --- Interface d'ajout Texte ---
            if st.session_state.get(show_text_input_key, False):
                st.markdown("#### Nouveau Souvenir Texte")
                widget_key = f"new_memory_text_widget_{normalize_name(chapter)}"
                content_key = f"new_memory_content_{normalize_name(chapter)}"

                if content_key not in st.session_state:
                    st.session_state[content_key] = ""

                st.text_area(
                    "Racontez votre souvenir ici :",
                    height=150,
                    value=st.session_state[content_key], 
                    key=widget_key, 
                    on_change=update_text_content, 
                    args=(widget_key, content_key) 
                )

                save_col, cancel_col = st.columns(2)
                with save_col:
                    if st.button("💾 Enregistrer ce Souvenir", key=f"save_new_text_{normalize_name(chapter)}"):
                        current_content = st.session_state.get(content_key, "").strip()
                        if current_content:
                            try:
                                save_memory(st.session_state.selected_template_name, chapter, current_content)
                                # Stocker le message de succès dans le dictionnaire des messages par chapitre
                                chapter_key = normalize_name(chapter)
                                st.session_state.chapter_messages[chapter_key] = {"type": "success", "text": f"Souvenir texte enregistré pour le chapitre '{chapter}' !"}
                                st.session_state[content_key] = ""
                                st.session_state.active_expander = expander_key
                                st.rerun()
                            except Exception as e:
                                # Stocker le message d'erreur dans le dictionnaire des messages par chapitre
                                chapter_key = normalize_name(chapter)
                                st.session_state.chapter_messages[chapter_key] = {"type": "error", "text": f"Erreur lors de l'enregistrement du souvenir texte : {e}"}
                                st.rerun()
                        else:
                            # Stocker le message d'avertissement dans le dictionnaire des messages par chapitre
                            chapter_key = normalize_name(chapter)
                            st.session_state.chapter_messages[chapter_key] = {"type": "warning", "text": "Le champ souvenir est vide."}
                            st.rerun()

                with cancel_col:
                    if st.button("Annuler", key=f"cancel_text_{normalize_name(chapter)}"):
                        st.session_state[show_text_input_key] = False
                        st.session_state[content_key] = ""
                        st.rerun()

            # --- Interface d'ajout Audio --- 
            if st.session_state.get(show_audio_recorder_key, False):
                st.markdown("#### Nouveau Souvenir Audio")
                st.info("Cliquez sur l'icône micro pour enregistrer, puis validez l'enregistrement.")

                # Utiliser le composant audio_input comme dans le tutoriel
                audio_data = st.audio_input(
                    label="Enregistrez votre souvenir ici :", 
                    key=f"audio_input_{normalize_name(chapter)}"
                )
                
                # Si l'audio a été enregistré OU uploadé (audio_data contient des données)
                if audio_data:
                    audio_bytes = audio_data.read() # Lire les bytes depuis l'objet retourné
                    
                    # Afficher le lecteur audio pour réécouter (facultatif mais utile)
                    st.audio(audio_bytes, format="audio/wav") # Assumer WAV pour l'instant
                    
                    # Préparation pour la transcription (méthode directe)
                    with st.spinner("Transcription en cours..."):
                        try:
                            # Initialiser le client OpenAI avec la clé API
                            client = openai.OpenAI(api_key=api_key)
                            
                            # Créer un objet fichier-en-mémoire à partir des bytes audio
                            audio_bio = io.BytesIO(audio_bytes)
                            # IMPORTANT: Donner un nom à ce fichier virtuel, requis par OpenAI
                            audio_bio.name = "audio.wav" 
                            
                            # Appel à l'API Whisper en passant l'objet BytesIO directement
                            transcription = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_bio, # Passer l'objet BytesIO
                                language="fr",
                                response_format="text"
                            )
                                
                            # Stocker la transcription dans l'état de session
                            content_key = f"new_memory_content_{normalize_name(chapter)}"
                            st.session_state[content_key] = transcription

                            # Afficher l'interface texte et masquer l'interface audio
                            show_text_input_key = f"show_text_input_{normalize_name(chapter)}"
                            st.session_state[show_text_input_key] = True
                            st.session_state[show_audio_recorder_key] = False
                            st.session_state.active_expander = expander_key
                            
                            # Déclencher le rafraîchissement pour afficher le texte dans le bon champ
                            st.rerun()
                                
                        except Exception as e:
                            st.error(f"Erreur lors de la transcription : {str(e)}")
                            # Optionnel: on pourrait aussi stocker l'erreur via user_message
                            # st.session_state.user_message = {"type": "error", "text": f"Erreur transcription: {e}"}
                            # st.rerun()

                # --- SUPPRIMER TOUTE LA SECTION SUIVANTE --- 
                # (Affichage de la zone de texte spécifique à l'audio et son bouton)
                # transcribed_text = st.text_area(
                #     "Texte transcrit (éditable) :", 
                #     value=st.session_state.get(transcribed_text_key, ""),
                #     height=150,
                #     key=f"transcribed_display_{normalize_name(chapter)}"
                # )

                # if st.button("", key=f"save_audio_{normalize_name(chapter)}"):
                #     if transcribed_text.strip():
                #         try:
                #             # Lire le texte potentiellement édité
                #             edited_text = st.session_state[f"transcribed_display_{normalize_name(chapter)}"]
                #             save_memory(st.session_state.selected_template_name, chapter, edited_text)
                #             st.success(f"Souvenir (issu de l'audio) sauvegardé pour le chapitre '{chapter}' !")
                #             # Réinitialiser et cacher
                #             st.session_state[show_audio_recorder_key] = False
                #             st.session_state[transcribed_text_key] = ""
                #             st.rerun()
                #         except Exception as e:
                #              st.error(f"Erreur sauvegarde audio: {e}")
                #     else:
                #         st.warning("Le champ transcrit est vide.")

            # Suppression de l'ancienne section 'souvenir rapide'
            # temp_memory_text = st.text_area("Écrire un souvenir rapide :", key=f"temp_text_{normalize_name(chapter)}")
            # if st.button(f" Sauvegarder souvenir rapide pour '{chapter}'", key=f"save_temp_{normalize_name(chapter)}"):
            #     if temp_memory_text.strip():
            #         # Utilisation de la fonction save_memory mise à jour
            #         success, message, file_path = save_memory(st.session_state.selected_template_name, chapter, temp_memory_text)
            #         if success:
            #             st.success(message)
            #             # Effacer le champ après sauvegarde et rafraîchir
            #             st.session_state[f"temp_text_{normalize_name(chapter)}"] = ""
            #             st.rerun()
            #         else:
            #             st.error(f"Erreur sauvegarde : {message}")
            #     else:
            #         st.warning("Le champ souvenir est vide.")
            
            st.markdown("--- ")
            
            # --- Génération du chapitre --- 
            st.markdown("**Générer le chapitre :**")
            chapter_filename = normalize_name(chapter) + ".md"
            chapter_path = os.path.join(os.getcwd(), "chapitres", normalize_name(st.session_state.selected_template_name), chapter_filename)
            
            # Charger le chapitre généré s'il existe déjà
            generated_content = None
            if os.path.exists(chapter_path):
                try:
                    with open(chapter_path, 'r', encoding='utf-8') as f:
                        generated_content = f.read()
                except Exception as e:
                    st.warning(f"Impossible de lire le chapitre généré: {e}")

            # Bouton pour générer
            if st.button(f"✨ Générer le chapitre '{chapter}'", key=f"generate_{normalize_name(chapter)}", disabled=not memories):
                if memories:
                    with st.spinner("🧠 L'IA réfléchit et rédige..."): 
                        # Appel réel à la fonction de génération
                        generated_text, success = generate_chapter_with_ai(
                            st.session_state.api_key, 
                            st.session_state.selected_template_name, 
                            chapter, 
                            memories
                        )
                        
                        # Vérifier si la génération a réussi
                        if success and generated_text:
                            # Sauvegarder le chapitre généré
                            save_success, save_message, save_path = save_chapter(
                                st.session_state.selected_template_name, 
                                chapter, 
                                generated_text
                            )
                            
                            if save_success:
                                # Mettre à jour l'état de session pour affichage immédiat
                                st.session_state.generated_chapters[normalize_name(chapter)] = generated_text
                                # Stocker le message de succès dans le dictionnaire des messages par chapitre
                                chapter_key = normalize_name(chapter)
                                st.session_state.chapter_messages[chapter_key] = {"type": "success", "text": f"Chapitre '{chapter}' généré et sauvegardé ! {save_message}"}
                                st.session_state.active_expander = expander_key
                                st.rerun() # Recharger pour afficher
                            else:
                                # Stocker le message d'erreur dans le dictionnaire des messages par chapitre
                                chapter_key = normalize_name(chapter)
                                st.session_state.chapter_messages[chapter_key] = {"type": "error", "text": f"Erreur lors de la sauvegarde du chapitre généré: {save_message}"}
                                st.session_state.active_expander = expander_key
                                st.rerun() # Recharger pour afficher l'erreur
                        else:
                            # Afficher l'erreur retournée par la fonction de génération
                            # Stocker le message d'erreur dans le dictionnaire des messages par chapitre
                            chapter_key = normalize_name(chapter)
                            st.session_state.chapter_messages[chapter_key] = {"type": "error", "text": f"La génération du chapitre a échoué : {generated_text}"}
                            st.session_state.active_expander = expander_key
                            st.rerun() # Recharger pour afficher l'erreur
                else:
                     st.warning("⚠️ Ajoutez au moins un souvenir avant de générer.")
                     st.session_state.active_expander = expander_key

            elif not memories:
                # Si aucun souvenir, on ne fait rien mais on reste sur le même expander
                st.caption("⚠️ Ajoutez des souvenirs pour pouvoir générer ce chapitre.") # Remettre le caption
                st.session_state.active_expander = expander_key

            # Afficher le chapitre généré (s'il existe dans l'état de session ou fichier)
            display_content = st.session_state.generated_chapters.get(normalize_name(chapter), generated_content)
            if display_content:
                st.markdown("**Texte généré :**")
                st.markdown(display_content)
            
    # Suppression du bloc else problématique
    # else:
    #     st.warning("Ce modèle de livre ne contient aucun chapitre.")

# --- Nettoyage/Footer ---
# (Peut contenir des infos de version, liens, etc.)
