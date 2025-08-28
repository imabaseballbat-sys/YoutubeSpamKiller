from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
import pickle
import os
import time
import sys
import select
import re
import tkinter as tk
from tkinter import messagebox, filedialog
from datetime import datetime
import hashlib
import webbrowser
import smtplib
import json
from email.message import EmailMessage
import threading
import random
import traceback
import pandas as pd
import pandas
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import numpy as np
from PIL import Image
import tempfile
from io import BytesIO
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import joblib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import os.path
from cryptography.fernet import Fernet
from termcolor import colored
import socket
import io
from g4f.client import Client
from g4f.Provider import You
from g4f import ChatCompletion
from g4f import get_model_and_provider
import contextlib
from deep_translator import GoogleTranslator
from deepface import DeepFace
import cv2
from bs4 import BeautifulSoup
from googlesearch import search
import csv
import paypalrestsdk


TOKEN_PICKLE = "token.pickle"
KEY_FILE = "encryption_key.txt"
def get_or_create_key():
    """Gets the key from the key file or creates a new one."""
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, 'rb') as key_file:
            return key_file.read()
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, 'wb') as key_file:
            key_file.write(key)
        return key

key = get_or_create_key()
test_data = {"message": "Hello, world!"}
filepath = "test_file.pickle"
temp_filepath = filepath + '.tmp'
def save_credentials(creds):
    token_bytes = pickle.dumps(creds)
    fernet_key = get_or_create_key()
    encrypted_token = Fernet(fernet_key).encrypt(token_bytes)

    face_embedding = None
    if os.path.exists("SpamKillerVerification.jpg"):
        try:
            embedding_obj = DeepFace.represent(img_path="SpamKillerVerification.jpg", model_name="Facenet", enforce_detection=False)
            face_embedding = embedding_obj[0]["embedding"]
            print(colored("🔐 Face embedding loaded from existing picture.", "green"))
        except Exception as e:
            print(colored(f"❌ Face embedding failed: {e}", "red"))

    if face_embedding is None:
        user_input = input(colored("For more security, take a picture of yourself? (yes/no):", "green"))
        if user_input.strip().lower() == "yes":
            print(colored("📸 Opening camera... Smile!", "blue"))
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                print(colored("❌ Camera couldn't be opened!", "red"))
                return
            for _ in range(3):
                ret, frame = cap.read()
            if ret:
                alpha = 1.5
                beta = 65
                frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
                cv2.imwrite("SpamKillerVerification.jpg", frame)
                cap.release()
                print(colored("✅ Picture taken!", "green"))
                try:
                    embedding_obj = DeepFace.represent(img_path="SpamKillerVerification.jpg", model_name="Facenet", enforce_detection=False)
                    face_embedding = embedding_obj[0]["embedding"]
                    print(colored("🔐 Face embedding saved for verification.", "green"))
                except Exception as e:
                    print(colored(f"❌ Face embedding failed: {e}", "red"))
                    print(colored("❌ Token not saved due to failed face verification.", "red"))
                    return
            else:
                print(colored("❌ Failed to take picture.", "red"))
                return

    # Only save if face_embedding is present (or user said no)
    if face_embedding is not None or user_input.strip().lower() == "no":
        data_to_save = {"token": encrypted_token, "face_embedding": face_embedding}
        temp_file = TOKEN_PICKLE + '.tmp'
        with open(temp_file, 'wb') as token_file:
            pickle.dump(data_to_save, token_file)
            token_file.flush()
            os.fsync(token_file.fileno())
        os.rename(temp_file, TOKEN_PICKLE)
        print("✅ Token saved successfully.")
    else:
        print(colored("❌ Token not saved due to failed face verification.", "red"))

def load_credentials():
    import traceback
    if not os.path.exists(TOKEN_PICKLE):
        return None
    try:
        with open(TOKEN_PICKLE, 'rb') as token_file:
            data = pickle.load(token_file)
        encrypted_token = data.get("token")
        saved_face_embedding = data.get("face_embedding")
        fernet_key = get_or_create_key()
        creds = pickle.loads(Fernet(fernet_key).decrypt(encrypted_token))

        if saved_face_embedding is not None:
            # ...face verification logic...
            pass

        print("✅ Credentials loaded successfully.")
        return creds
    except Exception as e:
        print(f"❌ Error loading token: {e!r}")  # Print the full exception object
        traceback.print_exc()                    # Print the full traceback
        # Only delete if it's a pickle or decryption error
        if "pickle" in str(e).lower() or "decrypt" in str(e).lower():
            if os.path.exists(TOKEN_PICKLE):
                os.remove(TOKEN_PICKLE)
            print("Corrupted token file deleted.")
        else:
            print("Token not loaded, but not deleted. Check error above.")
        return None

def load_credentials():
    """Loads and decrypts credentials, checks face if required."""
    if not os.path.exists(TOKEN_PICKLE):
        return None
    try:
        with open(TOKEN_PICKLE, 'rb') as token_file:
            data = pickle.load(token_file)
        encrypted_token = data.get("token")
        saved_face_embedding = data.get("face_embedding")
        fernet_key = get_or_create_key()
        creds = pickle.loads(Fernet(fernet_key).decrypt(encrypted_token))

        # If face_embedding is set, verify current face
        if saved_face_embedding:
            print("🔐 Face verification required...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print(colored("❌ Camera couldn't be opened!", "red"))
                return None
            ret, frame = cap.read()
            if ret:
                cv2.imwrite("CurrentVerification.jpg", frame)
                cap.release()
                try:
                    current_embedding_obj = DeepFace.represent(img_path="CurrentVerification.jpg", model_name="Facenet")
                    current_face_embedding = current_embedding_obj[0]["embedding"]
                    # Compare embeddings (Euclidean distance)
                    distance = np.linalg.norm(np.array(saved_face_embedding) - np.array(current_face_embedding))
                    if distance > 10:  # Threshold, adjust as needed
                        print(colored("❌ Face verification failed! Token not loaded.", "red"))
                        return None
                    else:
                        print(colored("✅ Face verification passed!", "green"))
                except Exception as e:
                    print(colored(f"❌ Face verification error: {e}", "red"))
                    return None
            else:
                print(colored("❌ Failed to take picture.", "red"))
                return None

        print("✅ Credentials loaded successfully.")
        return creds
    except Exception as e:
        print(f"Error loading pickle file: {e}. The file may be corrupted.")
        if os.path.exists(TOKEN_PICKLE):
            os.remove(TOKEN_PICKLE)
        return None

# Your main script will call these functions as needed.
# For example:
# token = load_credentials()
# if not token:
#     # Prompt for authorization and then call save_credentials
#     creds = flow.run_local_server(port=0)
#     save_credentials(creds)
#     token = creds
# Encrypt and save
encrypted_data = Fernet(key).encrypt(pickle.dumps(test_data))
with open(temp_filepath, "wb") as f:
    f.write(encrypted_data)
os.rename(temp_filepath, filepath)
def encrypt_token(token_data, key):
    f = Fernet(key)
    return f.encrypt(token_data)

def decrypt_token(encrypted_data, key):
    f = Fernet(key)
    return f.decrypt(encrypted_data)
# Load and decrypt

# Helper function to encrypt and save a file
def encrypt_and_save(obj, filename, key):
    # This is a general example, your code may be different
    f = Fernet(key)
    pickled_obj = pickle.dumps(obj)
    encrypted_obj = f.encrypt(pickled_obj)
    
    with open(filename, 'wb') as file:
        file.write(encrypted_obj)
        # Add these two lines to force the OS to write the file immediately
        file.flush()
        os.fsync(file.fileno())

# Helper function to load and decrypt a file
def load_and_decrypt(filepath, key):
   # Load and decrypt
    with open(filepath, "rb") as f:
        encrypted_data = f.read()
        decrypted_data = Fernet(key).decrypt(encrypted_data)
        loaded_data = pickle.loads(decrypted_data)
    return loaded_data
        
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DeprecationWarning = 1
def encrypt_token(token_data, key):
    f = Fernet(key)
    return f.encrypt(token_data)

def decrypt_token(encrypted_data, key):
    f = Fernet(key)
    return f.decrypt(encrypted_data)

REQUIRED_PACKAGES = [
    'google-auth-oauthlib',
    'google-api-python-client',
    'requests',
    'pandas',
    'scikit-learn',
    'numpy',
    'Pillow',
    'joblib',
    'cryptography',
    'termcolor',
    'selenium',
    'g4f',
    'deepface',
    'deep-translator',
]

def install_dependencies():
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing missing dependencies...")

install_dependencies()

CLIENT_SECRETS_PATH_PICKLE = os.path.join(SCRIPT_DIR, "client_secrets_path.pickle")

if os.path.exists(CLIENT_SECRETS_PATH_PICKLE):
    try:
        with open(CLIENT_SECRETS_PATH_PICKLE, "rb") as file:
            path = pickle.load(file)
            print(f"Successfully loaded path from pickle: {path}")
    except (pickle.UnpicklingError, EOFError) as e:
       
        traceback.print_exc()
        os.remove(CLIENT_SECRETS_PATH_PICKLE)
        print("Corrupted file has been deleted. Please run your main app again.")
else:
    print("Pickle file not found.")


    
#_________________Image Feature Extractor_________________#
class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        image_features = []
        for url in X:
            try:
                response = requests.get(url)
                response.raise_for_status()

                img = Image.open(BytesIO(response.content))
                
                hist = img.histogram()
                features = np.array(hist)
                image_features.append(features)

            except Exception as e:
                print(f"Error processing image {url}: {e}")
                image_features.append(np.zeros(256))
        return np.array(image_features)
#_________________Username Feature Extractor_________________#
class UsernameFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # The ColumnTransformer passes a Series, not a DataFrame.
        # We need to convert it to a DataFrame to use iterrows().
        if isinstance(X, pd.Series):
            X = X.to_frame()

        # We'll create a list of lists, where each inner list has a consistent shape
        features = []
        for _, row in X.iterrows():
            username_features = []
            username = row['username']

            # Check if the username is a valid string
            if isinstance(username, str):
                # We'll use 1 for a valid username, and 0 for a missing one
                username_features.append(1)
            else:
                username_features.append(0)

            # Check if the username contains a link (a common spam feature)
            if isinstance(username, str) and ('http' in username or 'www.' in username):
                username_features.append(1)
            else:
                username_features.append(0)

            features.append(username_features)

        # Convert the list of lists into a numpy array
        return np.array(features)

#_________________Data Table_________________#

def custom_preprocessor(text):
    if not isinstance(text, str):
        return ""
    typo_corrections = {
        'warnining': 'warning',
        'warninng': 'warning',
        'warnnnnnnnin': 'warning'
    }
    processed_text = text.lower()
    for typo, correction in typo_corrections.items():
        processed_text = processed_text.replace(typo, correction)
    preprocessed_text = re.sub(r'([0-9]|[^a-zA-Z0-9\s])', r' \1 ', processed_text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', preprocessed_text)
    normalized_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return normalized_text
def extract_username_features(usernames):
    features = []
    for username in usernames:
        length = len(str(username))
        has_numbers = bool(re.search(r'\d', str(username)))
        has_uttp_keyword = 1 if "UTTP" in str(username).upper() else 0
        features.append([length, int(has_numbers), has_uttp_keyword])
    return np.array(features)


# Load the AI model pipeline
try:
    model = joblib.load('spam_killer_model_tuned.joblib')
    print("AI model loaded successfully! Ready to check comments.")
except FileNotFoundError:
    print("Error: The model file 'spam_killer_model_tuned.joblib' was not found.")
    print("Please make sure you have run your training script first.")
    exit()

SPAM_THRESHOLD = 0.5

# NOTE: The DataFrame MUST contain all columns the model was trained on.
comments_to_check = pd.DataFrame({
    'comment_text': ["This is a clean comment.", "U T T P", "1ST WARNINING TROLL!"],
    'username': ["CleanUser", "Spammer", "Spammer"],
    'profile_pic_url': ["https://clean.com", "https://spam.com", "https://spam.com"]
})

for index, row in comments_to_check.iterrows():
    comment_data = pd.DataFrame(row).T
    comment_text = row['comment_text']

    if not isinstance(comment_text, str):
        print(f"Skipping comment with invalid text: {repr(comment_text)}")
        continue

    prediction_proba = model.predict_proba(comment_data)[0][1]


    

    if prediction_proba >= SPAM_THRESHOLD:
        print(f"🚨 Spam detected (Confidence: {prediction_proba:.2f}): {comment_text}")
    else:
        print(f"✅ Clean comment (Confidence: {prediction_proba:.2f}): {comment_text}")

print("\n--- The urllib3 warning is not a critical error and can be ignored. ---")


test_spam_comment = "U T T P"
processed_spam = custom_preprocessor(test_spam_comment)
print(f"Processed spam comment: '{processed_spam}'")

# Test a clean comment
test_clean_comment = "This is a clean comment."
processed_clean = custom_preprocessor(test_clean_comment)
print(f"Processed clean comment: '{processed_clean}'")
#_______________Keywords______________#


SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
SUSPICIOUS_KEYWORDS = [
    "uttp", "tap_me_for_free_rbx", "free robux", "tap me", "click here", "free rbx", "1st warning troll", "yfga", "zntp", "don’t translate...😾",
    "1st warning troll! my video is better than this incoherent & soulless goyslop uttp's far better than whatever this nonsense is!",
    "dont read my name", "dontreadmypicture", "don't read my name", "don’t translate...😾 dago saaqootah addat, ku sorkocô baxi atka soolele. dubuh yan giti tet kalaloonuh yi chaanaal sabiskiraayib abaanama……. . ……"



]
DeprecationWarning = 1
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_SECRETS_PATH_PICKLE = os.path.join(SCRIPT_DIR, "client_secrets_path.pickle")
TOKEN_PICKLE = os.path.join(SCRIPT_DIR, "token.pickle")
KEY_FILE = os.path.join(SCRIPT_DIR, 'encryption_key.txt')
DAILY_QUOTA_LIMIT = 10000
quota_used = 0

def show_setup_instructions():
    root = tk.Tk()
    root.withdraw()
    top = tk.Toplevel(root)
    top.withdraw()
    instructions = """
Welcome to YouTube Spam Killer Setup! 🎉

To start, you need a client_secrets.json file to get this program working! Steps are below on how to get it 😁:

How to get your client_secrets.json:

1️⃣ Go to Google Cloud Console: https://console.cloud.google.com/
2️⃣ Create a new project.
3️⃣ Enable YouTube Data API v3.
4️⃣ Create OAuth 2.0 credentials (Desktop app).
5️⃣ Download the client_secrets.json file.
6️⃣ Select the client_secrets.json file in the next window.

Click OK to select your client_secrets.json file.
"""
    messagebox.showinfo("Setup Instructions", instructions, parent=top)
    top.destroy()
    root.destroy()

def select_client_secrets_file():
    root = tk.Tk()
    root.withdraw()
    top = tk.Toplevel(root)
    top.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select client_secrets.json",
        filetypes=[("JSON files", "*.json")],
        parent=top
    )
    top.destroy()
    root.destroy()
    return file_path
def get_authenticated_service():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, 'wb') as key_file:
            key_file.write(key)
        print(colored("🔑 New encryption key created.", "yellow"))

    with open(KEY_FILE, 'rb') as key_file:
        key = key_file.read()

    client_secrets_path = None
    
    if os.path.exists(CLIENT_SECRETS_PATH_PICKLE) and os.path.getsize(CLIENT_SECRETS_PATH_PICKLE) > 0:
        try:
            with open(CLIENT_SECRETS_PATH_PICKLE, "rb") as file:
                path = pickle.load(file)
                print(f"Successfully loaded path from pickle: {path}")
        except (pickle.UnpicklingError, EOFError) as e:
            traceback.print_exc()
            os.remove(CLIENT_SECRETS_PATH_PICKLE)
            print("Corrupted file has been deleted. Please run your main app again.")
    else:
        print("Pickle file not found or is empty.")
    
    if not client_secrets_path or not os.path.exists(client_secrets_path):
        show_setup_instructions()
        client_secrets_path = select_client_secrets_file()
        if not client_secrets_path:
            print(colored("❌ No file selected. Exiting.", "red"))
            sys.exit(1)
        encrypt_and_save(client_secrets_path, CLIENT_SECRETS_PATH_PICKLE, key)

    creds = None
    if os.path.exists(TOKEN_PICKLE):
        try:
            creds = load_and_decrypt(TOKEN_PICKLE, key)
        except Exception as e:
            print(colored(f"❌ Error loading token: {e}. Deleting corrupted token file.", "red"))
            traceback.print_exc()
            print("Step 1: Credentials object:", creds)
            token_bytes = pickle.dumps(creds)
            print("Step 2: Pickled credentials (bytes):", token_bytes)

            fernet_key = get_or_create_key()
            encrypted_token = Fernet(fernet_key).encrypt(token_bytes)
            print("Step 3: Encrypted token:", encrypted_token)

    # ... (rest of the save function, including face verification and final save)
            os.remove(TOKEN_PICKLE)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print(colored("🔄 Token expired. Refreshing...", "cyan"))
            creds.refresh(Request())
        else:
            print(colored("🔒 No valid token found. Authorizing...", "yellow"))
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, SCOPES)
            creds = flow.run_local_server(port=0)

    print(colored("💾 Saving new token...", "cyan"))
    save_credentials(creds)
    print(colored("✅ Token saved successfully.", "green"))

        
    return build("youtube", "v3", credentials=creds)

def get_channel_id(youtube):
    request = youtube.channels().list(part="id", mine=True)
    response = request.execute()
    return response["items"][0]["id"]

def get_all_video_ids(youtube, channel_id):
    global quota_used
    video_ids = []
    next_page_token = None
    while True:
        if quota_used + 1 > DAILY_QUOTA_LIMIT:
            print("🚫 Quota limit reached while fetching videos.")
            break
        request = youtube.search().list(
            part="id",
            channelId=channel_id,
            order="date",
            maxResults=10000,
            type="video",
            pageToken=next_page_token
        )
        response = request.execute()
        quota_used += 1
        for item in response.get("items", []):
            video_ids.append(item["id"]["videoId"])
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return video_ids

def get_latest_video_id(youtube, channel_id):
    global quota_used
    if quota_used + 1 > DAILY_QUOTA_LIMIT:
        raise RuntimeError("Quota exceeded.")
    request = youtube.search().list(
        part="id",
        channelId=channel_id,
        order="date",
        maxResults=1,
        type="video"
    )
    response = request.execute()
    quota_used += 1
    return response["items"][0]["id"]["videoId"]

def fetch_comments(youtube, video_id):
    global quota_used
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=10000
    )
    response = request.execute()
    quota_used += 1
    for item in response.get("items", []):
        top_comment = item["snippet"]["topLevelComment"]["snippet"]
        author = top_comment["authorDisplayName"]
        comment_text = top_comment["textDisplay"]
        comment_id = item["snippet"]["topLevelComment"]["id"]
        profile_image_url = top_comment["authorProfileImageUrl"] # <-- The fix is here
        
        # Append all four values to the list
        comments.append((author, comment_text, comment_id, profile_image_url))

        if item["snippet"].get("totalReplyCount", 0) > 0:
            comments.extend(fetch_replies(youtube, comment_id))
    return comments

def fetch_replies(youtube, parent_id):
    global quota_used
    replies = []
    request = youtube.comments().list(
        part="snippet",
        parentId=parent_id,
        maxResults=10000
    )
    response = request.execute()
    quota_used += 1
    for item in response.get("items", []):
        reply = item["snippet"]
        author = reply["authorDisplayName"]
        comment_text = reply["textDisplay"]
        comment_id = item["id"]
        profile_image_url = reply["authorProfileImageUrl"] # <-- The fix is here
        
        # Append all four values to the list
        replies.append((author, comment_text, comment_id, profile_image_url))
    return replies

def delete_comment(youtube, comment_id):
    global quota_used
    if quota_used + 50 > DAILY_QUOTA_LIMIT:
        print("🚫 Skipping delete due to quota limit.")
        return
    try:
        youtube.comments().delete(id=comment_id).execute()
        quota_used += 50
        print(f"🧨 Deleted comment ID: {comment_id}")
        
        

    except Exception as e:
        print(f"⚠️ Failed to delete comment {comment_id}: {e}")

def is_bot(author, comment_text, profile_image_url):
    author_lower = author.lower()
    text_lower = comment_text.lower()
    allowed_patterns = [r'\bauttp\b', r'\banti-uttp\b', r'\banti uttp\b', r'\banti_uttp\b']
    for pattern in allowed_patterns:
        if re.search(pattern, author_lower) or re.search(pattern, text_lower):
            return False
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword == "uttp":
            if re.search(r'\buttp\b', author_lower) or re.search(r'\buttp\b', text_lower):
                return True
        elif keyword in author_lower or keyword in text_lower:
            return True
    return False

def run_spam_sweep(youtube, channel_id, scan_all=False):
    try:
        video_ids = get_all_video_ids(youtube, channel_id) if scan_all else [get_latest_video_id(youtube, channel_id)]
        for video_id in video_ids:
            print(f"🔍 Scanning video: {video_id}")
            comments = fetch_comments(youtube, video_id)
            
            def openai_check(comment_text, profile_image_url):
                prompt = f"Determine if this is spam: '{comment_text, profile_image_url}'and translate it to english if it isnt in english already and then scan it also be as quick as you can between scanning without sacrificing accuracy one last thing if a comment is spam always say yes also the the profile image url is their profile picture so scan that for potentially inapropriate stuff (yes/no)"
                response = ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                    )
                answer = str(response).lower()
                print(response)
                return "yes" in answer
            
            def openai_check4(comment_text, profile_image_url):
                prompt = f"Determine if this is spam: '{comment_text, profile_image_url}' and translate it to english if it isnt in english already and then scan it also be as quick as you can between scanning without sacrificing accuracy one last thing if a comment is spam always say  just yes nothing else one word also the the profile image url is their profile picture so scan that for potentially inapropriate stuff(yes/no)"
                response = ChatCompletion.create(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": prompt}]
                    )
                answer = str(response).lower()
                print(response)
                return "yes" in answer
            
            

            for author, comment_text, comment_id, profile_image_url in comments:
                
                # --- This is the correct Hybrid Detection Logic ---
                translated = GoogleTranslator(source='auto', target='es'.translate("{comment_text}"))
                # 1. First, check your existing keyword filter (the fast option)
                if is_bot(author, comment_text, profile_image_url):
                    print(f"🚨 Bot detected by keyword filter: {author} — {comment_text}")
                    delete_comment(youtube, comment_id)
                    time.sleep(1)
                if openai_check(comment_text, profile_image_url):
                    if openai_check(comment_text, profile_image_url):
                        print(f"🚨 Bot detected by OpenAI: {author} — {comment_text}")
                        delete_comment(youtube, comment_id)
                if openai_check4(comment_text, profile_image_url):
                    if openai_check4(comment_text, profile_image_url):
                        print(f"🚨 Bot detected by OpenAI 4.1: {author} - {comment_text}")
                        delete_comment(youtube, comment_id)
                
               
                  
                
                    # 2. If not a keyword bot, run the AI check
          
                    # Create the DataFrame for the AI model with all features
                        comment_data = pd.DataFrame({
                        'comment_text': [comment_text],
                        'username': [author],
                        'profile_pic_url': [profile_image_url] 
                    })
                  
                    # Correctly pass the DataFrame to the AI model
                    prediction = model.predict(comment_data)
                    prediction_proba = model.predict_proba(comment_data)[0][1]
              
                    if prediction[0] == 1:
                        print(f"🚨 Bot detected by AI model: {author} — {comment_text}")
                        # FIX 2: Added a print statement to show raw confidence
                        print(f"Comment: {comment_text} | Raw Confidence: {prediction_proba}")
                        delete_comment(youtube, comment_id)
                        time.sleep(1)   
                        if prediction_proba >= 0.7:
                            print("🚨 Spam probability high, reporting comment and channel...")
                            try:
                                youtube.comments().reportabuse(
                                id=comment_id,
                                abuseType = 'SPAM',
                                notes = 'Spammmm comment'                                                       
                                ).execute()
                                print("✅ Comment reported sucsessfully, reporting channel...")
                                DRY_RUN = False
                                spam_channels_queue = ["{channel_id}"]
                                print("Please select your browser:" \
                                "1 = Chrome... " \
                                "2 = FireFox... " \
                                "3 = Safari... ")

                                choice = input("Enter choice 1/2/3...")

                                if choice == "1":
                                    driver = webdriver.Chrome()
                                elif choice == "2":
                                    driver = webdriver.Firefox()
                                elif choice == "3":
                                    driver = webdriver.Safari()
                                else:
                                    print("Invalid choice. Please try again")
                                def report_channel(channel_id):
                                    url = "https://www.youtube.com/@{username}"
                                    driver.get(url)
                                    time.sleep(random.uniform(3,6))


                                    try:
                                        menu_button = driver.find_element(By.XPATH, "//yt-icon-button[@id='button' or @aria-label='More actions']")
                                        menu_button.click()
                                        time.sleep(random.uniform(2,4))
                                        if DRY_RUN:
                                            print(f"Would normally report {channel_id}")
                                        else:
                                            spam_checkbox = driver.find_element(By.XPATH, "//tp-yt-paper-checkbox[@name='SPAM_OR_MISLEADING']")
                                            spam_checkbox.click()
                                            time.sleep(random.uniform(1,2))
                                            submit_button = driver.find_element(By.XPATH, "//yt-button-renderer[@id='submit-buttom']")
                                            submit_button.click()
                                            print(f"✅ Channel {channel_id} reported")
                                    except Exception as e:
                                        print(f"❌ Error reporting channel {channel_id}")
                                        traceback.print_exc()

                                for cid in spam_channels_queue:
                                    report_channel(cid)
                                driver.quit()
                            except Exception as e:
                                print("❌ Failed to report comment")
                                traceback.print_exc()
                    
                   
                        
    except Exception as e:
        print(f"❌ Error during sweep: {e}")
        traceback.print_exc()
                    
        print(f"✅ Sweep done! Quota used: {quota_used} / {DAILY_QUOTA_LIMIT}\n")
    except (HttpError, BrokenPipeError, ConnectionResetError) as e:
        print(f"💥 Network error: {e}. Retrying in 5 seconds...")
        time.sleep(5)
        run_spam_sweep(youtube, channel_id, scan_all)
    except Exception as e:
        print(f"❌ Error during sweep: {e}")
        traceback.print_exc()
def main():
    print(r"""
          _______          _________          ______   _______ 
|\     /|(  ___  )|\     /|\__   __/|\     /|(  ___ \ (  ____ \
( \   / )| (   ) || )   ( |   ) (   | )   ( || (   ) )| (    \/
 \ (_) / | |   | || |   | |   | |   | |   | || (__/ / | (__    
  \   /  | |   | || |   | |   | |   | |   | ||  __ (  |  __)   
   ) (   | |   | || |   | |   | |   | |   | || (  \ \ | (      
   | |   | (___) || (___) |   | |   | (___) || )___) )| (____/\
   \_/   (_______)(_______)   )_(   (_______)|/ \___/ (_______/
        ⚔️ YOUTUBE SPAM KILLER - Making YouTube spam free! ⚔️
""")
    consent = input("Type YES (Must be all caps) to authorize and continue: ").strip()
    if consent != "YES":
        print("👋 Exiting. Stay spam-free!!!")
        return

    youtube = get_authenticated_service()
    channel_id = get_channel_id(youtube)
    print("\n🤖 Choose scan mode:")
    print("1️⃣  Latest video only")
    print("2️⃣  All videos (slower, more thorough)")
    print("3️⃣  Specific video")
    print("4️⃣  Enter a video id")
    print("5️⃣  Enter a channel id")
    scan_all = input("Enter 1, 2, 3, 4, or 5: ").strip() == "2"

    

    print("🛡️ Auto-sweep every 10 mins. Type 'DELETE SPAM', 'EXIT', 'QUOTA', 'JOKE', 'RANDOM WEBSITE', 'DICE', 'ABOUT', 'ADD BANNED', 'VERIFICATION BYPASS', 'ANTI GRAVITY'  ")
    last_sweep = 0

    while True:
        i, _, _ = select.select([sys.stdin], [], [], 5)
        if i:
            cmd = sys.stdin.readline().strip().upper()
            if cmd == "DELETE SPAM":
                
                run_spam_sweep(youtube, channel_id, scan_all)
                last_sweep = time.time()
            elif cmd == "EXIT":
                print("👋 Goodbye!")
                break
            elif cmd == "QUOTA":
                print(f"Currently, you are using {quota_used}. You still have {DAILY_QUOTA_LIMIT - quota_used} left for today 😁")
            elif cmd == "ABISGABISFLABIS":
                print("What")
            elif cmd == "I LOVE UTTP":
                print("GET REKT UTTP SCUMMER")
            elif cmd == "IUDF":
                print("I really don't understand")
            elif cmd == "ABOUT":
                print("This is a YouTube Spam Killer script designed to help you clean up spammy comments on your channel. It uses the YouTube Data API to fetch comments and delete those that match suspicious patterns.")
            elif cmd == "JOKE":
                
                random_jokes = [
                    "Why did the scarecrow win an award? Because he was outstanding in his field! 😂",
                    "Why don't skeletons fight each other? They don't have the guts! 😂",
                    "What do you call fake spaghetti? An impasta! 😂",
                    "Why did the bicycle fall over? Because it was two-tired! 😂",
                    "why did the computer go to the doctor? Because it had a virus! 😂",
                    "Why did the math book look sad? Because it had too many problems! 😂",
                    "Why did the golfer bring two pairs of pants? In case he got a hole in one! 😂",
                    "Why don't scientists trust atoms? Because they make up everything! 😂",
                    "How do trees access the internet? They log in! 😂",
                    "What did the sushi say to the bee? Wasabi! 😂",
                    "Why did the tomato turn red? Because it saw the salad dressing! 😂",
                    "What do you call cheese that isn't yours? Nacho cheese! 😂",
                    "9 + 10 = 21! Just kidding, it's 19! 😂",
                    "baller"


                    
                ]
                print(random.choice(random_jokes))
                print("Want another joke?")
            elif cmd == "DICE":
            
                dice_roll = random.randint(1, 6)
                print(f"You rolled a {dice_roll} 🎲")
            elif cmd == "LOCKED: $MONEY PRINTER$ locked":
                print("💸 Money printer goes brrrrr... 💸")
                time.sleep(5)
                print("Keep this program open to help keep your channel safe from bots 🤑✌️")
                def show_paypal_instructions():
                    root = tk.Tk()
                    root.withdraw()
                    top = tk.Toplevel(root)
                    top.withdraw()
                    instructions = """
Welcome to YouTube Spam Killer Setup Money Printer! 🎉

To start, you need a client_id and client_secret code to get the Money Printer working! Steps are below on how to get it 😁:

How to get your client_id and client_secret:

1️⃣ Go to Paypal Account creator: https://www.paypal.com/us/webapps/mpp/account-selection
2️⃣ Create a new buisness account.
3️⃣ Go to developer.
4️⃣ Go to API/SDKs.
5️⃣ Find your client_id and client_secret.
6️⃣ Select the client_secrets and client_id file when prompte in the terminal.

Type them in when prompted in the terminal.
"""
                    messagebox.showinfo("Setup Instructions", instructions, parent=top)
                    top.destroy()
                    root.destroy()
                paypalrestsdk.configure({
                    "mode": "sandbox",
                    "client_id": input("Enter your Paypal Buisness Client ID:"),
                    "client_secret": input("Enter your Paypal Buisness Client secret:")
                })
                payment = paypalrestsdk.Payment({
                    "intent": "sale",
                    "payer": {"payment_method": "paypal"},
                    "transactions": [{
                        "amount": {
                            "total": "300.00",
                            "currency": "USD"
                        },
                        "description": "1000 Client leads. Improve your buisness with a client lead package!"
                    }],
                    "redirect_urls": {
                        "return_url": "http://localhost:3000/payment/execute",
                        "cancel_url": "http://localhost:3000/cancel"
                    }
                })
                def find_emails(query, num_results=10):
                    emails = set()
                    for url in search(query, num_results=num_results):
                        try:
                            # Ensure the URL is valid (starts with http)
                            if not url.startswith("http"):
                                print(f"Skipping invalid URL: {url}")
                                traceback.print_exc()
                                continue
                            page = requests.get(url, timeout=5)
                            soup = BeautifulSoup(page.text, "html.parser")
                            
                            found = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", soup.text)
                            for email in found:
                                emails.add(email)
                        except Exception as e:
                            print(f"Error fetching {url}: {e}")
                            traceback.print_exc()
                            continue
                    return emails
           
                
                LEADS_FILE = "data/leads.csv"
                def openai_lead_detector(comment_text):
                    prompt = f"You are a lead generation bot :) . Find potential buisness oppurtinities if they seem to be fair leads. A buisness oppurtinity is someone who is searching for a specific service or product, looking to hire someone, looking for a partnership/collaberation, showing intrest in a niche topic that could make some cash 💰. Return 'YES' if you find an oppurtinuty :) Also translate comments if they arent in english already :)"
                    response = ChatCompletion.create(
                        model="gpt-4.1",
                        messages=[{"role": "user", "content": prompt}]
                        )
                def save_lead(comment_text, gpt_response):
                    lead = {
                        "comment": comment_text,
                        "gpt_response": gpt_response
                    }
                    df = pd.DataFrame([lead])
                    try:
                        exsisting_df = pd.read_csv(LEADS_FILE)
                        df = pd.concat([exsisting_df, df], ignore_index=True)
                    except FileNotFoundError:
                        pass
                    df.to_csv(LEADS_FILE, index=False)
                    print("✅ Lead saved!")
                def run_lead_bot(comments):
                    for c in comments:
                        author2, comment_text2, comment_id2, profile_image_url2 = c
                        if openai_lead_detector(comment_text2):
                            result = openai_lead_detector(c)
                            if "YES" in result.lower():
                                save_lead(c, result)
                            else:
                                print("Lead not found")
                            gpt_response = openai_lead_detector(comment_text2)
                            save_lead(comment_text2, gpt_response)
                            print(colored(f"💸 Lead found and saved 💸"))
                def generate_query():
                    prompt = f"Generate a google search query to find potential leads for a buisness that does web design, web development, app development, seo, digital marketing, graphic design, video editing, 3d modelling, 3d animation, ai services etc. The query should be in english and should be specific to finding potential clients or buisness oppurtinities. The query should not be too broad or too narrow. The query should be in the format of a question. The query should not contain any special characters or punctuation. The query should be concise and to the point. The query should not contain any personal information or sensitive data. The query should not contain any negative or harmful language. The query should not contain any spammy or clickbait language. The query should not contain any misleading or false information. The query should not contain any copyrighted material or trademarks. The query should not contain any illegal or unethical content. The query should be professinal but very persuasive and use psyology tricks to help improve the chances of getting a deal"
                    response = ChatCompletion.create(
                        model="gpt-4.1",
                        messages=[{"role": "user", "content": prompt}]
                        )
                def generate_email_contents():
                    prompt = f"Imagine this: You are a lawyer who is trying to get the judge to rule in your favor. But instead of a lawyer, you are a buisnessman trying to convince a company or buisness to buy leads from you. Assume that they won't care one second unless it looks legitamate and a good deal. Try your absolute best to convince them to buy your leads and you are allowed to use psycology tricks and tactics but make sure that whatever you say is ethical and professinal."
                    response = ChatCompletion.create(
                        model="gpt-4.1",
                        messages=[{"role": "user", "content": prompt}]
                        )
                AI_RESSPONSE = generate_email_contents()
                query = generate_query()
                emails = find_emails(query, num_results=20)
                EMAIL_ADRESS = find_emails('"contact", "Buisness", "@gmail.com"')
                def email_leads():
                    gpt_response = openai_lead_detector(comment_text)
                    try:
                        clients_df = pd.read_csv("data/clients.csv")
                    except FileNotFoundError:
                        print("No client found")
                        return
                    industry = "Unknown"
                    if "," in gpt_response:
                        parts = gpt_response.split(",")
                        if len(parts) > 1:
                            industry = parts[1].strip()
                    matching_clients = clients_df[clients_df['industry'].str.contains(industry, case=False, na=False)]
                    if matching_clients.empty:
                        print(colored("❌ No matching clients found for lead", "red"))
                        return
                    for _, client in matching_clients.iterrows():
                        msg = EmailMessage()
                        msg['subject'] = f"New lead found in your industry!: {industry}"
                        
                        msg['from'] = input("Please enter an email adress that is NOT your personal email adress: ")
                        msg['to'] = EMAIL_ADRESS
                        msg.set_content(f"Dear {client['name']},\n\n{AI_RESSPONSE}")
                        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                            smtp.login(EMAIL_ADRESS,) # Email password or app password)
                            smtp.send_message(msg)
                            print(colored(f"✅ Email sent to {client['name']} at {client['email']}", "green"))
                    clients = []
                    with open("leads.csv", newline="", encoding="utf-8") as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            clients.append({"name": row["name"], "email": row["email"]})
                    emailed_clients = set()
                    for client in clients:
                        if client['email'] in emailed_clients:
                            #Skip duplicate email to the same client
                            continue

                    
                # Email leads to clients 

                    



             

                
                
          
            elif cmd == "RANDOM WEBSITE":
                random_websites = [
                    "https://www.vexrobotics.com/iq#learn-more-iq-competition"
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "https://patorjk.com/software/taag/#p=display&f=Epic&t=Hello",
                    "https://www.youtube.com/watch?v=9bZkp7q19f0",

                    "https://sketchfab.com/search?q=vdbhfskgolhvwioevwfygerurbeyv&type=models",


                ]
            elif cmd == "ADD BANNED":
                new_keyword = input("Enter the keyword(s) to ban. Please separate each word with a coma and make sure it's all lowercase otherwise the bot can't pick up on it 😄 : ").strip()
                if new_keyword:
                    SUSPICIOUS_KEYWORDS.extend([word.strip() for word in new_keyword.split(",")])
                    print(f"✅ Added new keywords: {new_keyword}")
                else:
                    print("Please enter something to add to the list.")
            
            elif cmd == "DARLING HOLD MY HAND":
                print("Nothing beats a Jet2 Holiday 😂")
            
            

            elif cmd == "SCHLEP":
                print("Bro is a legend. Roblox is terrible right now")
            elif cmd == "VERIFICATION BYPASS":
                print("Verification bypass activating...")
                print("Please select your browser:" \
                "1 = Chrome... " \
                "2 = FireFox... " \
                "3 = Safari... ")

                choice = input("Enter choice 1/2/3...")

                if choice == "1":
                    driver = webdriver.Chrome()
                elif choice == "2":
                    driver = webdriver.Firefox()
                elif choice == "3":
                    driver = webdriver.Safari()
                else:
                    print("Invalid choice. Please try again")

                def inject_piped_redirect(driver):
                    js_code = """ 
                    document.addEventListener('click', function(event) {
                        let target = event.target;" \
                        while (target && target.tagname !== 'A') {
                            target = target.parentElement;
                        } 
                        if (target && target.href && target.href.includes('youtube.com/watch')) {
                            event.preventDefault();" 
                            const newURL = target.href.replace('youtube.com', 'piped.kavin.rocks'); 
                            window.location.href = newURL; 
                    } 
                }, true);
                    """
                    driver.execute_script(js_code)
                    driver.get("https://www.youtube.com")
                    inject_piped_redirect(driver)
            elif cmd == "CHANGELOG":
                print(colored("New changes in version 3.0.0:", "orange"))
                print(colored("Added 3rd and 4th layer of spam detection so absolutely no bots get by 😁", "blue"))
                print(colored("Added language translation so bots trying to cheat the system still get caught 🍭", "cyan"))
                print(colored("Added a feature where the bot promotes itself on different programs ⌨️", "green"))
                print(colored("Added some more cool features to pass the time while you are bored 😺", "yellow"))
                print(colored("Added a new GUI 👀✨", "purple"))
                print(colored("Added some bug fixes in 🐛", "red"))
                print(colored("Added a new feature to report channels and spam comments for max bot destruction 💀", "magenta"))
                print(colored("Added facial recognition to help encrypt your tokens 🔐", "yellow"))
            elif cmd == "ANTIGRAVITY":
                import antigravity
        if time.time() - last_sweep >= 600:
            run_spam_sweep(youtube, channel_id, scan_all)
            last_sweep = time.time()

if __name__ == "__main__":
    main()
