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
import hashlib
import webbrowser
import threading
import random
import pip
import pandas as pd
import pandas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import numpy as np
from PIL import Image
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
FILENAME = "/Users/gra/Documents/YoutubeSpamKillerAI/client_secrets_path.pickle"

if os.path.exists(FILENAME):
    try:
        with open(FILENAME, "rb") as file:
            path = pickle.load(file)
            print(f"Successfully loaded path from pickle: {path}")
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"Error loading pickle file: {e}. The file may be corrupted.")
        os.remove(FILENAME)
        print("Corrupted file has been deleted. Please run your main app again.")
else:
    print("Pickle file not found.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
        'warninng': 'warning'
    }
    processed_text = text.lower()
    for typo, correction in typo_corrections.items():
        processed_text = processed_text.replace(typo, correction)
    preprocessed_text = re.sub(r'([0-9]|[^a-zA-Z0-9\s])', r' \1 ', processed_text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', preprocessed_text)
    normalized_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return normalized_text

# This function MUST also be in your live script.
def extract_username_features(usernames):
    features = []
    for username in usernames:
        length = len(str(username))
        has_numbers = bool(re.search(r'\d', str(username)))
        has_uttp_keyword = 1 if "UTTP" in str(username).upper() else 0
        features.append([length, int(has_numbers), has_uttp_keyword])
    return np.array(features)

# --- Your main script code starts here ---

# Load the AI model pipeline
try:
    model = joblib.load('spam_killer_model_tuned.joblib')
    print("AI model loaded successfully! Ready to check comments.")
except FileNotFoundError:
    print("Error: The model file 'spam_killer_model_tuned.joblib' was not found.")
    print("Please make sure you have run your training script first.")
    exit()

# --- Example of how to use the model with a DataFrame ---
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
    "1st warning troll! My video is better than this incoherent & soulless goyslop UTTP's far better than whatever this nonsense is!",
    "dont read my name", "dontreadmypicture", "don't read my name", 



]

quota_used = 0
DAILY_QUOTA_LIMIT = 9900
CLIENT_SECRETS_PATH_PICKLE = os.path.join(SCRIPT_DIR, "client_secrets_path.pickle")
TOKEN_PICKLE = os.path.join(SCRIPT_DIR, "token.pickle")

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
    # Attempt to load the path to client_secrets.json
    client_secrets_path = None
    if os.path.exists(CLIENT_SECRETS_PATH_PICKLE):
        try:
            with open(CLIENT_SECRETS_PATH_PICKLE, "rb") as path_file:
                client_secrets_path = pickle.load(path_file)
        except (pickle.UnpicklingError, EOFError):
            print("❌ Corrupted client_secrets_path.pickle file detected. Deleting.")
            os.remove(CLIENT_SECRETS_PATH_PICKLE)

    # If the path is not found, prompt the user to select it
    if not client_secrets_path or not os.path.exists(client_secrets_path):
        show_setup_instructions()
        client_secrets_path = select_client_secrets_file()
        if not client_secrets_path:
            print("❌ No file selected. Exiting.")
            sys.exit(1)
        # Save the path for the next run
        with open(CLIENT_SECRETS_PATH_PICKLE, "wb") as path_file:
            pickle.dump(client_secrets_path, path_file)
    
    # Now, with a valid client_secrets_path, we can get or create the token
    creds = None
    if os.path.exists(TOKEN_PICKLE):
        with open(TOKEN_PICKLE, "rb") as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, SCOPES)
            creds = flow.run_local_server(port=0)
            
        with open(TOKEN_PICKLE, "wb") as token:
            pickle.dump(creds, token)
            
    return build("youtube", "v3", credentials=creds)

# Call the function to get the authenticated service





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
            maxResults=9900,
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
        maxResults=9900
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
        maxResults=9900
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
    allowed_patterns = [r'\bauttp\b', r'\banti-uttp\b']
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
            for author, comment_text, comment_id, profile_image_url in comments:
                
                
                # --- This is the correct Hybrid Detection Logic ---
        
                # 1. First, check your existing keyword filter (the fast option)
                if is_bot(author, comment_text, profile_image_url):
                    print(f"🚨 Bot detected by keyword filter: {author} — {comment_text}")
                    delete_comment(youtube, comment_id)
                    time.sleep(1)
                else:
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
                        delete_comment(youtube, comment_id)
                        time.sleep(1)   
                    
        print(f"✅ Sweep done! Quota used: {quota_used} / {DAILY_QUOTA_LIMIT}\n")
        print(f"Comment: {comment_text} | Raw Confidence: {prediction_proba}")
    except (HttpError, BrokenPipeError, ConnectionResetError) as e:
        print(f"💥 Network error: {e}. Retrying in 5 seconds...")
        time.sleep(5)
        run_spam_sweep(youtube, channel_id, scan_all)
    except Exception as e:
        print(f"❌ Error during sweep: {e}")
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
    scan_all = input("Enter 1 or 2: ").strip() == "2"

    

    print("🛡️ Auto-sweep every 10 mins. Type 'DELETE SPAM', 'EXIT', 'QUOTA', 'JOKE', 'RANDOM WEBSITE', 'DICE', 'ABOUT', 'ADD BANNED', 'VERIFICATION BYPASS', 'UPDATE LOG'  ")
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

                    
                ]
                print(random.choice(random_jokes))
                print("Want another joke?")
            elif cmd == "DICE":
            
                dice_roll = random.randint(1, 6)
                print(f"You rolled a {dice_roll} 🎲")
          
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
                    input("Verification bypass enabled. Press enter to terminate this process...")
            elif cmd == "UPDATE LOG":
                print(f"Changes in version 1.0.0 to 2.0.0:
                - Fixed bug where keyword filter doesn't catch some words
                - Fixed bug where you have to re enter your client secrets file every time you use the tool")
                - Added AI to detect typos or bypasses and delete the comments
                - Added "Verification Bypass" which bypasses Youtube's new AI Age verification feature by opening the video in a piped enviorment
                - Verification Bypass also blocks ads due to the piped enviorment
                - Increased quota limit from 9800 to 10000")
                
               

            
                
        if time.time() - last_sweep >= 600:
            run_spam_sweep(youtube, channel_id, scan_all)
            last_sweep = time.time()

if __name__ == "__main__":
    main()
