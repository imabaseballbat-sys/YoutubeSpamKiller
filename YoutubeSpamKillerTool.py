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

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
SUSPICIOUS_KEYWORDS = [
    "uttp", "tap_me_for_free_rbx", "free robux", "tap me", "click here", "free rbx", "1st warning troll", "YFGA", "ZNTP", "Don’t translate...😾",
    ""
]

TOKEN_PICKLE = "token.pickle"
quota_used = 0
DAILY_QUOTA_LIMIT = 9800

def show_setup_instructions():
    root = tk.Tk()
    root.withdraw()
    top = tk.Toplevel(root)
    top.withdraw()
    instructions = """
Welcome to YouTube Spam Killer Setup! 🎉

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

show_setup_instructions()
CLIENT_SECRETS_PATH = select_client_secrets_file()
if not CLIENT_SECRETS_PATH:
    print("❌ No file selected. Please run the program again and select the client_secrets.json file.")
    sys.exit(1)

def get_authenticated_service():
    creds = None
    if os.path.exists(TOKEN_PICKLE):
        with open(TOKEN_PICKLE, "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PICKLE, "wb") as token:
            pickle.dump(creds, token)
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
            maxResults=50,
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
        maxResults=9800
    )
    response = request.execute()
    quota_used += 1
    for item in response.get("items", []):
        top_comment = item["snippet"]["topLevelComment"]["snippet"]
        author = top_comment["authorDisplayName"]
        comment_text = top_comment["textDisplay"]
        comment_id = item["snippet"]["topLevelComment"]["id"]
        comments.append((author, comment_text, comment_id))

        if item["snippet"].get("totalReplyCount", 0) > 0:
            comments.extend(fetch_replies(youtube, comment_id))
    return comments

def fetch_replies(youtube, parent_id):
    global quota_used
    replies = []
    request = youtube.comments().list(
        part="snippet",
        parentId=parent_id,
        maxResults=9800
    )
    response = request.execute()
    quota_used += 1
    for item in response.get("items", []):
        reply = item["snippet"]
        author = reply["authorDisplayName"]
        comment_text = reply["textDisplay"]
        comment_id = item["id"]
        replies.append((author, comment_text, comment_id))
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

def is_bot(author, comment_text):
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
            for author, comment_text, comment_id in comments:
                if is_bot(author, comment_text):
                    print(f"🚨 Bot detected: {author} — {comment_text}")
                    delete_comment(youtube, comment_id)
                    time.sleep(1)
        print(f"✅ Sweep done! Quota used: {quota_used} / {DAILY_QUOTA_LIMIT}\n")
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
        ⚔️ YOUTUBE SPAM KILLER - Making YouTube A Better Place! ⚔️
""")
    consent = input("Type YES (all caps) to authorize and continue: ").strip()
    if consent != "YES":
        print("👋 Exiting. Stay spam-free!!!")
        return

    youtube = get_authenticated_service()
    channel_id = get_channel_id(youtube)

    print("\n🤖 Choose scan mode:")
    print("1️⃣  Latest video only")
    print("2️⃣  All videos (slower, more thorough)")
    scan_all = input("Enter 1 or 2: ").strip() == "2"

    print("🛡️ Auto-sweep every 10 mins. Type 'DELETE SPAM', 'EXIT', 'QUOTA', 'JOKE', 'BLACKJACK', 'DICE', 'ABOUT'   ")
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
            elif cmd == "BLACKJACK":
                print("🃏 Welcome to Blackjack! 🃏\n"
                      "Type 'hit' to draw a card, 'stay' to end your turn, or 'exit' to quit.")
                random_cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
                print("🃏 Drawing a random card...")
                card = random.choice(random_cards)
                card2 = random.choice(random_cards)
                card3 = random.choice(random_cards)
                card4 = random.choice(random_cards)
                card5 = random.choice(random_cards)
                print(f"You drew a {card} and {card2}! Do you want to hit or stay?")
                while True:
                    hit_or_stay = input("Type 'hit' to draw another card or 'stay' to end your turn: ").strip().lower()
                    if hit_or_stay == "hit":
                        card = random.choice(random_cards)
                        print(f"You drew a {card3}!")
                        if card + card2 + card3  <= 21:
                            print("Would you like to draw another card? (yes/no)")
                            if input().strip().lower() == "no":
                                print("Staying? Aright! Good luck! Dealer's turn now.")
                                break
                            if input().strip().lower() == "yes":
                                print(f"You drew a {card4}!")
                                if card + card2 + card3 +card4 <= 21:
                                    print("Would you like to draw another card? (yes/no)")
                                    if input().strip().lower() == "yes":
                                        print(f"You drew a {card5}!")
                                        if card + card2 + card3 + card4 + card5 <= 21:
                                            print("You stayed within 21! Good luck! Dealer's turn now.")
                                            break
                                        else:
                                            print("💥 You busted! Total: ", card + card2 + card3 + card4 + card5)
                                            break
                                    else:
                                        print("You chose to stay. Good luck! Dealer's turn now.")
                                        break
                        if card + card2 + card3 > 21:
                            print("💥 You busted! Total: ", card + card2 + card3)
                            break
                    elif hit_or_stay == "stay":
                        print("You chose to stay. Good luck!")
                    dealer_card = random.choice(random_cards)
                    dealer_card2 = random.choice(random_cards)
                    dealer_card3 = random.choice(random_cards)
                    dealer_card4 = random.choice(random_cards)
                    dealer_card5 = random.choice(random_cards)
                    dealer_total = dealer_card + dealer_card2 + dealer_card3 + dealer_card4 + dealer_card5
                    print(f"Dealer's cards: {dealer_card}, {dealer_card2},")
                    if dealer_total <11:
                        print(f"Dealer drew a {dealer_card3}.")
                        if dealer_card + dealer_card2 + dealer_card3 > 21:
                            print("Dealer busted! You win! 🎉")
                            break
                        if dealer_card + dealer_card2 + dealer_card3 <16:
                            print(f"Dealer drew a {dealer_card4}.")
                            if dealer_card + dealer_card2 + dealer_card3 + dealer_card4 > 21:
                                print("Dealer busted :) You win! 🎉")
                                break
                            if dealer_card + dealer_card2 + dealer_card3 + dealer_card4 <16:
                                print(f"Dealer drew a {dealer_card5}.")
                                if dealer_card + dealer_card2 + dealer_card3 + dealer_card4 + dealer_card5 > 21:
                                    print("Dealer busted :) You win! 🎉")
                                    break
                                if dealer_card + dealer_card2 + dealer_card3 + dealer_card4 + dealer_card5 <card + card2 + card3 + card4 + card5:
                                    print("Dealer's total is ", dealer_total, ". Dealer must hit again 🎉")
                                    print(f"Dealer drew a {dealer_card5}.")
                                    dealer_total += dealer_card5
                                    if dealer_total > 21:
                                        print("You win! Dealer busted! 🎉")
                                    if dealer_total > card + card2 + card3 + card4 + card5:
                                        print("Dealer won! Better luck next time.")
                                else:
                                    print(f"Dealer's total is {dealer_total}.")
                                    if dealer_total > card + card2 + card3 + card4 + card5:
                                        print("Dealer wins! Better luck next time.")
                                    elif dealer_total < card + card2 + card3 + card4 + card5:
                                        print("You win! 🎉")
                                    else:
                                        print("It's a tie!")

            elif hit_or_stay == "exit":
                print("👋 Exiting Blackjack. Goodbye!")
                break
                print("Invalid command. Please type 'hit', 'stay', or 'exit'.")
            elif cmd == "Random Website":
                random_websites = [
                    "https://www.vexrobotics.com/iq#learn-more-iq-competition"
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "https://patorjk.com/software/taag/#p=display&f=Epic&t=Youtube",
                    "https://www.youtube.com/watch?v=9bZkp7q19f0",


                ]
                
        if time.time() - last_sweep >= 600:
            run_spam_sweep(youtube, channel_id, scan_all)
            last_sweep = time.time()

if __name__ == "__main__":
    main()
