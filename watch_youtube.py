import os
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from playsound import playsound
from dotenv import load_dotenv
import threading

load_dotenv()

# YouTube API setup
API_KEY = os.getenv('YOUTUBE_API_KEY')
CHANNEL_ID = os.getenv('CHANNEL_ID')

def get_channel_videos():
    try:
        # Build the YouTube service
        youtube = build('youtube', 'v3', developerKey=API_KEY)
        
        # Get channel's uploads playlist ID
        channel_response = youtube.channels().list(
            id=CHANNEL_ID,
            part='contentDetails'
        ).execute()
        
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # Get recent videos from the uploads playlist
        playlist_response = youtube.playlistItems().list(
            playlistId=uploads_playlist_id,
            part='snippet',
            maxResults=5
        ).execute()
        
        return playlist_response['items']
        
    except HttpError as e:
        print(f'An HTTP error occurred: {e}')
        return []

def search_channel_by_name(channel_name):
    try:
        # Build the YouTube service
        youtube = build('youtube', 'v3', developerKey=API_KEY)
        
        # Search for channels with the given name
        search_response = youtube.search().list(
            q=channel_name,
            type='channel',
            part='snippet',
            maxResults=5
        ).execute()
        
        channels = []
        for item in search_response['items']:
            channel_info = {
                'id': item['id']['channelId'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'thumbnail': item['snippet']['thumbnails']['default']['url'],
                'url': f"https://www.youtube.com/channel/{item['id']['channelId']}"
            }
            channels.append(channel_info)
        
        return channels
        
    except HttpError as e:
        print(f'An HTTP error occurred: {e}')
        return []

def play_notification():
    try:
        playsound('Notification/notification.mp3')
    except Exception as e:
        print(f"Could not play notification sound: {e}")

def main():
    print(f"\nMonitoring channel {CHANNEL_ID} for new videos...")
    
    # Keep track of the last video we've seen
    last_video_id = "m2nelEfVVHc"
    
    while True:
        videos = get_channel_videos()
        
        if videos:
            latest_video = videos[0]
            current_video_id = latest_video['snippet']['resourceId']['videoId']

            print(f"current_video_id: {current_video_id}")
            
            # If this is a new video
            if current_video_id != last_video_id:
                video_title = latest_video['snippet']['title']
                published_at = latest_video['snippet']['publishedAt']
                print(f"\nNew video detected!")
                print(f"Title: {video_title}")
                print(f"Published at: {published_at}")
                print(f"URL: https://www.youtube.com/watch?v={current_video_id}")
                
                # Start notification sound in a separate thread
                sound_thread = threading.Thread(target=play_notification)
                sound_thread.start()
                
                last_video_id = current_video_id

                with open('youtube_links.txt', 'w') as f:
                    f.write(f"https://www.youtube.com/watch?v={current_video_id}\n")

                os.system("sh start.sh")
        
        # Wait for 5 minutes before checking again
        time.sleep(60)

if __name__ == "__main__":
    main()
