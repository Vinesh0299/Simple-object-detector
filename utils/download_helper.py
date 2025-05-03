import os
import yt_dlp
from urllib.parse import urlparse, parse_qs
from utils.logger import setup_logger

def is_valid_youtube_url(url):
    """
    Validate if the URL is a valid YouTube URL.
    
    Args:
        url (str): URL to validate
        
    Returns:
        bool: True if valid YouTube URL, False otherwise
    """
    try:
        parsed_url = urlparse(url)
        if parsed_url.netloc not in ['youtube.com', 'www.youtube.com', 'youtu.be']:
            return False
        if parsed_url.netloc == 'youtu.be':
            return len(parsed_url.path) > 1
        if parsed_url.netloc in ['youtube.com', 'www.youtube.com']:
            return 'v' in parse_qs(parsed_url.query)
        return False
    except:
        return False

def download_youtube_video(url, output_path, filename="dataset.mp4", logger=None):
    """
    Download a YouTube video using yt-dlp.

    Args:
        url (str): The URL of the YouTube video.
        output_path (str): The path to save the downloaded video.
        logger: Logger instance for logging messages.

    Returns:
        str: The path to the downloaded video.
    """
    try:
        # If logger is not provided, create a new one
        if logger is None:
            logger = setup_logger(__name__)

        # Validate URL
        if not is_valid_youtube_url(url):
            logger.error(f"Invalid YouTube URL: {url}")
            return None
        
        os.makedirs(output_path, exist_ok=True)

        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]',  # Target 720p or lower
            'outtmpl': os.path.join(output_path, filename),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'merge_output_format': 'mp4',  # Ensure output is MP4,
            'concurrent_fragments': 4,  # Download multiple fragments simultaneously
            'threads': 4,  # Use multiple threads for downloading
        }
        
        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Downloading video from: {url}")
            ydl.download([url])

        # Calculate and log download duration
        logger.info(f"Video downloaded successfully to: {output_path}")
        
        return output_path
        
    except Exception as err:
        logger.error(f"Error downloading video: {str(err)}")
        return None
