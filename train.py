import os
from ultralytics import YOLO
from utils.logger import setup_logger
from utils.download_helper import download_youtube_video
from utils.dataset_creator import convert_to_training_frames
from multiprocessing import Pool

logger = setup_logger(__name__)

RAW_DIR = "Dataset/raw/"
PROCESSED_DIR = "Dataset/Processed/"

def download_videos(logger):
    """
    Download videos from youtube links
    """
    global RAW_DIR
    try:
        # Get urls from the file
        urls = []

        file_count = 1

        with open("youtube_links.txt", "r") as f:
            for line in f:
                url = line.strip()
                if url:
                    urls.append((url, RAW_DIR, f"dataset_{file_count}.mp4", logger))
                    file_count += 1

        # Download videos in parallel
        with Pool(processes=len(urls)) as pool:
            pool.starmap(download_youtube_video, urls)

    except Exception as e:
        logger.error(f"Error downloading videos: {e}")

def create_dataset(logger):
    """
    Create a dataset from the videos
    """
    global RAW_DIR, PROCESSED_DIR

    for video_file in os.listdir(RAW_DIR):
        convert_to_training_frames(os.path.join(RAW_DIR, video_file), PROCESSED_DIR, logger)

def train_model(logger):
    """
    Train a model
    """
    # Load the model
    logger.info("Loading model")
    model = YOLO("yolo11n.pt")

    # Train the model
    logger.info("Training model")
    model.train(
        data="Dataset/Processed/data.yaml",
        epochs=20,
        imgsz=640,
        batch=8,
        workers=1,
        device="cuda",
        cache=False,
        optimizer="Adam",
        patience=10,
        save=True,
        name="yolo11n_trained"
    )
    
    # Save the model
    logger.info("Saving model")
    model.save("yolo11n_trained_new.pt")

if __name__ == "__main__":
    #download_videos(logger)
    create_dataset(logger)
    train_model(logger)