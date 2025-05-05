from ultralytics import YOLO
from utils.logger import setup_logger
from utils.download_helper import download_youtube_video
from multiprocessing import Pool
from queue import Queue
from uuid import uuid4
import threading
import time
import cv2
import os

logger = setup_logger(__name__)

RAW_DIR = "Dataset/raw-live/"
PROCESSED_DIR = "Dataset/processed-live/"
RESULTS_DIR = "./Results/"

# Create a queue for frame chunks
frame_queue = Queue(maxsize=50)  # Limit queue size to prevent memory issues

# Create a stop event
stop_event = threading.Event()

# Perform object detection
model = YOLO("yolo11n_trained.pt")

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

def consumer():
    """Consumer thread that processes frame chunks"""
    try:
        while not stop_event.is_set():
            try:
                frame_chunk = frame_queue.get(timeout=1)
            except Exception as e:
                logger.error(f"No data found in queue")
                continue
                
            # Process your frame chunk here
            # For example: save frames, perform detection, etc.
            logger.info(f"Processing chunk of {len(frame_chunk)} frames")

            results = model(frame_chunk, imgsz=640)

            for i, result in enumerate(results):
                # Access the results
                xywh = result.boxes.xywh  # center-x, center-y, width, height
                xywhn = result.boxes.xywhn  # normalized
                xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
                xyxyn = result.boxes.xyxyn  # normalized
                names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
                confs = result.boxes.conf  # confidence score of each box

                # Only save frames where at least one detection has confidence > 0.5
                if names and any(conf > 0.7 for conf in confs):
                    id = uuid4()
                    result.save(f"{RESULTS_DIR}/result_{id}.jpg")
                    cv2.imwrite(f"{PROCESSED_DIR}/raw_result_{id}.jpg", frame_chunk[i])
            
            frame_queue.task_done()
            
        logger.info("Consumer thread finished")
    except Exception as e:
        logger.error(f"Error in consumer thread: {e}")

def producer():
    """Producer thread that reads frames from videos"""
    try:
        for video_file in os.listdir(RAW_DIR):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(RAW_DIR, video_file)
                cap = cv2.VideoCapture(video_path)
                
                frame_chunk = []
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1

                    if frame_count % 12 != 0:
                        continue

                    frame_chunk.append(frame)
                    if len(frame_chunk) == 5:
                        frame_queue.put(frame_chunk)
                        frame_chunk = []
                
                # Put remaining frames if any
                if frame_chunk:
                    frame_queue.put(frame_chunk)
                
                cap.release()
        
        # Signal the consumer that we're done
        stop_event.set()
        logger.info("Producer thread finished")
        
    except Exception as e:
        logger.error(f"Error in producer thread: {e}")
        stop_event.set()  # Signal error condition

def get_localized_objects(logger):
    """
    Convert videos to frames using producer-consumer pattern
    """
    global RAW_DIR, PROCESSED_DIR, RESULTS_DIR

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    start_time = time.time()

    # Create and start threads
    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)
    
    producer_thread.start()
    consumer_thread.start()
    
    # Wait for both threads to complete
    producer_thread.join()
    consumer_thread.join()

    end_time = time.time()
    logger.info(f"Total time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    #download_videos(logger)
    get_localized_objects(logger)
