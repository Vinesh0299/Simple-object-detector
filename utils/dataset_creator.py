import os
import cv2
import random
import string
import numpy as np

def generate_random_section(length):
    """
    Generate a random section of text.

    Args:
        length (int): The length of the text to generate.

    Returns:
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length)).upper()

def create_required_directories(output_dir):
    """
    Create the required directories.
    """
    # Create train/test/val directories for images
    train_dir = os.path.join(output_dir, 'images', 'train')
    test_dir = os.path.join(output_dir, 'images', 'test')
    val_dir = os.path.join(output_dir, 'images', 'val')
    
    # Create train/test/val directories for labels
    train_labels_dir = os.path.join(output_dir, 'labels', 'train')
    test_labels_dir = os.path.join(output_dir, 'labels', 'test')
    val_labels_dir = os.path.join(output_dir, 'labels', 'val')
    
    # Create all directories
    for dir_path in [train_dir, test_dir, val_dir, 
                    train_labels_dir, test_labels_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)

    return train_dir, test_dir, val_dir, train_labels_dir, test_labels_dir, val_labels_dir

def get_random_position(width, height):
    """
    Get a random position for the text.
    """
    x = random.randint(10, width - 200)  # Increased margin for longer text
    y = random.randint(10, height - 30)
    return x, y

def convert_to_frames(video_path, output_dir, logger):
    """
    Convert a video file to a sequence of frames.

    Args:
        video_path (str): The path to the video file.
        output_dir (str): The directory to save the frames.
        logger (logging.Logger): The logger to use for logging.

    Returns:
        None
    """
    try:
        # Get the video name
        video_name = video_path.split("/")[-1].replace(".mp4", "")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create required directories
        logger.info(f"Creating required directories in {output_dir}")
        train_dir, test_dir, val_dir, train_labels_dir, test_labels_dir, val_labels_dir = create_required_directories(output_dir)

        # Initialize frame counter
        frame_index = 0
        frame_save_interval = 10

        # Read and save frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save every nth frame
            if frame_index % frame_save_interval == 0:
                # Resize frame to 640x640
                frame = cv2.resize(frame, (640, 640))
                
                # Update width and height for normalized calculations
                width = 640
                height = 640
                
                # Randomly decide whether to add text to this frame (50% chance)
                if random.random() < 0.5:
                    # Add random alphanumeric text to the frame
                    # Generate random text in format XXXX-XXXXXX-XXXX
                    
                    random_text = f"{generate_random_section(4)}-{generate_random_section(6)}-{generate_random_section(4)}"
                    logger.info(f"Generating random text: {random_text}")

                    # Random position for the text
                    x, y = get_random_position(width, height)
                    logger.info(f"Random position: ({x}, {y})")

                    # Add text to the frame
                    fonts = [
                        cv2.FONT_HERSHEY_SERIF,  # Times New Roman-like serif font
                        cv2.FONT_HERSHEY_COMPLEX,  # Similar to Times New Roman
                        cv2.FONT_HERSHEY_TRIPLEX,  # Bold serif font
                        cv2.FONT_HERSHEY_COMPLEX_SMALL  # Smaller serif font
                    ]
                    font = random.choice(fonts)
                    font_scale = random.uniform(0.5, 1.5)  # Random font size
                    font_thickness = random.randint(1, 3)  # Random thickness
                    font_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color
                    
                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(random_text, font, font_scale, font_thickness)
                    
                    # Calculate text bounding box
                    text_center_x = x + text_width // 2
                    text_center_y = y - text_height // 2
                    
                    # Ensure text stays within frame boundaries
                    margin = 20  # Add some margin from the edges
                    max_x = width - text_width - margin
                    max_y = height - text_height - margin
                    
                    # Adjust position if text would go out of bounds
                    x = min(max(margin, x), max_x)
                    y = min(max(margin + text_height, y), max_y)
                    
                    # Recalculate center after position adjustment
                    text_center_x = x + text_width // 2
                    text_center_y = y - text_height // 2
                    
                    # Create a blank image for the text
                    text_img = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(text_img, random_text, (x, y), font, font_scale, font_color, font_thickness)
                    
                    # Randomly decide whether to rotate the text (50% chance)
                    if random.random() < 0.5:
                        # Random rotation angle (-30 to 30 degrees)
                        angle = random.uniform(-30, 30)
                        
                        # Get rotation matrix
                        rotation_matrix = cv2.getRotationMatrix2D((text_center_x, text_center_y), angle, 1.0)
                        
                        # Apply rotation to text image
                        rotated_text = cv2.warpAffine(text_img, rotation_matrix, (width, height))
                        
                        # Combine original frame with rotated text
                        frame = cv2.addWeighted(frame, 1, rotated_text, 1, 0)
                        
                        # Calculate rotated bounding box corners
                        corners = np.array([
                            [x, y - text_height],
                            [x + text_width, y - text_height],
                            [x + text_width, y + baseline],
                            [x, y + baseline]
                        ])
                        
                        # Rotate corners
                        rotated_corners = cv2.transform(corners.reshape(-1, 1, 2), rotation_matrix).reshape(-1, 2)
                        
                        # Calculate new bounding box
                        min_x = np.min(rotated_corners[:, 0])
                        max_x = np.max(rotated_corners[:, 0])
                        min_y = np.min(rotated_corners[:, 1])
                        max_y = np.max(rotated_corners[:, 1])
                        
                        # Ensure rotated text stays within frame
                        if min_x < 0 or max_x > width or min_y < 0 or max_y > height:
                            # If text goes out of bounds, try again with a different position
                            continue
                    else:
                        # Add text without rotation
                        frame = cv2.addWeighted(frame, 1, text_img, 1, 0)
                        min_x = x
                        max_x = x + text_width
                        min_y = y - text_height
                        max_y = y + baseline
                    
                    # Draw bounding box
                    # box_color = (0, 255, 0)  # Green color
                    # box_thickness = 2
                    # cv2.rectangle(frame, 
                    #             (int(min_x), int(min_y)), 
                    #             (int(max_x), int(max_y)), 
                    #             box_color, 
                    #             box_thickness)

                    logger.info(f"Text: {random_text}, Center: ({text_center_x}, {text_center_y}), Size: ({text_width}, {text_height}), Position: ({x}, {y})")

                    # Determine which directory to save to based on frame index
                    if frame_index/10 % 10 < 8:  # 80% for training
                        logger.info(f"Saving to train directory")
                        save_dir = train_dir
                        labels_dir = train_labels_dir
                    elif frame_index/10 % 10 < 9:  # 10% for testing
                        logger.info(f"Saving to test directory")
                        save_dir = test_dir
                        labels_dir = test_labels_dir
                    else:  # 10% for validation
                        logger.info(f"Saving to val directory")
                        save_dir = val_dir
                        labels_dir = val_labels_dir

                    # Save the frame
                    frame_path = os.path.join(save_dir, f"{video_name}_frame_{frame_index:06d}.jpg")
                    cv2.imwrite(frame_path, frame)

                    # Create corresponding label file
                    label_path = os.path.join(labels_dir, f"{video_name}_frame_{frame_index:06d}.txt")
                    
                    # Calculate normalized coordinates
                    center_x_norm = text_center_x / width
                    center_y_norm = text_center_y / height
                    width_norm = (max_x - min_x) / width
                    height_norm = (max_y - min_y) / height
                    
                    # Write label file with format: class x_center y_center width height
                    with open(label_path, 'w') as f:
                        f.write(f"0 {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
                else:
                    # Save frame without text
                    if frame_index/10 % 10 < 8:  # 80% for training
                        logger.info(f"Saving to train directory")
                        save_dir = train_dir
                    elif frame_index/10 % 10 < 9:  # 10% for testing
                        logger.info(f"Saving to test directory")
                        save_dir = test_dir
                    else:  # 10% for validation
                        logger.info(f"Saving to val directory")
                        save_dir = val_dir
                        
                    frame_path = os.path.join(save_dir, f"{video_name}_frame_{frame_index:06d}.jpg")
                    cv2.imwrite(frame_path, frame)

            frame_index += 1
    except Exception as e:
        logger.error(f"Error converting video to frames: {e}")
