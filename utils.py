#!/usr/bin/env python3
"""
Utility functions for the Gavel Lower Thirds Extractor
"""

import os
import sys
import logging
import json
import csv
import cv2
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def setup_logging(debug=False, log_file="lower_thirds_extractor.log"):
    """
    Set up logging configuration
    
    Args:
        debug (bool): Whether to enable debug logging
        log_file (str): Path to the log file
        
    Returns:
        logging.Logger: Configured logger
    """
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create a formatter with timestamp
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure the file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Configure the console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create and return the module logger
    logger = logging.getLogger("gavel_lower_thirds")
    
    return logger

def format_timecode(seconds):
    """
    Format seconds as HH:MM:SS timecode
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted timecode
    """
    return str(timedelta(seconds=seconds)).split('.')[0]

def save_json(data, output_path, video_path=None):
    """
    Save data to a JSON file
    
    Args:
        data (dict or list): Data to save
        output_path (str): Path to save the JSON file
        video_path (str, optional): Path to the source video
        
    Returns:
        str: Path to the saved file
    """
    # Create metadata wrapper if data is a list
    if isinstance(data, list):
        metadata = {
            'video': os.path.basename(video_path) if video_path else None,
            'extraction_date': datetime.now().isoformat(),
            'lower_thirds': data
        }
    else:
        metadata = data
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    return output_path

def save_csv(lower_thirds, output_path):
    """
    Save lower thirds data to a CSV file
    
    Args:
        lower_thirds (list): List of lower third dictionaries
        output_path (str): Path to save the CSV file
        
    Returns:
        str: Path to the saved file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Type', 'Main Text', 'Context Text', 
            'Start Time', 'End Time', 'Duration (sec)',
            'Date', 'Time', 'Session', 'Station'
        ])
        
        # Write data
        for lt in lower_thirds:
            writer.writerow([
                lt.get('type', ''),
                lt.get('main_text', ''),
                lt.get('context_text', ''),
                lt.get('start_time_str', ''),
                lt.get('end_time_str', ''),
                lt.get('duration', ''),
                lt.get('date', ''),
                lt.get('time', ''),
                lt.get('session', ''),
                lt.get('station', '')
            ])
            
    return output_path

def get_video_info(video_path):
    """
    Get basic information about a video file
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: Video information
    """
    # Open the video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return None
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    
    video.release()
    
    return {
        'path': video_path,
        'filename': os.path.basename(video_path),
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration,
        'duration_str': format_timecode(duration)
    }

def extract_frame(video_path, timestamp, output_path=None):
    """
    Extract a single frame from a video at the specified timestamp
    
    Args:
        video_path (str): Path to the video file
        timestamp (float): Timestamp in seconds
        output_path (str, optional): Path to save the frame
        
    Returns:
        numpy.ndarray: Extracted frame or None if failed
    """
    # Open the video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return None
    
    # Set position
    video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    
    # Read frame
    ret, frame = video.read()
    video.release()
    
    if not ret:
        return None
    
    # Save if requested
    if output_path and ret:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame)
    
    return frame

def extract_lower_third_region(frame):
    """
    Extract just the lower third region from a full video frame
    
    Args:
        frame (numpy.ndarray): Video frame
        
    Returns:
        numpy.ndarray: Lower third region
    """
    if frame is None:
        return None
        
    h, w = frame.shape[:2]
    lower_region = frame[int(h * 0.75):h, 0:w]
    return lower_region

def save_debug_image(image, name, output_dir="debug_frames"):
    """
    Save an image for debugging purposes
    
    Args:
        image (numpy.ndarray): Image to save
        name (str): Filename
        output_dir (str): Output directory
        
    Returns:
        str: Path to the saved image
    """
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate path
    path = os.path.join(output_dir, name)
    
    # Save image
    cv2.imwrite(path, image)
    
    return path

def generate_output_filename(video_path, output_type, output_dir):
    """
    Generate an output filename based on the video path
    
    Args:
        video_path (str): Path to the video file
        output_type (str): Type of output (json, csv, etc.)
        output_dir (str): Output directory
        
    Returns:
        str: Path to the output file
    """
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Generate output path
    if output_type == 'json':
        return os.path.join(output_dir, f"{base_name}_lower_thirds.json")
    elif output_type == 'csv':
        return os.path.join(output_dir, f"{base_name}_lower_thirds.csv")
    else:
        return os.path.join(output_dir, f"{base_name}_{output_type}")
