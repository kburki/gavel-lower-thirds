#!/usr/bin/env python3
"""
Lower Thirds Extractor for SNS EVO

This script:
1. Processes proxy video files from SNS EVO
2. Detects lower thirds graphics
3. Extracts text using OCR
4. Generates metadata with speaker names and timestamps
5. Integrates with EVO via Slingshot API

Requirements:
- Python 3.8+
- OpenCV
- Pytesseract (Tesseract OCR)
- Requests (for API calls)
- FFmpeg (for video processing)
"""

import os
import sys
import cv2
import json
import argparse
import subprocess
import time
import requests
import re
import logging
import sqlite3
from datetime import datetime, timedelta
import pytesseract
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lower_thirds_extractor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LowerThirdsExtractor:
    def __init__(self, config_path=None):
        # Load default configuration
        self.config = self._load_config(config_path)
        
        # Configure OCR engine
        pytesseract.pytesseract.tesseract_cmd = self.config.get('tesseract_path', r'/usr/local/bin/tesseract')
        
        # Set OCR configuration
        self.ocr_config = self.config.get('ocr_settings', {}).get('config', '--oem 1 --psm 6 -l eng --dpi 300')
        
        # For tracking lower thirds across frames
        self.current_lower_third = {
            'main_text': None,        # Top portion text (e.g., senator name, bill number)
            'context_text': None,     # Bottom portion text (session info and date/time)
            'type': None,             # Type of lower third (e.g., "Speaker", "Bill", "Next")
            'start_time': None,       # When this lower third appeared
            'last_seen_time': None,   # When this lower third was last detected
        }
        
        self.lower_thirds = []        # List of all detected lower thirds
        self.last_frame_had_graphic = False
        
        # Special text markers to identify types of lower thirds
        self.markers = self.config.get('markers', {})
        if not self.markers:
            # Default markers if not provided in config
            self.markers = {
                'speaker': ['Sen.', 'Rep.', 'Speaker', 'President', 'Chairman', 'Chair'],
                'bill': ['HB', 'SB', 'SCR', 'HCR', 'Bill', 'CS', 'Resolution', 'Amendment', 'Motion'],
                'event': ['Presentation', 'Pledge', 'Introduction', 'At Ease', 'Colors', 'Allegiance', 'Recess', 'Adjournment'],
                'voice': ['Voice of:'],
                'next': ['Next']
            }
        
        # Create debug directory if needed
        self.debug_mode = self.config.get('debug_mode', False)
        if self.debug_mode:
            os.makedirs('debug_frames', exist_ok=True)
        
        # Database connection
        self.db_connection = None
        if self.config.get('use_database', False):
            self._initialize_database()
    
    def _load_config(self, config_path):
        """Load configuration from a JSON file"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {str(e)}")
        
        # Load default config from default paths if available
        default_paths = [
            'config/config.json',
            os.path.join(os.path.dirname(__file__), 'config/config.json'),
            os.path.join(os.path.expanduser('~'), '.lower_thirds_extractor', 'config/config.json')
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        config = json.load(f)
                    logger.info(f"Loaded configuration from {path}")
                    return config
                except Exception as e:
                    logger.error(f"Error loading config from {path}: {str(e)}")
        
        # Return default configuration
        logger.warning("No configuration file found. Using default configuration.")
        return {
            'evo_address': 'http://192.168.1.66',
            'username': 'api',
            'password': 'pedicab7NEEDY!mould',
            'sampling_rate': 1,
            'min_text_confidence': 60,
            'debug_mode': False,
            'use_database': False,
            'database_path': 'lower_thirds.db'
        }
    
    def _initialize_database(self):
        """Initialize SQLite database for storing lower thirds data"""
        try:
            db_path = self.config.get('database_path', 'lower_thirds.db')
            self.db_connection = sqlite3.connect(db_path)
            cursor = self.db_connection.cursor()
            
            # Create tables if they don't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                path TEXT,
                processed_date TEXT,
                duration REAL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS lower_thirds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER,
                main_text TEXT,
                context_text TEXT,
                type TEXT,
                start_time REAL,
                end_time REAL,
                start_time_str TEXT,
                end_time_str TEXT,
                duration REAL,
                date TEXT,
                time TEXT,
                session TEXT,
                station TEXT,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lower_third_id INTEGER,
                tag TEXT,
                category TEXT,
                value TEXT,
                FOREIGN KEY (lower_third_id) REFERENCES lower_thirds (id)
            )
            ''')
            
            self.db_connection.commit()
            logger.info(f"Database initialized at {db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            self.db_connection = None
    
    def process_video(self, video_path):
        """Process a video file to extract lower thirds text"""
        logger.info(f"Processing video: {video_path}")
        
        # Reset state for new video
        self.current_lower_third = {
            'main_text': None,
            'context_text': None,
            'type': None,
            'start_time': None,
            'last_seen_time': None
        }
        self.lower_thirds = []
        
        # Get video details
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logger.error(f"Could not open video {video_path}")
            return None
            
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        logger.info(f"Video duration: {timedelta(seconds=duration)}")
        logger.info(f"Sampling every {self.config.get('sampling_rate', 1)} seconds")
        
        # Store video in database if enabled
        video_id = None
        if self.db_connection:
            cursor = self.db_connection.cursor()
            cursor.execute(
                "INSERT INTO videos (filename, path, processed_date, duration) VALUES (?, ?, ?, ?)",
                (os.path.basename(video_path), video_path, datetime.now().isoformat(), duration)
            )
            self.db_connection.commit()
            video_id = cursor.lastrowid
        
        # Calculate frames to sample based on sampling rate
        sample_interval = int(fps * self.config.get('sampling_rate', 1))
        if sample_interval < 1:
            sample_interval = 1
        
        frame_number = 0
        stable_frames_without_graphic = 0
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
                
            # Process only at specified intervals
            if frame_number % sample_interval == 0:
                current_time = frame_number / fps
                time_str = str(timedelta(seconds=current_time)).split('.')[0]
                
                # Process the frame for lower thirds
                has_graphic = self._process_frame(frame, current_time, time_str, frame_number)
                
                # If we've gone several frames without seeing a graphic, end the current segment
                if not has_graphic:
                    stable_frames_without_graphic += 1
                    if stable_frames_without_graphic >= 3 and self.current_lower_third['main_text'] is not None:
                        self._end_current_lower_third(current_time, video_id)
                else:
                    stable_frames_without_graphic = 0
                
            frame_number += 1
            
        # Close the current lower third segment if we have one
        if self.current_lower_third['main_text'] is not None:
            self._end_current_lower_third(duration, video_id)
            
        video.release()
        logger.info(f"Extracted {len(self.lower_thirds)} lower thirds segments")
        return self.lower_thirds
    
    def _process_frame(self, frame, current_time, time_str, frame_number):
        """Process a single frame to detect and extract lower thirds text"""
        h, w = frame.shape[:2]
        has_graphic = False
        
        # Define the regions we want to check for lower thirds
        regions_to_check = self.config.get('regions', {}).get('modern_widescreen', {})
        if not regions_to_check:
            # Default regions if not specified in config
            regions_to_check = [
                # Main lower third region - covers both the top and bottom portions
                {'name': 'full_lower_third', 'y1': int(h * 0.75), 'y2': h, 'x1': 0, 'x2': w},
                
                # Top portion of lower third - contains the main information
                {'name': 'top_bar', 'y1': int(h * 0.75), 'y2': int(h * 0.85), 'x1': 0, 'x2': w},
                
                # Bottom portion - contains session and date info
                {'name': 'bottom_bar', 'y1': int(h * 0.85), 'y2': h, 'x1': 0, 'x2': w}
            ]
        
        # Store OCR results for each region
        ocr_results = {}
        
        # Check each region
        for region in regions_to_check:
            # Extract the region
            y1, y2 = region['y1'], region['y2']
            x1, x2 = region['x1'], region['x2']
            region_img = frame[y1:y2, x1:x2]
            
            # Check if this region might contain a graphic
            if self._is_lower_third(region_img):
                has_graphic = True
                
                # Save debug image if enabled
                if self.debug_mode:
                    debug_path = f"debug_frames/frame_{frame_number}_{region['name']}.jpg"
                    cv2.imwrite(debug_path, region_img)
                
                # Preprocess the image for better OCR
                preprocessed = self._preprocess_for_ocr(region_img)
                
                # Run OCR
                text = pytesseract.image_to_string(preprocessed, config=self.ocr_config).strip()
                
                # Get confidence values
                ocr_data = pytesseract.image_to_data(preprocessed, config=self.ocr_config, output_type=pytesseract.Output.DICT)
                
                # Process OCR data to get high-confidence text
                region_text = self._process_ocr_data(ocr_data)
                
                # Store the results
                ocr_results[region['name']] = {
                    'text': text,
                    'processed_text': region_text,
                    'confidence': self._calculate_avg_confidence(ocr_data)
                }
        
        # If we found a graphic in any region, process the lower third information
        if has_graphic and 'top_bar' in ocr_results and 'bottom_bar' in ocr_results:
            top_text = ocr_results['top_bar']['processed_text']
            bottom_text = ocr_results['bottom_bar']['processed_text']
            
            # Only process if we have enough text
            if len(top_text) > 3:
                # Determine the type of lower third
                lower_third_type = self._determine_lower_third_type(top_text)
                
                # Check if this is a new lower third or continuation
                if (self.current_lower_third['main_text'] != top_text or 
                    self.current_lower_third['context_text'] != bottom_text):
                    
                    # If we had a previous lower third, end it
                    if self.current_lower_third['main_text'] is not None:
                        self._end_current_lower_third(current_time)
                    
                    # Start a new lower third segment
                    self.current_lower_third = {
                        'main_text': top_text,
                        'context_text': bottom_text,
                        'type': lower_third_type,
                        'start_time': current_time,
                        'last_seen_time': current_time
                    }
                    
                    logger.info(f"New lower third at {time_str}: [{lower_third_type}] {top_text}")
                else:
                    # Update the last seen time for this lower third
                    self.current_lower_third['last_seen_time'] = current_time
        
        return has_graphic
    
    def _preprocess_for_ocr(self, image):
        """Preprocess image to improve OCR accuracy for Gavel Alaska lower thirds"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold to handle varying lighting conditions
        # This works better for white text on blue backgrounds
        thresh = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        # Invert the image if it's predominantly dark
        # This helps with white text on dark backgrounds
        mean_val = cv2.mean(thresh)[0]
        if mean_val < 127:
            thresh = cv2.bitwise_not(thresh)
        
        # Apply slight blur to reduce noise
        blur = cv2.GaussianBlur(thresh, (3, 3), 0)
        
        # Dilate to connect broken character components
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(blur, kernel, iterations=1)
        
        return dilated
    
    def _process_ocr_data(self, ocr_data):
        """Process OCR data to get high-confidence text"""
        processed_text = []
        
        for i in range(len(ocr_data['text'])):
            if (ocr_data['conf'][i] > self.config['min_text_confidence'] and 
                ocr_data['text'][i].strip()):
                processed_text.append(ocr_data['text'][i])
        
        return ' '.join(processed_text)
    
    def _calculate_avg_confidence(self, ocr_data):
        """Calculate average confidence of OCR results"""
        confidences = [conf for conf in ocr_data['conf'] if conf > 0]
        if confidences:
            return sum(confidences) / len(confidences)
        return 0
    
    def _is_lower_third(self, image):
        """Detect if an image region contains a lower third graphic"""
        if image.size == 0:
            return False
            
        # Get detection parameters from config
        detection_config = self.config.get('lower_third_detection', {})
        lower_blue = np.array(detection_config.get('blue_hsv_lower', [90, 50, 50]))
        upper_blue = np.array(detection_config.get('blue_hsv_upper', [130, 255, 255]))
        blue_threshold = detection_config.get('blue_density_threshold', 0.15)
        edge_threshold = detection_config.get('edge_density_threshold', 0.1)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for blue regions
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_pixels = np.count_nonzero(blue_mask)
        
        # If we have enough blue pixels, it's likely a lower third
        blue_density = blue_pixels / (image.shape[0] * image.shape[1])
        if blue_density > blue_threshold:
            return True
        
        # As a backup, use edge detection for cases where color detection fails
        # This helps with older footage or different lower third styles
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        edge_density = edge_pixels / (image.shape[0] * image.shape[1])
        
        # If we have enough edges, it could be a lower third
        if edge_density > edge_threshold:
            # Additional check for horizontal lines which are common in chyrons
            horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                             threshold=100, 
                                             minLineLength=image.shape[1] * 0.3, 
                                             maxLineGap=20)
            
            if horizontal_lines is not None and len(horizontal_lines) > 0:
                return True
            
        return False
    
    def _determine_lower_third_type(self, text):
        """Determine the type of lower third based on its content"""
        text_lower = text.lower()
        
        # Use generalized detection logic based on markers
        
        # Check for senators or representatives
        if any(marker in text for marker in self.markers['speaker']):
            return 'Speaker'
        
        # Check for bill references (HB, SB, etc.)
        if any(marker in text for marker in self.markers['bill']):
            return 'Bill'
        
        # Check for "Voice of:" pattern
        if any(marker in text for marker in self.markers['voice']):
            return 'Voice'
        
        # Check for special events
        if any(marker in text for marker in self.markers['event']):
            # Special case for Presentation of Colors
            if 'Colors' in text and 'Allegiance' in text:
                return 'Presentation of Colors'
            # Special case for Introduction of Guests
            elif 'Introduction' in text and 'Guests' in text:
                return 'Introduction of Guests'
            return 'Event'
        
        # Check for "Next" indicator
        if any(marker in text for marker in self.markers['next']):
            return 'Next'
        
        # Handle Dr. or specific titles that don't match other categories
        if 'Dr.' in text or 'Executive Dir.' in text or 'Director' in text:
            return 'Expert'
            
        # Default to "Generic" for unclassified lower thirds
        return 'Generic'
    
    def _extract_date_time(self, text):
        """Extract date and time from the context text"""
        result = {}
        
        # Handle "KTOO Tue. 1/21/25 12:59 pm" format
        date_pattern = r'(\w{3}\.\s+\d{1,2}/\d{1,2}/\d{2})'
        time_pattern = r'(\d{1,2}:\d{2}\s*(?:am|pm))'
        station_pattern = r'(KTOO|AKLEG|ARCS)'
        
        date_match = re.search(date_pattern, text, re.IGNORECASE)
        time_match = re.search(time_pattern, text, re.IGNORECASE)
        station_match = re.search(station_pattern, text, re.IGNORECASE)
        
        if date_match:
            result['date'] = date_match.group(1)
        
        if time_match:
            result['time'] = time_match.group(1)
            
        if station_match:
            result['station'] = station_match.group(1)
            
        # Extract session information - use a more generic approach
        if 'Senate' in text and 'Floor' in text:
            result['session'] = 'Senate Floor'
        elif 'House' in text and 'Floor' in text:
            result['session'] = 'House Floor'
        elif 'Committee' in text:
            # Extract the committee name if it contains 'Committee'
            committee_pattern = r'([A-Za-z\s]+Committee)'
            committee_match = re.search(committee_pattern, text)
            if committee_match:
                result['session'] = committee_match.group(1).strip()
            else:
                result['session'] = 'Committee'
            
        return result
    
    def _end_current_lower_third(self, end_time, video_id=None):
        """End the current lower third segment and add to the list"""
        if self.current_lower_third['main_text'] is not None:
            # Extract date and time from context text if available
            date_time = self._extract_date_time(self.current_lower_third['context_text'])
            
            # Create the lower third entry
            lower_third = {
                'main_text': self.current_lower_third['main_text'],
                'context_text': self.current_lower_third['context_text'],
                'type': self.current_lower_third['type'],
                'start_time': self.current_lower_third['start_time'],
                'end_time': end_time,
                'start_time_str': str(timedelta(seconds=self.current_lower_third['start_time'])).split('.')[0],
                'end_time_str': str(timedelta(seconds=end_time)).split('.')[0],
                'duration': end_time - self.current_lower_third['start_time']
            }
            
            # Add extracted date/time/session information
            for key, value in date_time.items():
                lower_third[key] = value
            
            # Add to the list
            self.lower_thirds.append(lower_third)
            
            # If database is enabled, store in database
            if self.db_connection and video_id:
                self._store_lower_third_in_db(lower_third, video_id)
            
            # Reset current lower third
            self.current_lower_third = {
                'main_text': None,
                'context_text': None,
                'type': None,
                'start_time': None,
                'last_seen_time': None
            }
    
    def _store_lower_third_in_db(self, lower_third, video_id):
        """Store lower third information in the database"""
        try:
            cursor = self.db_connection.cursor()
            
            # Insert lower third
            cursor.execute('''
            INSERT INTO lower_thirds 
            (video_id, main_text, context_text, type, start_time, end_time, 
            start_time_str, end_time_str, duration, date, time, session, station)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                video_id,
                lower_third['main_text'],
                lower_third['context_text'],
                lower_third['type'],
                lower_third['start_time'],
                lower_third['end_time'],
                lower_third['start_time_str'],
                lower_third['end_time_str'],
                lower_third['duration'],
                lower_third.get('date'),
                lower_third.get('time'),
                lower_third.get('session'),
                lower_third.get('station')
            ))
            
            lower_third_id = cursor.lastrowid
            
            # Generate and store tags
            tags = self._generate_tags_for_lower_third(lower_third)
            for tag in tags:
                parts = tag.split(':', 1)
                if len(parts) == 2:
                    category, value = parts
                    cursor.execute('''
                    INSERT INTO tags (lower_third_id, tag, category, value)
                    VALUES (?, ?, ?, ?)
                    ''', (
                        lower_third_id,
                        tag,
                        category,
                        value
                    ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing lower third in database: {str(e)}")
    
    def _generate_tags_for_lower_third(self, lower_third):
        """Generate tags for a single lower third"""
        tags = []
        
        # Add type-specific tags
        if lower_third['type'] == 'Speaker':
            # Extract the name portion 
            name = lower_third['main_text']
            # Clean up the name
            name = re.sub(r'^(Sen\.|Rep\.)\s+', '', name) # Remove Sen./Rep. prefix
            
            # Extract district information if present
            district_match = re.search(r'([A-Z]\s*-\s*[A-Za-z\s]+)', name)
            if district_match:
                district = district_match.group(1).strip()
                # Clean up the name to remove district info
                name = re.sub(r'\([A-Z]\s*-\s*[A-Za-z\s]+\)', '', name).strip()
                tags.append(f"District:{district}")
            
            tags.append(f"Speaker:{name}")
            
        elif lower_third['type'] == 'Bill':
            # Extract bill number
            bill_match = re.search(r'(HB|SB|SCR|HCR|CS)\s+(\d+)', lower_third['main_text'])
            if bill_match:
                bill_num = f"{bill_match.group(1)} {bill_match.group(2)}"
                tags.append(f"Bill:{bill_num}")
                
        elif lower_third['type'] == 'Voice':
            # Extract the name from "Voice of: Name"
            voice_match = re.search(r'Voice of:\s+(.*)', lower_third['main_text'])
            if voice_match:
                voice_name = voice_match.group(1)
                tags.append(f"Voice:{voice_name}")
                
                # Look for title/organization in the same text
                org_match = re.search(r'([A-Za-z\s]+,\s+[A-Za-z\s\.]+)(marker in text for marker in self.markers['voice']):
            return 'Voice'
        
        # Check for special events
        if any(marker in text for marker in self.markers['event']):
            if 'Colors' in text and 'Allegiance' in text:
                return 'Presentation of Colors'
            elif 'Introduction' in text and 'Guests' in text:
                return 'Introduction of Guests'
            return 'Event'
        
        # Check for "Next" indicator
        if any(marker in text for marker in self.markers['next']):
            return 'Next'
        
        # Handle Dr. or specific titles that don't match other categories
        if 'Dr.' in text or 'Executive Dir.' in text:
            return 'Expert'
            
        # Default to "Generic" for unclassified lower thirds
        return 'Generic'
    
    def _end_current_lower_third(self, end_time):
        """End the current lower third segment and add to the list"""
        if self.current_lower_third['main_text'] is not None:
            # Extract date and time from context text if available
            date_time = self._extract_date_time(self.current_lower_third['context_text'])
            
            self.lower_thirds.append({
                'main_text': self.current_lower_third['main_text'],
                'context_text': self.current_lower_third['context_text'],
                'type': self.current_lower_third['type'],
                'date': date_time.get('date', None),
                'time': date_time.get('time', None),
                'start_time': self.current_lower_third['start_time'],
                'end_time': end_time,
                'start_time_str': str(timedelta(seconds=self.current_lower_third['start_time'])).split('.')[0],
                'end_time_str': str(timedelta(seconds=end_time)).split('.')[0],
                'duration': end_time - self.current_lower_third['start_time']
            })
            
            self.current_lower_third = {
                'main_text': None,
                'context_text': None,
                'type': None,
                'start_time': None,
                'last_seen_time': None
            }
    
    def _extract_date_time(self, text):
        """Extract date and time from the context text"""
        result = {}
        
        # Handle "KTOO Tue. 1/21/25 12:59 pm" format
        date_pattern = r'(\w{3}\.\s+\d{1,2}/\d{1,2}/\d{2})'
        time_pattern = r'(\d{1,2}:\d{2}\s*(?:am|pm))'
        station_pattern = r'(KTOO)'
        
        date_match = re.search(date_pattern, text, re.IGNORECASE)
        time_match = re.search(time_pattern, text, re.IGNORECASE)
        station_match = re.search(station_pattern, text, re.IGNORECASE)
        
        if date_match:
            result['date'] = date_match.group(1)
        
        if time_match:
            result['time'] = time_match.group(1)
            
        if station_match:
            result['station'] = station_match.group(1)
            
        # Extract session information
        if 'Senate Floor Session' in text:
            result['session'] = 'Senate Floor Session'
        elif 'House Education Committee' in text:
            result['session'] = 'House Education Committee'
        elif 'Committee' in text:
            # Extract the committee name if it contains 'Committee'
            committee_pattern = r'([A-Za-z\s]+Committee)'
            committee_match = re.search(committee_pattern, text)
            if committee_match:
                result['session'] = committee_match.group(1).strip()
            
        return result
    
    def save_metadata(self, output_path, video_path):
        """Save extracted metadata to a JSON file"""
        metadata = {
            'video': os.path.basename(video_path),
            'extraction_date': datetime.now().isoformat(),
            'lower_thirds': self.lower_thirds
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {output_path}")
        return metadata
    
    def generate_tags_for_evo(self):
        """Generate tags from extracted lower thirds for EVO system"""
        tags = []
        
        for lt in self.lower_thirds:
            # Add type-specific tags
            if lt['type'] == 'Speaker':
                # Extract the name portion 
                name = lt['main_text']
                # Clean up the name
                name = re.sub(r'^(Sen\.|Rep\.)\s+', '', name) # Remove Sen./Rep. prefix
                
                # Extract district information if present
                district_match = re.search(r'([A-Z]\s*-\s*[A-Za-z\s]+)', name)
                if district_match:
                    district = district_match.group(1).strip()
                    # Clean up the name to remove district info
                    name = re.sub(r'\([A-Z]\s*-\s*[A-Za-z\s]+\)', '', name).strip()
                    tags.append(f"District:{district}")
                
                tags.append(f"Speaker:{name}")
                
            elif lt['type'] in ['Bill', 'Education Bill']:
                # Extract bill number
                bill_match = re.search(r'(HB|SB|SCR|HCR)\s+(\d+)', lt['main_text'])
                if bill_match:
                    bill_num = f"{bill_match.group(1)} {bill_match.group(2)}"
                    tags.append(f"Bill:{bill_num}")
                    
                    # Add education tag if it's an education bill
                    if lt['type'] == 'Education Bill':
                        tags.append("Topic:Education")
                    
            elif lt['type'] == 'Voice':
                # Extract the name from "Voice of: Name"
                voice_match = re.search(r'Voice of:\s+(.*)', lt['main_text'])
                if voice_match:
                    voice_name = voice_match.group(1)
                    tags.append(f"Voice:{voice_name}")
                    
                    # Look for title/organization in the same text
                    org_match = re.search(r'([A-Za-z\s]+,\s+[A-Za-z\s\.]+)
    
    def upload_to_evo(self, metadata, video_path):
        """Upload extracted metadata to EVO via Slingshot API"""
        print("Uploading metadata to EVO...")
        
        try:
            # Generate EVO-compatible tags
            tags = self.generate_tags_for_evo()
            video_filename = os.path.basename(video_path)
            
            print(f"Generated {len(tags)} tags for {video_filename}")
            
            # Construct the Slingshot API request
            # Note: This API endpoint should be verified with your specific EVO configuration
            api_url = f"{self.config['evo_address']}/v1/sharebrowser/tag"
            auth = (self.config['username'], self.config['password'])
            
            # Prepare the API payload - adjust based on actual Slingshot API documentation
            payload = {
                "file_path": video_path,
                "tags": tags,
                "metadata": {
                    "lower_thirds": metadata['lower_thirds']
                }
            }
            
            # Make the API request
            # Uncomment this code when ready to implement the actual API call
            # response = requests.post(api_url, json=payload, auth=auth)
            # print(f"API Response: {response.status_code}")
            # return response.status_code == 200
            
            # For testing, just print the payload
            print(f"Would add the following tags to EVO for {video_filename}:")
            for tag in tags:
                print(f"  - {tag}")
            return True
            
        except Exception as e:
            print(f"Error uploading to EVO: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Extract lower thirds from video files')
    parser.add_argument('video_path', help='Path to the video file or directory of videos')
    parser.add_argument('--output', '-o', help='Output directory for metadata files')
    parser.add_argument('--config', '-c', help='Path to configuration JSON file')
    parser.add_argument('--upload', '-u', action='store_true', help='Upload metadata to EVO')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode with image saves')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return
    else:
        # Default config
        config = {
            'evo_address': 'http://your-evo-server',
            'username': 'your-username',
            'password': 'your-password',
            'sampling_rate': 1,
            'min_text_confidence': 60,
            'debug_mode': args.debug
        }
    
    extractor = LowerThirdsExtractor(config)
    
    # Set output directory
    output_dir = args.output or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process video(s)
    if os.path.isdir(args.video_path):
        # Process all videos in directory
        for filename in os.listdir(args.video_path):
            if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_path = os.path.join(args.video_path, filename)
                process_single_video(extractor, video_path, output_dir, args.upload)
    else:
        # Process single video
        process_single_video(extractor, args.video_path, output_dir, args.upload)


def process_single_video(extractor, video_path, output_dir, upload):
    """Process a single video file"""
    # Reset extractor state for new video
    extractor.current_lower_third = {
        'main_text': None,
        'context_text': None,
        'type': None,
        'start_time': None,
        'last_seen_time': None
    }
    extractor.lower_thirds = []
    
    # Process the video
    lower_thirds = extractor.process_video(video_path)
    
    if lower_thirds:
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        
        # Save metadata
        metadata = extractor.save_metadata(output_path, video_path)
        
        # Upload to EVO if requested
        if upload:
            extractor.upload_to_evo(metadata, video_path)
    else:
        print("No lower thirds data extracted from video")


if __name__ == "__main__":
    main()now().isoformat(),
            'speakers': self.speakers
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {output_path}")
        return metadata
    
    def upload_to_evo(self, metadata, video_path):
        """Upload extracted metadata to EVO via Slingshot API"""
        print("Uploading metadata to EVO...")
        
        # This is where you would implement the Slingshot API integration
        # The following is a placeholder for the API call
        
        video_filename = os.path.basename(video_path)
        
        # Generate tags for each speaker
        tags = []
        for speaker in self.speakers:
            tags.append(f"Speaker:{speaker['speaker']}")
            tags.append(f"Time:{speaker['start_time_str']}-{speaker['end_time_str']}")
        
        # Remove duplicates
        tags = list(set(tags))
        
        # Construct the API request
        api_url = f"{self.config['evo_address']}/v1/sharebrowser/tag"
        auth = (self.config['username'], self.config['password'])
        
        # Prepare the API payload - adjust based on actual Slingshot API documentation
        payload = {
            "file_path": video_path,
            "tags": tags,
            "metadata": {
                "speakers": metadata['speakers']
            }
        }
        
        try:
            # Make the API request
            # response = requests.post(api_url, json=payload, auth=auth)
            # Uncomment above and implement real API call when ready
            
            print(f"Would upload {len(tags)} tags to EVO for {video_filename}")
            print(f"Tags: {tags}")
            # For testing, just return success
            return True
            
        except Exception as e:
            print(f"Error uploading to EVO: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Extract lower thirds from video files')
    parser.add_argument('video_path', help='Path to the video file or directory of videos')
    parser.add_argument('--output', '-o', help='Output directory for metadata files')
    parser.add_argument('--config', '-c', help='Path to configuration JSON file')
    parser.add_argument('--upload', '-u', action='store_true', help='Upload metadata to EVO')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return
    
    extractor = LowerThirdsExtractor(config)
    
    # Set output directory
    output_dir = args.output or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process video(s)
    if os.path.isdir(args.video_path):
        # Process all videos in directory
        for filename in os.listdir(args.video_path):
            if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_path = os.path.join(args.video_path, filename)
                process_single_video(extractor, video_path, output_dir, args.upload)
    else:
        # Process single video
        process_single_video(extractor, args.video_path, output_dir, args.upload)


def process_single_video(extractor, video_path, output_dir, upload):
    """Process a single video file"""
    # Reset extractor state for new video
    extractor.current_speaker = None
    extractor.speaker_start_time = None
    extractor.speakers = []
    
    # Process the video
    speakers = extractor.process_video(video_path)
    
    if speakers:
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        
        # Save metadata
        metadata = extractor.save_metadata(output_path, video_path)
        
        # Upload to EVO if requested
        if upload:
            extractor.upload_to_evo(metadata, video_path)
    else:
        print("No speaker data extracted from video")


if __name__ == "__main__":
    main()
, voice_name)
                    if org_match:
                        name_parts = voice_name.split(',', 1)
                        if len(name_parts) > 1:
                            person = name_parts[0].strip()
                            org = name_parts[1].strip()
                            tags.append(f"Person:{person}")
                            tags.append(f"Organization:{org}")
                    
            elif lt['type'] == 'Presentation of Colors':
                tags.append("Event:Presentation of Colors")
                # Extract the group if present (e.g., Girl Scouts of Alaska)
                if ":" in lt['main_text']:
                    group = lt['main_text'].split(':', 1)[1].strip()
                    tags.append(f"Group:{group}")
                    
            elif lt['type'] == 'Introduction of Guests':
                tags.append("Event:Introduction of Guests")
                
            elif lt['type'] == 'Event':
                # Add event tag
                event_text = lt['main_text'].split(':')[0] if ':' in lt['main_text'] else lt['main_text']
                tags.append(f"Event:{event_text}")
                
            elif lt['type'] == 'Expert':
                # Extract name and title/organization
                if "," in lt['main_text']:
                    parts = lt['main_text'].split(',', 1)
                    name = parts[0].strip()
                    org = parts[1].strip() if len(parts) > 1 else ""
                    tags.append(f"Expert:{name}")
                    if org:
                        tags.append(f"Organization:{org}")
                else:
                    tags.append(f"Expert:{lt['main_text']}")
            
            # Extract session information if available
            if 'session' in lt and lt['session']:
                tags.append(f"Session:{lt['session']}")
                
            # Add timestamp tags for all lower thirds
            tags.append(f"Time:{lt['start_time_str']}")
            
            # Add date if available
            if 'date' in lt and lt['date']:
                # Clean up the date format
                clean_date = lt['date'].replace('.', '').strip()
                tags.append(f"Date:{clean_date}")
            
        # Remove duplicates while preserving order
        unique_tags = []
        for tag in tags:
            if tag not in unique_tags:
                unique_tags.append(tag)
                
        return unique_tags
    
    def upload_to_evo(self, metadata, video_path):
        """Upload extracted metadata to EVO via Slingshot API"""
        print("Uploading metadata to EVO...")
        
        try:
            # Generate EVO-compatible tags
            tags = self.generate_tags_for_evo()
            video_filename = os.path.basename(video_path)
            
            print(f"Generated {len(tags)} tags for {video_filename}")
            
            # Construct the Slingshot API request
            # Note: This API endpoint should be verified with your specific EVO configuration
            api_url = f"{self.config['evo_address']}/v1/sharebrowser/tag"
            auth = (self.config['username'], self.config['password'])
            
            # Prepare the API payload - adjust based on actual Slingshot API documentation
            payload = {
                "file_path": video_path,
                "tags": tags,
                "metadata": {
                    "lower_thirds": metadata['lower_thirds']
                }
            }
            
            # Make the API request
            # Uncomment this code when ready to implement the actual API call
            # response = requests.post(api_url, json=payload, auth=auth)
            # print(f"API Response: {response.status_code}")
            # return response.status_code == 200
            
            # For testing, just print the payload
            print(f"Would add the following tags to EVO for {video_filename}:")
            for tag in tags:
                print(f"  - {tag}")
            return True
            
        except Exception as e:
            print(f"Error uploading to EVO: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Extract lower thirds from video files')
    parser.add_argument('video_path', help='Path to the video file or directory of videos')
    parser.add_argument('--output', '-o', help='Output directory for metadata files')
    parser.add_argument('--config', '-c', help='Path to configuration JSON file')
    parser.add_argument('--upload', '-u', action='store_true', help='Upload metadata to EVO')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode with image saves')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return
    else:
        # Default config
        config = {
            'evo_address': 'http://your-evo-server',
            'username': 'your-username',
            'password': 'your-password',
            'sampling_rate': 1,
            'min_text_confidence': 60,
            'debug_mode': args.debug
        }
    
    extractor = LowerThirdsExtractor(config)
    
    # Set output directory
    output_dir = args.output or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process video(s)
    if os.path.isdir(args.video_path):
        # Process all videos in directory
        for filename in os.listdir(args.video_path):
            if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_path = os.path.join(args.video_path, filename)
                process_single_video(extractor, video_path, output_dir, args.upload)
    else:
        # Process single video
        process_single_video(extractor, args.video_path, output_dir, args.upload)


def process_single_video(extractor, video_path, output_dir, upload):
    """Process a single video file"""
    # Reset extractor state for new video
    extractor.current_lower_third = {
        'main_text': None,
        'context_text': None,
        'type': None,
        'start_time': None,
        'last_seen_time': None
    }
    extractor.lower_thirds = []
    
    # Process the video
    lower_thirds = extractor.process_video(video_path)
    
    if lower_thirds:
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        
        # Save metadata
        metadata = extractor.save_metadata(output_path, video_path)
        
        # Upload to EVO if requested
        if upload:
            extractor.upload_to_evo(metadata, video_path)
    else:
        print("No lower thirds data extracted from video")


if __name__ == "__main__":
    main()now().isoformat(),
            'speakers': self.speakers
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {output_path}")
        return metadata
    
    def upload_to_evo(self, metadata, video_path):
        """Upload extracted metadata to EVO via Slingshot API"""
        print("Uploading metadata to EVO...")
        
        # This is where you would implement the Slingshot API integration
        # The following is a placeholder for the API call
        
        video_filename = os.path.basename(video_path)
        
        # Generate tags for each speaker
        tags = []
        for speaker in self.speakers:
            tags.append(f"Speaker:{speaker['speaker']}")
            tags.append(f"Time:{speaker['start_time_str']}-{speaker['end_time_str']}")
        
        # Remove duplicates
        tags = list(set(tags))
        
        # Construct the API request
        api_url = f"{self.config['evo_address']}/v1/sharebrowser/tag"
        auth = (self.config['username'], self.config['password'])
        
        # Prepare the API payload - adjust based on actual Slingshot API documentation
        payload = {
            "file_path": video_path,
            "tags": tags,
            "metadata": {
                "speakers": metadata['speakers']
            }
        }
        
        try:
            # Make the API request
            # response = requests.post(api_url, json=payload, auth=auth)
            # Uncomment above and implement real API call when ready
            
            print(f"Would upload {len(tags)} tags to EVO for {video_filename}")
            print(f"Tags: {tags}")
            # For testing, just return success
            return True
            
        except Exception as e:
            print(f"Error uploading to EVO: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Extract lower thirds from video files')
    parser.add_argument('video_path', help='Path to the video file or directory of videos')
    parser.add_argument('--output', '-o', help='Output directory for metadata files')
    parser.add_argument('--config', '-c', help='Path to configuration JSON file')
    parser.add_argument('--upload', '-u', action='store_true', help='Upload metadata to EVO')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return
    
    extractor = LowerThirdsExtractor(config)
    
    # Set output directory
    output_dir = args.output or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process video(s)
    if os.path.isdir(args.video_path):
        # Process all videos in directory
        for filename in os.listdir(args.video_path):
            if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_path = os.path.join(args.video_path, filename)
                process_single_video(extractor, video_path, output_dir, args.upload)
    else:
        # Process single video
        process_single_video(extractor, args.video_path, output_dir, args.upload)


def process_single_video(extractor, video_path, output_dir, upload):
    """Process a single video file"""
    # Reset extractor state for new video
    extractor.current_speaker = None
    extractor.speaker_start_time = None
    extractor.speakers = []
    
    # Process the video
    speakers = extractor.process_video(video_path)
    
    if speakers:
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        
        # Save metadata
        metadata = extractor.save_metadata(output_path, video_path)
        
        # Upload to EVO if requested
        if upload:
            extractor.upload_to_evo(metadata, video_path)
    else:
        print("No speaker data extracted from video")


if __name__ == "__main__":
    main()
, voice_name)
                if org_match:
                    name_parts = voice_name.split(',', 1)
                    if len(name_parts) > 1:
                        person = name_parts[0].strip()
                        org = name_parts[1].strip()
                        tags.append(f"Person:{person}")
                        tags.append(f"Organization:{org}")
                
        elif lower_third['type'] == 'Presentation of Colors':
            tags.append("Event:Presentation of Colors")
            # Extract the group if present (e.g., Girl Scouts of Alaska)
            if ":" in lower_third['main_text']:
                group = lower_third['main_text'].split(':', 1)[1].strip()
                tags.append(f"Group:{group}")
                
        elif lower_third['type'] == 'Introduction of Guests':
            tags.append("Event:Introduction of Guests")
            
        elif lower_third['type'] == 'Event':
            # Add event tag
            event_text = lower_third['main_text'].split(':')[0] if ':' in lower_third['main_text'] else lower_third['main_text']
            tags.append(f"Event:{event_text}")
            
        elif lower_third['type'] == 'Expert':
            # Extract name and title/organization
            if "," in lower_third['main_text']:
                parts = lower_third['main_text'].split(',', 1)
                name = parts[0].strip()
                org = parts[1].strip() if len(parts) > 1 else ""
                tags.append(f"Expert:{name}")
                if org:
                    tags.append(f"Organization:{org}")
            else:
                tags.append(f"Expert:{lower_third['main_text']}")
        
        # Extract session information if available
        if 'session' in lower_third and lower_third['session']:
            tags.append(f"Session:{lower_third['session']}")
            
        # Add timestamp tags for all lower thirds
        tags.append(f"Time:{lower_third['start_time_str']}")
        
        # Add date if available
        if 'date' in lower_third and lower_third['date']:
            # Clean up the date format
            clean_date = lower_third['date'].replace('.', '').strip()
            tags.append(f"Date:{clean_date}")
        
        return tags
    
    def generate_tags_for_evo(self):
        """Generate all tags from extracted lower thirds for EVO system"""
        all_tags = []
        
        for lt in self.lower_thirds:
            # Get tags for this lower third
            tags = self._generate_tags_for_lower_third(lt)
            all_tags.extend(tags)
            
        # Remove duplicates while preserving order
        unique_tags = []
        for tag in all_tags:
            if tag not in unique_tags:
                unique_tags.append(tag)
                
        return unique_tags(marker in text for marker in self.markers['voice']):
            return 'Voice'
        
        # Check for special events
        if any(marker in text for marker in self.markers['event']):
            if 'Colors' in text and 'Allegiance' in text:
                return 'Presentation of Colors'
            elif 'Introduction' in text and 'Guests' in text:
                return 'Introduction of Guests'
            return 'Event'
        
        # Check for "Next" indicator
        if any(marker in text for marker in self.markers['next']):
            return 'Next'
        
        # Handle Dr. or specific titles that don't match other categories
        if 'Dr.' in text or 'Executive Dir.' in text:
            return 'Expert'
            
        # Default to "Generic" for unclassified lower thirds
        return 'Generic'
    
    def _end_current_lower_third(self, end_time):
        """End the current lower third segment and add to the list"""
        if self.current_lower_third['main_text'] is not None:
            # Extract date and time from context text if available
            date_time = self._extract_date_time(self.current_lower_third['context_text'])
            
            self.lower_thirds.append({
                'main_text': self.current_lower_third['main_text'],
                'context_text': self.current_lower_third['context_text'],
                'type': self.current_lower_third['type'],
                'date': date_time.get('date', None),
                'time': date_time.get('time', None),
                'start_time': self.current_lower_third['start_time'],
                'end_time': end_time,
                'start_time_str': str(timedelta(seconds=self.current_lower_third['start_time'])).split('.')[0],
                'end_time_str': str(timedelta(seconds=end_time)).split('.')[0],
                'duration': end_time - self.current_lower_third['start_time']
            })
            
            self.current_lower_third = {
                'main_text': None,
                'context_text': None,
                'type': None,
                'start_time': None,
                'last_seen_time': None
            }
    
    def _extract_date_time(self, text):
        """Extract date and time from the context text"""
        result = {}
        
        # Handle "KTOO Tue. 1/21/25 12:59 pm" format
        date_pattern = r'(\w{3}\.\s+\d{1,2}/\d{1,2}/\d{2})'
        time_pattern = r'(\d{1,2}:\d{2}\s*(?:am|pm))'
        station_pattern = r'(KTOO)'
        
        date_match = re.search(date_pattern, text, re.IGNORECASE)
        time_match = re.search(time_pattern, text, re.IGNORECASE)
        station_match = re.search(station_pattern, text, re.IGNORECASE)
        
        if date_match:
            result['date'] = date_match.group(1)
        
        if time_match:
            result['time'] = time_match.group(1)
            
        if station_match:
            result['station'] = station_match.group(1)
            
        # Extract session information
        if 'Senate Floor Session' in text:
            result['session'] = 'Senate Floor Session'
        elif 'House Education Committee' in text:
            result['session'] = 'House Education Committee'
        elif 'Committee' in text:
            # Extract the committee name if it contains 'Committee'
            committee_pattern = r'([A-Za-z\s]+Committee)'
            committee_match = re.search(committee_pattern, text)
            if committee_match:
                result['session'] = committee_match.group(1).strip()
            
        return result
    
    def save_metadata(self, output_path, video_path):
        """Save extracted metadata to a JSON file"""
        metadata = {
            'video': os.path.basename(video_path),
            'extraction_date': datetime.now().isoformat(),
            'lower_thirds': self.lower_thirds
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {output_path}")
        return metadata
    
    def generate_tags_for_evo(self):
        """Generate tags from extracted lower thirds for EVO system"""
        tags = []
        
        for lt in self.lower_thirds:
            # Add type-specific tags
            if lt['type'] == 'Speaker':
                # Extract the name portion 
                name = lt['main_text']
                # Clean up the name
                name = re.sub(r'^(Sen\.|Rep\.)\s+', '', name) # Remove Sen./Rep. prefix
                
                # Extract district information if present
                district_match = re.search(r'([A-Z]\s*-\s*[A-Za-z\s]+)', name)
                if district_match:
                    district = district_match.group(1).strip()
                    # Clean up the name to remove district info
                    name = re.sub(r'\([A-Z]\s*-\s*[A-Za-z\s]+\)', '', name).strip()
                    tags.append(f"District:{district}")
                
                tags.append(f"Speaker:{name}")
                
            elif lt['type'] in ['Bill', 'Education Bill']:
                # Extract bill number
                bill_match = re.search(r'(HB|SB|SCR|HCR)\s+(\d+)', lt['main_text'])
                if bill_match:
                    bill_num = f"{bill_match.group(1)} {bill_match.group(2)}"
                    tags.append(f"Bill:{bill_num}")
                    
                    # Add education tag if it's an education bill
                    if lt['type'] == 'Education Bill':
                        tags.append("Topic:Education")
                    
            elif lt['type'] == 'Voice':
                # Extract the name from "Voice of: Name"
                voice_match = re.search(r'Voice of:\s+(.*)', lt['main_text'])
                if voice_match:
                    voice_name = voice_match.group(1)
                    tags.append(f"Voice:{voice_name}")
                    
                    # Look for title/organization in the same text
                    org_match = re.search(r'([A-Za-z\s]+,\s+[A-Za-z\s\.]+)
    
    def upload_to_evo(self, metadata, video_path):
        """Upload extracted metadata to EVO via Slingshot API"""
        print("Uploading metadata to EVO...")
        
        try:
            # Generate EVO-compatible tags
            tags = self.generate_tags_for_evo()
            video_filename = os.path.basename(video_path)
            
            print(f"Generated {len(tags)} tags for {video_filename}")
            
            # Construct the Slingshot API request
            # Note: This API endpoint should be verified with your specific EVO configuration
            api_url = f"{self.config['evo_address']}/v1/sharebrowser/tag"
            auth = (self.config['username'], self.config['password'])
            
            # Prepare the API payload - adjust based on actual Slingshot API documentation
            payload = {
                "file_path": video_path,
                "tags": tags,
                "metadata": {
                    "lower_thirds": metadata['lower_thirds']
                }
            }
            
            # Make the API request
            # Uncomment this code when ready to implement the actual API call
            # response = requests.post(api_url, json=payload, auth=auth)
            # print(f"API Response: {response.status_code}")
            # return response.status_code == 200
            
            # For testing, just print the payload
            print(f"Would add the following tags to EVO for {video_filename}:")
            for tag in tags:
                print(f"  - {tag}")
            return True
            
        except Exception as e:
            print(f"Error uploading to EVO: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Extract lower thirds from video files')
    parser.add_argument('video_path', help='Path to the video file or directory of videos')
    parser.add_argument('--output', '-o', help='Output directory for metadata files')
    parser.add_argument('--config', '-c', help='Path to configuration JSON file')
    parser.add_argument('--upload', '-u', action='store_true', help='Upload metadata to EVO')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode with image saves')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return
    else:
        # Default config
        config = {
            'evo_address': 'http://your-evo-server',
            'username': 'your-username',
            'password': 'your-password',
            'sampling_rate': 1,
            'min_text_confidence': 60,
            'debug_mode': args.debug
        }
    
    extractor = LowerThirdsExtractor(config)
    
    # Set output directory
    output_dir = args.output or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process video(s)
    if os.path.isdir(args.video_path):
        # Process all videos in directory
        for filename in os.listdir(args.video_path):
            if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_path = os.path.join(args.video_path, filename)
                process_single_video(extractor, video_path, output_dir, args.upload)
    else:
        # Process single video
        process_single_video(extractor, args.video_path, output_dir, args.upload)


def process_single_video(extractor, video_path, output_dir, upload):
    """Process a single video file"""
    # Reset extractor state for new video
    extractor.current_lower_third = {
        'main_text': None,
        'context_text': None,
        'type': None,
        'start_time': None,
        'last_seen_time': None
    }
    extractor.lower_thirds = []
    
    # Process the video
    lower_thirds = extractor.process_video(video_path)
    
    if lower_thirds:
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        
        # Save metadata
        metadata = extractor.save_metadata(output_path, video_path)
        
        # Upload to EVO if requested
        if upload:
            extractor.upload_to_evo(metadata, video_path)
    else:
        print("No lower thirds data extracted from video")


if __name__ == "__main__":
    main()now().isoformat(),
            'speakers': self.speakers
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {output_path}")
        return metadata
    
    def upload_to_evo(self, metadata, video_path):
        """Upload extracted metadata to EVO via Slingshot API"""
        print("Uploading metadata to EVO...")
        
        # This is where you would implement the Slingshot API integration
        # The following is a placeholder for the API call
        
        video_filename = os.path.basename(video_path)
        
        # Generate tags for each speaker
        tags = []
        for speaker in self.speakers:
            tags.append(f"Speaker:{speaker['speaker']}")
            tags.append(f"Time:{speaker['start_time_str']}-{speaker['end_time_str']}")
        
        # Remove duplicates
        tags = list(set(tags))
        
        # Construct the API request
        api_url = f"{self.config['evo_address']}/v1/sharebrowser/tag"
        auth = (self.config['username'], self.config['password'])
        
        # Prepare the API payload - adjust based on actual Slingshot API documentation
        payload = {
            "file_path": video_path,
            "tags": tags,
            "metadata": {
                "speakers": metadata['speakers']
            }
        }
        
        try:
            # Make the API request
            # response = requests.post(api_url, json=payload, auth=auth)
            # Uncomment above and implement real API call when ready
            
            print(f"Would upload {len(tags)} tags to EVO for {video_filename}")
            print(f"Tags: {tags}")
            # For testing, just return success
            return True
            
        except Exception as e:
            print(f"Error uploading to EVO: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Extract lower thirds from video files')
    parser.add_argument('video_path', help='Path to the video file or directory of videos')
    parser.add_argument('--output', '-o', help='Output directory for metadata files')
    parser.add_argument('--config', '-c', help='Path to configuration JSON file')
    parser.add_argument('--upload', '-u', action='store_true', help='Upload metadata to EVO')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return
    
    extractor = LowerThirdsExtractor(config)
    
    # Set output directory
    output_dir = args.output or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process video(s)
    if os.path.isdir(args.video_path):
        # Process all videos in directory
        for filename in os.listdir(args.video_path):
            if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_path = os.path.join(args.video_path, filename)
                process_single_video(extractor, video_path, output_dir, args.upload)
    else:
        # Process single video
        process_single_video(extractor, args.video_path, output_dir, args.upload)


def process_single_video(extractor, video_path, output_dir, upload):
    """Process a single video file"""
    # Reset extractor state for new video
    extractor.current_speaker = None
    extractor.speaker_start_time = None
    extractor.speakers = []
    
    # Process the video
    speakers = extractor.process_video(video_path)
    
    if speakers:
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        
        # Save metadata
        metadata = extractor.save_metadata(output_path, video_path)
        
        # Upload to EVO if requested
        if upload:
            extractor.upload_to_evo(metadata, video_path)
    else:
        print("No speaker data extracted from video")


if __name__ == "__main__":
    main()
, voice_name)
                    if org_match:
                        name_parts = voice_name.split(',', 1)
                        if len(name_parts) > 1:
                            person = name_parts[0].strip()
                            org = name_parts[1].strip()
                            tags.append(f"Person:{person}")
                            tags.append(f"Organization:{org}")
                    
            elif lt['type'] == 'Presentation of Colors':
                tags.append("Event:Presentation of Colors")
                # Extract the group if present (e.g., Girl Scouts of Alaska)
                if ":" in lt['main_text']:
                    group = lt['main_text'].split(':', 1)[1].strip()
                    tags.append(f"Group:{group}")
                    
            elif lt['type'] == 'Introduction of Guests':
                tags.append("Event:Introduction of Guests")
                
            elif lt['type'] == 'Event':
                # Add event tag
                event_text = lt['main_text'].split(':')[0] if ':' in lt['main_text'] else lt['main_text']
                tags.append(f"Event:{event_text}")
                
            elif lt['type'] == 'Expert':
                # Extract name and title/organization
                if "," in lt['main_text']:
                    parts = lt['main_text'].split(',', 1)
                    name = parts[0].strip()
                    org = parts[1].strip() if len(parts) > 1 else ""
                    tags.append(f"Expert:{name}")
                    if org:
                        tags.append(f"Organization:{org}")
                else:
                    tags.append(f"Expert:{lt['main_text']}")
            
            # Extract session information if available
            if 'session' in lt and lt['session']:
                tags.append(f"Session:{lt['session']}")
                
            # Add timestamp tags for all lower thirds
            tags.append(f"Time:{lt['start_time_str']}")
            
            # Add date if available
            if 'date' in lt and lt['date']:
                # Clean up the date format
                clean_date = lt['date'].replace('.', '').strip()
                tags.append(f"Date:{clean_date}")
            
        # Remove duplicates while preserving order
        unique_tags = []
        for tag in tags:
            if tag not in unique_tags:
                unique_tags.append(tag)
                
        return unique_tags
    
    def upload_to_evo(self, metadata, video_path):
        """Upload extracted metadata to EVO via Slingshot API"""
        print("Uploading metadata to EVO...")
        
        try:
            # Generate EVO-compatible tags
            tags = self.generate_tags_for_evo()
            video_filename = os.path.basename(video_path)
            
            print(f"Generated {len(tags)} tags for {video_filename}")
            
            # Construct the Slingshot API request
            # Note: This API endpoint should be verified with your specific EVO configuration
            api_url = f"{self.config['evo_address']}/v1/sharebrowser/tag"
            auth = (self.config['username'], self.config['password'])
            
            # Prepare the API payload - adjust based on actual Slingshot API documentation
            payload = {
                "file_path": video_path,
                "tags": tags,
                "metadata": {
                    "lower_thirds": metadata['lower_thirds']
                }
            }
            
            # Make the API request
            # Uncomment this code when ready to implement the actual API call
            # response = requests.post(api_url, json=payload, auth=auth)
            # print(f"API Response: {response.status_code}")
            # return response.status_code == 200
            
            # For testing, just print the payload
            print(f"Would add the following tags to EVO for {video_filename}:")
            for tag in tags:
                print(f"  - {tag}")
            return True
            
        except Exception as e:
            print(f"Error uploading to EVO: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Extract lower thirds from video files')
    parser.add_argument('video_path', help='Path to the video file or directory of videos')
    parser.add_argument('--output', '-o', help='Output directory for metadata files')
    parser.add_argument('--config', '-c', help='Path to configuration JSON file')
    parser.add_argument('--upload', '-u', action='store_true', help='Upload metadata to EVO')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode with image saves')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return
    else:
        # Default config
        config = {
            'evo_address': 'http://your-evo-server',
            'username': 'your-username',
            'password': 'your-password',
            'sampling_rate': 1,
            'min_text_confidence': 60,
            'debug_mode': args.debug
        }
    
    extractor = LowerThirdsExtractor(config)
    
    # Set output directory
    output_dir = args.output or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process video(s)
    if os.path.isdir(args.video_path):
        # Process all videos in directory
        for filename in os.listdir(args.video_path):
            if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_path = os.path.join(args.video_path, filename)
                process_single_video(extractor, video_path, output_dir, args.upload)
    else:
        # Process single video
        process_single_video(extractor, args.video_path, output_dir, args.upload)


def process_single_video(extractor, video_path, output_dir, upload):
    """Process a single video file"""
    # Reset extractor state for new video
    extractor.current_lower_third = {
        'main_text': None,
        'context_text': None,
        'type': None,
        'start_time': None,
        'last_seen_time': None
    }
    extractor.lower_thirds = []
    
    # Process the video
    lower_thirds = extractor.process_video(video_path)
    
    if lower_thirds:
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        
        # Save metadata
        metadata = extractor.save_metadata(output_path, video_path)
        
        # Upload to EVO if requested
        if upload:
            extractor.upload_to_evo(metadata, video_path)
    else:
        print("No lower thirds data extracted from video")


if __name__ == "__main__":
    main()now().isoformat(),
            'speakers': self.speakers
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {output_path}")
        return metadata
    
    def upload_to_evo(self, metadata, video_path):
        """Upload extracted metadata to EVO via Slingshot API"""
        print("Uploading metadata to EVO...")
        
        # This is where you would implement the Slingshot API integration
        # The following is a placeholder for the API call
        
        video_filename = os.path.basename(video_path)
        
        # Generate tags for each speaker
        tags = []
        for speaker in self.speakers:
            tags.append(f"Speaker:{speaker['speaker']}")
            tags.append(f"Time:{speaker['start_time_str']}-{speaker['end_time_str']}")
        
        # Remove duplicates
        tags = list(set(tags))
        
        # Construct the API request
        api_url = f"{self.config['evo_address']}/v1/sharebrowser/tag"
        auth = (self.config['username'], self.config['password'])
        
        # Prepare the API payload - adjust based on actual Slingshot API documentation
        payload = {
            "file_path": video_path,
            "tags": tags,
            "metadata": {
                "speakers": metadata['speakers']
            }
        }
        
        try:
            # Make the API request
            # response = requests.post(api_url, json=payload, auth=auth)
            # Uncomment above and implement real API call when ready
            
            print(f"Would upload {len(tags)} tags to EVO for {video_filename}")
            print(f"Tags: {tags}")
            # For testing, just return success
            return True
            
        except Exception as e:
            print(f"Error uploading to EVO: {str(e)}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Extract lower thirds from video files')
    parser.add_argument('video_path', help='Path to the video file or directory of videos')
    parser.add_argument('--output', '-o', help='Output directory for metadata files')
    parser.add_argument('--config', '-c', help='Path to configuration JSON file')
    parser.add_argument('--upload', '-u', action='store_true', help='Upload metadata to EVO')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return
    
    extractor = LowerThirdsExtractor(config)
    
    # Set output directory
    output_dir = args.output or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process video(s)
    if os.path.isdir(args.video_path):
        # Process all videos in directory
        for filename in os.listdir(args.video_path):
            if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_path = os.path.join(args.video_path, filename)
                process_single_video(extractor, video_path, output_dir, args.upload)
    else:
        # Process single video
        process_single_video(extractor, args.video_path, output_dir, args.upload)


def process_single_video(extractor, video_path, output_dir, upload):
    """Process a single video file"""
    # Reset extractor state for new video
    extractor.current_speaker = None
    extractor.speaker_start_time = None
    extractor.speakers = []
    
    # Process the video
    speakers = extractor.process_video(video_path)
    
    if speakers:
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        
        # Save metadata
        metadata = extractor.save_metadata(output_path, video_path)
        
        # Upload to EVO if requested
        if upload:
            extractor.upload_to_evo(metadata, video_path)
    else:
        print("No speaker data extracted from video")


if __name__ == "__main__":
    main()