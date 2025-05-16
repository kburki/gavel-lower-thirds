#!/usr/bin/env python3
"""
Lower Thirds Detector Module

Handles the core functionality of detecting and tracking lower thirds in videos.
Compatible with the existing Gavel Lower Thirds project configuration.
"""

import cv2
import numpy as np
import pytesseract
import re
import os
import logging
from datetime import timedelta

class LowerThirdsDetector:
    def __init__(self, config=None):
        """
        Initialize the lower thirds detector with configuration
        
        Args:
            config (dict): Configuration dictionary loaded from config.json
        """
        self.logger = logging.getLogger("lower_thirds_detector")
        
        # Set defaults
        self.debug_mode = False
        self.sampling_rate = 1.0
        self.tesseract_path = r'/usr/bin/tesseract'
        self.ocr_config = '--oem 1 --psm 6 -l eng --dpi 300'
        
        # Load configuration if provided
        if config:
            self._load_from_config(config)
            
        # Set Tesseract path
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            
        # For tracking lower thirds across frames
        self.current_lower_third = {
            'main_text': None,        # Top portion text (e.g., senator name, bill number)
            'context_text': None,     # Bottom portion text (session info and date/time)
            'type': None,             # Type of lower third (e.g., "Speaker", "Bill", "Next")
            'start_time': None,       # When this lower third appeared
            'last_seen_time': None,   # When this lower third was last detected
            'speaker_info': {},       # Additional speaker information (name, title, district, etc.)
            'voice_of': None,         # Track the current "Voice of" person
            'bill_info': {}           # Additional bill information (number, title, etc.)
        }
        
        self.lower_thirds = []        # List of all detected lower thirds
        
        # Create debug directory if needed
        if self.debug_mode:
            os.makedirs('debug_frames', exist_ok=True)
    
    def _load_from_config(self, config):
        """Load detector settings from configuration dictionary"""
        # Debug mode
        self.debug_mode = config.get('debug_mode', False)
        
        # Sampling rate
        self.sampling_rate = config.get('video_settings', {}).get('sampling_rate', 1.0)
        
        # OCR configuration
        self.tesseract_path = config.get('video_settings', {}).get('tesseract_path', self.tesseract_path)
        self.ocr_config = config.get('ocr_settings', {}).get('config', self.ocr_config)
        self.min_text_confidence = config.get('video_settings', {}).get('min_text_confidence', 60)
        
        # Detection parameters
        self.detection_params = config.get('lower_third_detection', {})
        
        # Video regions to check
        self.regions = config.get('regions', {}).get('modern_widescreen', [])
        
        # Text markers
        self.markers = config.get('markers', {})
        if not self.markers:
            # Default markers if not provided in config
            self.markers = {
                'speaker': ['Sen.', 'Rep.', 'Speaker', 'President', 'Chairman', 'Chair'],
                'bill': ['HB', 'SB', 'SCR', 'HCR', 'Bill', 'CS', 'Resolution', 'Amendment', 'Motion'],
                'event': ['Presentation', 'Pledge', 'Introduction', 'At Ease', 'Colors', 'Allegiance', 'Recess', 'Adjournment'],
                'voice': ['Voice of:'],
                'next': ['Next']
            }
    
    def process_video(self, video_path):
        """Process a video file to extract lower thirds text"""
        self.logger.info(f"Processing video: {video_path}")
        
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
            self.logger.error(f"Could not open video {video_path}")
            return None
            
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        self.logger.info(f"Video duration: {timedelta(seconds=duration)}")
        self.logger.info(f"Sampling every {self.sampling_rate} seconds")
        
        # Calculate frames to sample based on sampling rate
        sample_interval = int(fps * self.sampling_rate)
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
                        self._end_current_lower_third(current_time)
                else:
                    stable_frames_without_graphic = 0
                
            frame_number += 1
                
        # Close the current lower third segment if we have one
        if self.current_lower_third['main_text'] is not None:
            self._end_current_lower_third(duration)
            
        video.release()
        self.logger.info(f"Extracted {len(self.lower_thirds)} lower thirds segments")
        return self.lower_thirds
    
    def _process_frame(self, frame, current_time, time_str, frame_number):
        """Process a single frame to detect and extract lower thirds text"""
        h, w = frame.shape[:2]
        has_graphic = False
    
        # If regions not specified, use defaults
        if not self.regions:
            self.regions = [
                # Main lower third region - covers both the top and bottom portions
                {'name': 'full_lower_third', 'y1': 0.75, 'y2': 1.0, 'x1': 0, 'x2': 1.0},
            
                # Top portion of lower third - contains the main information
                {'name': 'top_bar', 'y1': 0.75, 'y2': 0.85, 'x1': 0, 'x2': 1.0},
            
                # Bottom portion - contains session and date info
                {'name': 'bottom_bar', 'y1': 0.85, 'y2': 1.0, 'x1': 0, 'x2': 1.0}
            ]
    
        # Store OCR results for each region
        ocr_results = {}
    
        # Check each region
        for region in self.regions:
            # Extract the region - ensuring we have integer coordinates
            y1 = int(region['y1'] * h if isinstance(region['y1'], float) else region['y1'])
            y2 = int(region['y2'] * h if isinstance(region['y2'], float) else region['y2'])
            x1 = int(region['x1'] * w if isinstance(region['x1'], float) else region['x1'])
            x2 = int(region['x2'] * w if isinstance(region['x2'], float) else region['x2'])
        
            region_img = frame[y1:y2, x1:x2]
        
            # Check if this region might contain a graphic
            if self._is_lower_third(region_img, frame_number):
                has_graphic = True
            
                # Save debug image if enabled
                if self.debug_mode and frame_number % 100 == 0:
                    debug_path = f"debug_frames/frame_{frame_number}_{region['name']}.jpg"
                    cv2.imwrite(debug_path, region_img)
            
                # Preprocess the image for better OCR
                preprocessed = self._preprocess_for_ocr(region_img)
            
                # Run OCR
                text = pytesseract.image_to_string(preprocessed, config=self.ocr_config).strip()
            
                # Get confidence values
                ocr_data = pytesseract.image_to_data(preprocessed, config=self.ocr_config, output_type=pytesseract.Output.DICT)
            
                # Calculate average confidence
                confidences = [conf for conf in ocr_data['conf'] if conf > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
                # Store the results
                ocr_results[region['name']] = {
                    'text': text,
                    'confidence': avg_confidence
                }
            
                # Debug OCR results
                if self.debug_mode and frame_number % 100 == 0:
                    self.logger.debug(f"OCR result for {region['name']}: {text}")
                    self.logger.debug(f"Confidence: {avg_confidence}")
    
        # If we found a graphic in any region, process the lower third information
        if has_graphic and 'top_bar' in ocr_results and 'bottom_bar' in ocr_results:
            top_text = ocr_results['top_bar']['text']
            bottom_text = ocr_results['bottom_bar']['text']
        
            # Only process if we have enough text and confidence is above threshold
            if len(top_text) > 3 and ocr_results['top_bar']['confidence'] >= self.min_text_confidence:
                # Determine the type of lower third
                current_type = self.current_lower_third['type'] if self.current_lower_third['main_text'] is not None else None
                lower_third_type, attributes = self._determine_lower_third_type(top_text, current_type)
            
                # Special handling for Voice overs - if the content has only slightly changed but it's still
                # a voice-over, we want to maintain the same lower third rather than create a new one
                is_continuing_voice = (
                    lower_third_type == 'Voice' and 
                    self.current_lower_third['type'] == 'Voice' and
                    'Voice of:' in top_text and 'Voice of:' in self.current_lower_third['main_text'] and
                    self._text_similarity(self.current_lower_third['main_text'], top_text) > 0.7
                )
            
                # Check if this is a new lower third or continuation
                if (self.current_lower_third['main_text'] != top_text or 
                    self.current_lower_third['context_text'] != bottom_text) and not is_continuing_voice:
                
                    # If we had a previous lower third, end it
                    if self.current_lower_third['main_text'] is not None:
                        self._end_current_lower_third(current_time)
                
                    # Start a new lower third segment
                    self.current_lower_third = {
                        'main_text': top_text,
                        'context_text': bottom_text,
                        'type': lower_third_type,
                        'attributes': attributes,
                        'start_time': current_time,
                        'last_seen_time': current_time
                    }
                
                    self.logger.info(f"New lower third at {time_str}: [{lower_third_type}] {top_text}")
                else:
                    # Update the last seen time for this lower third
                    self.current_lower_third['last_seen_time'] = current_time
    
        return has_graphic
    
    def _text_similarity(self, text1, text2):
        """
        Calculate the similarity between two text strings using token-based comparison
    
        Args:
            text1 (str): First text string
            text2 (str): Second text string
        
        Returns:
            float: Similarity score between 0 and 1
        """
        # Convert to lowercase and split into tokens
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
    
        # Calculate Jaccard similarity
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
    
        return intersection / union
        
    def _extract_bill_info(self, text):
        """Extract bill information from text"""
        info = {
            'number': '',
            'title': ''
        }
        
        # Extract bill number (e.g., "SB 64")
        bill_pattern = r'(HB|SB|SCR|HCR|CS)\s*(\d+)'
        bill_match = re.search(bill_pattern, text)
        if bill_match:
            info['number'] = f"{bill_match.group(1)} {bill_match.group(2)}"
            
            # Try to extract title - usually after a dash or hyphen
            title_pattern = r'(?:HB|SB|SCR|HCR|CS)\s*\d+\s*[-–]\s*([^\n]+)'
            title_match = re.search(title_pattern, text)
            if title_match:
                info['title'] = title_match.group(1).strip()
        
        return info
    
    def _preprocess_for_ocr(self, image):
        """Preprocess image to improve OCR accuracy for lower thirds"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Invert the image since we have white text on dark background
        inverted = cv2.bitwise_not(gray)
    
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            inverted, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
    
        # Apply slight blur to reduce noise
        blur = cv2.GaussianBlur(thresh, (3, 3), 0)
    
        # Dilate to connect broken character components
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(blur, kernel, iterations=1)
    
        # Save preprocessed image if in debug mode
        if self.debug_mode:
            import time
            debug_path = f"debug_frames/preprocessed_{int(time.time() * 1000) % 10000}.jpg"
            cv2.imwrite(debug_path, dilated)
    
        return dilated
    
    def _is_lower_third(self, image, frame_number=0):
        """Detect if an image region contains a lower third graphic"""
        if image.size == 0:
            return False
        
        # Get detection parameters from config
        lower_blue = np.array(self.detection_params.get('blue_hsv_lower', [90, 50, 50]))
        upper_blue = np.array(self.detection_params.get('blue_hsv_upper', [130, 255, 255]))
        blue_threshold = self.detection_params.get('blue_density_threshold', 0.15)
        edge_threshold = self.detection_params.get('edge_density_threshold', 0.1)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
        # Create mask for blue regions
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_pixels = np.count_nonzero(blue_mask)
    
        # Calculate blue density
        blue_density = blue_pixels / (image.shape[0] * image.shape[1])
    
        # Log detection values in debug mode
        if self.debug_mode and frame_number % 100 == 0:
            self.logger.debug(f"Blue density: {blue_density:.4f}, threshold: {blue_threshold:.4f}")
            # Save the mask for debugging
            debug_path = f"debug_frames/blue_mask_{frame_number}.jpg"
            cv2.imwrite(debug_path, blue_mask)
    
        # If we have enough blue pixels, it's likely a lower third
        if blue_density > blue_threshold:
            return True
        
        # Backup detection using edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        edge_density = edge_pixels / (image.shape[0] * image.shape[1])
        
        # If we have enough edges, check for horizontal lines
        if edge_density > edge_threshold:
            horizontal_lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, 
                threshold=100, 
                minLineLength=image.shape[1] * 0.3, 
                maxLineGap=20
            )
            
            if horizontal_lines is not None and len(horizontal_lines) > 0:
                return True
            
        return False
    
    def _determine_lower_third_type(self, text, current_type=None):
        """
        Determine the type of lower third based on its content.
    
        Args:
            text (str): The text to analyze
            current_type (str, optional): The current type if we're updating an existing lower third
        
        Returns:
            str: The detected type
            dict: Additional attributes extracted (party, district, role, etc.)
        """
        # Initialize additional attributes dictionary
        attributes = {}
    
        # Extract district and party information
        district_match = re.search(r'([DRI])\s*-\s*([A-Za-z]+)\s*\(?(?:Dist\.|District)?\.?\s*(\d+)?\)?', text)
        if district_match:
            party = district_match.group(1)
            if party == 'D':
                attributes['party'] = 'Democrat'
            elif party == 'R':
                attributes['party'] = 'Republican'
            elif party == 'I':
                attributes['party'] = 'Independent'
            
            attributes['location'] = district_match.group(2)
            if district_match.group(3):
                attributes['district'] = district_match.group(3)
    
        # Check for committee roles
        if re.search(r'(?:Chair|Co-Chair|Chairman|Chairwoman|Chairperson)', text, re.IGNORECASE):
            attributes['role'] = 'Chair'
            if 'Co-' in text or 'Co ' in text:
                attributes['role'] = 'Co-Chair'
            
        # Special case: if this is a "Voice of" type but the current type is already Voice,
        # keep the existing type to avoid creating new segments for continued voice-overs
        if current_type == 'Voice' and 'Voice of:' in text:
            return 'Voice', attributes
            
        # Determine the type based on content markers
    
        # Check for voice overs
        if 'Voice of:' in text:
            # Extract the name from "Voice of: Name"
            voice_match = re.search(r'Voice of:\s*(.*)', text)
            if voice_match:
                attributes['speaker_name'] = voice_match.group(1).strip()
            return 'Voice', attributes
    
        # Check for senators or representatives
        if any(marker in text for marker in self.markers.get('speaker', [])):
            # Try to extract the name
            name_match = re.search(r'(?:Sen\.|Rep\.)\s+([A-Za-z\s\.]+)', text)
            if name_match:
                attributes['speaker_name'] = name_match.group(1).strip()
            return 'Speaker', attributes
    
        # Check for bill references with more comprehensive patterns
        if any(marker in text for marker in self.markers.get('bill', [])):
            # Extract bill number and title
            bill_match = re.search(r'(HB|SB|SCR|HCR|CS)\s*(\d+)(?:\s*-\s*(.+))?', text)
            if bill_match:
                attributes['bill_type'] = bill_match.group(1)
                attributes['bill_number'] = bill_match.group(2)
                if bill_match.group(3):
                    attributes['bill_title'] = bill_match.group(3)
            return 'Bill', attributes
    
        # Check for legislature information
        if re.search(r'Legislature|Legislative Day|Regular Session', text, re.IGNORECASE):
            return 'Legislature', attributes
    
        # Check for location information
        if re.search(r'Alaska State Capitol|Juneau', text, re.IGNORECASE):
            return 'Location', attributes
    
        # Check for special events
        if any(marker in text for marker in self.markers.get('event', [])):
            # Special case for Presentation of Colors
            if 'Colors' in text and 'Allegiance' in text:
                return 'Presentation of Colors', attributes
            # Special case for Introduction of Guests
            elif 'Introduction' in text and 'Guests' in text:
                return 'Introduction of Guests', attributes
            return 'Event', attributes
    
        # Check for "Next" indicator
        if any(marker in text for marker in self.markers.get('next', [])):
            return 'Next', attributes
    
        # Handle Dr. or specific titles that don't match other categories
        if 'Dr.' in text or 'Executive Dir.' in text or 'Director' in text:
            return 'Expert', attributes
            
        # Default to "Generic" for unclassified lower thirds
        return 'Generic', attributes
        
    def extract_speaker_info(self, text):
        """Extract speaker information from the text"""
        info = {
            'name': '',
            'title': '',
            'district': '',
            'party': ''
        }
        
        # Extract district information - pattern like "D - Anchorage (Dist. 13)"
        district_pattern = r'([DRI])\s*-\s*([A-Za-z\s]+)(?:\s*\((?:Dist\.|District)?\s*(\d+)\))?'
        district_match = re.search(district_pattern, text)
        if district_match:
            info['party'] = district_match.group(1)
            info['district'] = district_match.group(3) if district_match.group(3) else ''
            info['location'] = district_match.group(2).strip() if district_match.group(2) else ''
        
        # Extract name - typically "Rep. FirstName LastName" or similar
        name_pattern = r'(Sen\.|Rep\.|Speaker|President|Chairman|Chair)\s+([A-Za-z\s\.]+?)(?:\s+[DRI]\s*-|$|\n)'
        name_match = re.search(name_pattern, text)
        if name_match:
            info['title'] = name_match.group(1)
            info['name'] = name_match.group(2).strip()
        
        return info
    
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
            
        # Extract session information
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
    
    def _end_current_lower_third(self, end_time):
        """End the current lower third segment and add to the list"""
        if self.current_lower_third['main_text'] is not None:
            # Extract date and time from context text if available
            date_time = self._extract_date_time(self.current_lower_third['context_text'] or '')
            
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
            
            # Reset current lower third
            self.current_lower_third = {
                'main_text': None,
                'context_text': None,
                'type': None,
                'start_time': None,
                'last_seen_time': None
            }
            
            return lower_third
        
        return None
    
    def generate_tags(self):
        """Generate tag data from extracted lower thirds"""
        all_tags = []
        
        for lt in self.lower_thirds:
            # Generate tags for this lower third
            lt_tags = self._generate_tags_for_lower_third(lt)
            all_tags.extend(lt_tags)
            
        # Remove duplicates
        unique_tags = []
        for tag in all_tags:
            if tag not in unique_tags:
                unique_tags.append(tag)
                
        return unique_tags
    
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
                
        elif lower_third['type'] == 'Event':
            # Add event tag
            event_text = lower_third['main_text'].split(':')[0] if ':' in lower_third['main_text'] else lower_third['main_text']
            tags.append(f"Event:{event_text}")
            
        # Extract session information if available
        if 'session' in lower_third:
            tags.append(f"Session:{lower_third['session']}")
            
        # Add timestamp tags
        tags.append(f"Time:{lower_third['start_time_str']}")
        
        # Add date if available
        if 'date' in lower_third:
            tags.append(f"Date:{lower_third['date']}")
        
        return tags
