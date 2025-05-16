#!/usr/bin/env python3
"""
Visualization Module for Gavel Lower Thirds Extractor

Generates HTML reports and thumbnails to visualize extracted lower thirds
"""

import os
import sys
import json
import logging
import cv2
from datetime import datetime
from pathlib import Path

from utils import extract_frame, extract_lower_third_region, format_timecode

class LowerThirdsVisualizer:
    """Creates visual representations of detected lower thirds"""
    
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
        self.min_text_confidence = 60
    
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
            'attributes': {},         # Additional attributes like party, role, etc.
            'start_time': None,       # When this lower third appeared
            'last_seen_time': None,   # When this lower third was last detected
        }
    
        self.lower_thirds = []        # List of all detected lower thirds
    
        # Special text markers to identify types of lower thirds
        self.markers = {
            'speaker': ['Sen.', 'Rep.', 'Speaker', 'President', 'Chairman', 'Chair'],
            'bill': ['HB', 'SB', 'SCR', 'HCR', 'CS', 'Bill', 'CS', 'Resolution', 'Amendment', 'Motion'],
            'event': ['Presentation', 'Pledge', 'Introduction', 'At Ease', 'Colors', 'Allegiance', 'Recess', 'Adjournment'],
            'voice': ['Voice of:'],
            'next': ['Next'],
            'legislature': ['Legislature', 'Legislative Day', 'Regular Session'],
            'location': ['Alaska State Capitol', 'Juneau']
        }
    
        # Create debug directory if needed
        if self.debug_mode:
            os.makedirs('debug_frames', exist_ok=True)
    
    def create_thumbnails(self, video_path, lower_thirds, output_dir):
        """
        Generate thumbnail images for each lower third
        
        Args:
            video_path (str): Path to the video file
            lower_thirds (list): List of lower third dictionaries
            output_dir (str): Output directory for thumbnails
            
        Returns:
            int: Number of thumbnails generated
        """
        self.logger.info(f"Generating thumbnails from video: {video_path}")
        
        # Verify the video exists
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found at: {video_path}")
            return 0
            
        # Verify the video can be opened
        try:
            import cv2
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                self.logger.error(f"Could not open video file: {video_path}")
                return 0
                
            # Get basic video info
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.info(f"Video details: {width}x{height}, {fps} fps, {frame_count} frames")
            
            video.release()
        except Exception as e:
            self.logger.error(f"Error opening video: {e}")
            return 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a thumbnail for each lower third
        count = 0
        for i, lt in enumerate(lower_thirds):
            # Get the timestamp (1 second after start)
            timestamp = lt.get('start_time', 0) + 1.0
            
            # Extract frame
            frame = extract_frame(video_path, timestamp)
            if frame is None:
                self.logger.warning(f"Could not extract frame at {format_timecode(timestamp)}")
                continue
            
            # Extract lower third region
            lower_region = extract_lower_third_region(frame)
            
            # Save the lower third region
            thumbnail_path = os.path.join(output_dir, f"lower_third_{i:03d}.jpg")
            cv2.imwrite(thumbnail_path, lower_region)
            
            # Save the full frame for context
            frame_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            
            count += 1
            
        self.logger.info(f"Generated {count} thumbnails in {output_dir}")
        return count
    
    def create_report(self, json_path, output_dir, generate_thumbnails=True):
        """
        Create an HTML report of the lower thirds
        
        Args:
            json_path (str): Path to the JSON results file
            output_dir (str): Output directory for the report
            generate_thumbnails (bool): Whether to generate thumbnails
            
        Returns:
            str: Path to the HTML report
        """
        self.logger.info(f"Creating HTML report from {json_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the JSON data
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            if 'lower_thirds' in data:
                lower_thirds = data['lower_thirds']
                video_path = data.get('video')
            else:
                lower_thirds = data
                video_path = None
                
        except Exception as e:
            self.logger.error(f"Error loading JSON file: {str(e)}")
            return None
        
        # Generate thumbnails if requested
        has_thumbnails = False
        if generate_thumbnails and video_path:
            # Try to find the full path if only filename is provided
            if not os.path.exists(video_path):
                # Try various possible locations relative to the script's location
                possible_paths = [
                    video_path,  # Original path (might be absolute)
                    os.path.join(os.getcwd(), video_path),  # Relative to current directory
                    os.path.join(os.path.dirname(json_path), video_path),  # Relative to JSON location
                    os.path.join(os.path.dirname(json_path), '..', video_path),  # One level up from JSON
                    os.path.join(os.path.dirname(json_path), '..', '..', video_path),  # Two levels up from JSON
                    os.path.join(os.path.dirname(json_path), '..', 'video', os.path.basename(video_path)),  # In 'video' dir one level up
                    os.path.join(os.getcwd(), 'video', os.path.basename(video_path)),  # In 'video' dir under current dir
                ]
                
                self.logger.info(f"Looking for video file '{video_path}' in possible locations...")
                for path in possible_paths:
                    self.logger.debug(f"Checking: {path}")
                    if os.path.exists(path):
                        self.logger.info(f"Found video at: {path}")
                        video_path = path
                        break
                else:
                    # If the loop completes without finding a path, try with just the basename
                    basename = os.path.basename(video_path)
                    for search_path in ["./video", "../video", "../../video", os.path.join(os.getcwd(), "video")]:
                        potential_path = os.path.join(search_path, basename)
                        self.logger.debug(f"Checking: {potential_path}")
                        if os.path.exists(potential_path):
                            self.logger.info(f"Found video at: {potential_path}")
                            video_path = potential_path
                            break
            
            # Create the thumbnails directory
            images_dir = os.path.join(output_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            # Generate the thumbnails
            if os.path.exists(video_path):
                self.create_thumbnails(video_path, lower_thirds, images_dir)
                has_thumbnails = True
            else:
                self.logger.warning(f"Could not find video file: {video_path}")
        
        # Create the HTML report
        html_path = os.path.join(output_dir, "lower_thirds_report.html")
        self._create_html(lower_thirds, html_path, has_thumbnails)
        
        return html_path
    
    def _create_html(self, lower_thirds, output_path, has_thumbnails=False):
        """
        Create the HTML report file with improved display of attributes
    
        Args:
            lower_thirds (list): List of lower third dictionaries
            output_path (str): Path to save the HTML file
            has_thumbnails (bool): Whether thumbnails are available
        
        Returns:
            str: Path to the HTML file
        """
        # Start building the HTML content
        html_content = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gavel Lower Thirds Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #004165; }} /* Alaska blue */
            h2 {{ color: #00838f; }} /* Teal */
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr:hover {{ background-color: #f2f2f2; }}
            .thumbnail {{ max-width: 320px; border: 1px solid #ddd; }}
            .thumbnail-cell {{ text-align: center; }}
            .time {{ font-family: monospace; }}
            .speaker {{ color: #004165; font-weight: bold; }}
            .bill {{ color: #7B3294; font-weight: bold; }}
            .voice {{ color: #008f11; font-weight: bold; }}
            .event {{ color: #008837; font-weight: bold; }}
            .legislature {{ color: #6A1B9A; font-weight: bold; }}
            .location {{ color: #2196F3; font-weight: bold; }}
            .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .badges {{ margin-top: 5px; }}
            .badge {{ 
                display: inline-block; 
                padding: 2px 8px; 
                border-radius: 12px; 
                background-color: #eee; 
                color: #333; 
                font-size: 0.8em;
                margin-right: 5px;
            }}
            .badge-party-democrat {{ background-color: #577cff; color: white; }}
            .badge-party-republican {{ background-color: #ff5757; color: white; }}
            .badge-party-independent {{ background-color: #57b8ff; color: white; }}
            .badge-role {{ background-color: #9c27b0; color: white; }}
            .badge-bill {{ background-color: #ff9800; color: white; }}
            .badge-district {{ background-color: #009688; color: white; }}
        </style>
    </head>
    <body>
        <h1>Gavel Lower Thirds Extraction Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
        <div class="summary">
            <h2>Summary</h2>
            <p>Total lower thirds detected: <strong>{len(lower_thirds)}</strong></p>
    """

        # Calculate some statistics
        types = {}
        speakers = {}
        bills = {}
    
        for lt in lower_thirds:
            # Count by type
            lt_type = lt.get('type', 'Unknown')
            if lt_type in types:
                types[lt_type] += 1
            else:
                types[lt_type] = 1
            
            # Count speakers
            if lt_type == 'Speaker' and 'speaker_name' in lt:
                speaker_name = lt['speaker_name']
                if speaker_name in speakers:
                    speakers[speaker_name] += 1
                else:
                    speakers[speaker_name] = 1
                
            # Count bills
            if lt_type == 'Bill' and 'bill_type' in lt and 'bill_number' in lt:
                bill_id = f"{lt['bill_type']} {lt['bill_number']}"
                if bill_id in bills:
                    bills[bill_id] += 1
                else:
                    bills[bill_id] = 1
    
        # Add type breakdown
        html_content += "        <p>Types breakdown:</p>\n        <ul>\n"
        for type_name, count in types.items():
            html_content += f"            <li><strong>{type_name}:</strong> {count}</li>\n"
        html_content += "        </ul>\n"
    
        # Add speaker breakdown if we have any
        if speakers:
            html_content += "        <p>Speakers:</p>\n        <ul>\n"
            for speaker_name, count in sorted(speakers.items(), key=lambda x: x[1], reverse=True)[:10]:
                html_content += f"            <li><strong>{speaker_name}:</strong> {count} appearances</li>\n"
            html_content += "        </ul>\n"
    
        # Add bill breakdown if we have any
        if bills:
            html_content += "        <p>Bills mentioned:</p>\n        <ul>\n"
            for bill_id, count in sorted(bills.items(), key=lambda x: x[1], reverse=True):
                html_content += f"            <li><strong>{bill_id}:</strong> {count} appearances</li>\n"
            html_content += "        </ul>\n"
        
        html_content += "    </div>\n\n"
    
        # Add the table of lower thirds
        html_content += """    <h2>Detected Lower Thirds</h2>
        <table>
            <tr>
                <th>#</th>
                <th>Type</th>
                <th>Content</th>
                <th>Start Time</th>
                <th>End Time</th>
                <th>Duration</th>
    """
    
        # Add thumbnail column if available
        if has_thumbnails:
            html_content += "            <th>Thumbnail</th>\n"
        
        html_content += "        </tr>\n"
    
        # Add each lower third to the table
        for i, lt in enumerate(lower_thirds):
            duration_sec = lt.get('duration', 0)
            duration_str = str(timedelta(seconds=duration_sec)).split('.')[0]
            lt_type = lt.get('type', '')
        
            # Determine CSS class based on type
            css_class = ''
            if 'Speaker' in lt_type:
                css_class = 'speaker'
            elif 'Bill' in lt_type:
                css_class = 'bill'
            elif 'Voice' in lt_type:
                css_class = 'voice'
            elif 'Event' in lt_type or 'Presentation' in lt_type:
                css_class = 'event'
            elif 'Legislature' in lt_type:
                css_class = 'legislature'
            elif 'Location' in lt_type:
                css_class = 'location'
        
            # Create badges for attributes
            badges_html = '<div class="badges">'
        
            if 'party' in lt:
                party_class = 'badge-party-' + lt['party'].lower()
                badges_html += f'<span class="badge {party_class}">{lt["party"]}</span>'
            
            if 'district' in lt:
                badges_html += f'<span class="badge badge-district">District {lt["district"]}</span>'
            
            if 'location' in lt:
                badges_html += f'<span class="badge">{lt["location"]}</span>'
            
            if 'role' in lt:
                badges_html += f'<span class="badge badge-role">{lt["role"]}</span>'
            
            if 'bill_type' in lt and 'bill_number' in lt:
                badges_html += f'<span class="badge badge-bill">{lt["bill_type"]} {lt["bill_number"]}</span>'
            
            badges_html += '</div>'
        
            # Format content with main text and available attributes
            content_html = f"<div>{lt.get('main_text', '')}</div>"
            if lt.get('context_text'):
                content_html += f"<div><small>{lt.get('context_text', '')}</small></div>"
        
            # Add badges if we have any
            if 'party' in lt or 'district' in lt or 'location' in lt or 'role' in lt or 'bill_type' in lt:
                content_html += badges_html
        
            # Create thumbnail cell if available
            thumbnail_cell = ""
            if has_thumbnails:
                thumbnail_path = f"images/lower_third_{i:03d}.jpg"
                full_frame_path = f"images/frame_{i:03d}.jpg"
                thumbnail_cell = f"""            <td class="thumbnail-cell">
                    <a href="{full_frame_path}" target="_blank">
                        <img src="{thumbnail_path}" class="thumbnail" alt="Lower Third {i+1}">
                    </a>
                </td>
    """
        
            # Add the table row
            html_content += f"""        <tr>
                <td>{i+1}</td>
                <td class="{css_class}">{lt_type}</td>
                <td>{content_html}</td>
                <td class="time">{lt.get('start_time_str', '')}</td>
                <td class="time">{lt.get('end_time_str', '')}</td>
                <td class="time">{duration_str}</td>
    {thumbnail_cell if has_thumbnails else ''}        </tr>
    """
    
        # Close the table and HTML document
        html_content += """    </table>
    </body>
    </html>
    """
    
        # Write the HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path

def generate_report(json_path, output_dir=None, video_path=None):
    """
    Standalone function to generate a report from a JSON results file
    
    Args:
        json_path (str): Path to the JSON results file
        output_dir (str, optional): Output directory for the report
        video_path (str, optional): Path to the source video
        
    Returns:
        str: Path to the HTML report
    """
    # Set default output directory if not provided
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(json_path), 'report')
    
    # Create visualizer and generate report
    visualizer = LowerThirdsVisualizer()
    html_path = visualizer.create_report(json_path, output_dir)
    
    return html_path

if __name__ == "__main__":
    # If run directly, generate a report from a JSON file
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate a visual report from lower thirds data')
    parser.add_argument('json_file', help='Path to the JSON results file')
    parser.add_argument('--output', '-o', help='Output directory for the report')
    parser.add_argument('--video', '-v', help='Path to the video file (for thumbnails)')
    parser.add_argument('--no-thumbnails', action='store_true', help='Disable thumbnail generation')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set output directory
    output_dir = args.output or os.path.join(os.path.dirname(args.json_file), 'report')
    
    # Create visualizer
    visualizer = LowerThirdsVisualizer()
    
    # Generate report
    html_path = visualizer.create_report(
        args.json_file, 
        output_dir,
        generate_thumbnails=not args.no_thumbnails
    )
    
    if html_path:
        print(f"Report generated at: {html_path}")
    else:
        print("Failed to generate report")