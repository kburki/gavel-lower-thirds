#!/usr/bin/env python3
"""
EVO Proxy Finder for Gavel Alaska

This script scans the specified SNS EVO shares for video files matching the Gavel Alaska
naming patterns, retrieves their proxy URLs, and processes them to extract lower thirds.

Usage:
    python evo_proxy_finder.py --config config.json [--process] [--list-only] [--single] [--limit N]
"""

import os
import re
import json
import argparse
import logging
import requests
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from lower_thirds_extractor import LowerThirdsExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evo_proxy_finder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EVOProxyFinder:
    def __init__(self, config_path=None):
        """Initialize the proxy finder with the given configuration"""
        self.config = self._load_config(config_path)
        self.session_id = None
        
        # Compile the patterns for better performance
        self.patterns = [
            re.compile(r'^G[SHJOG][A-Z]{2,4}\d{6}[A-Z]?', re.IGNORECASE),  # G[CHAMBER][COMMITTEE_CODE]YYMMDD[SEGMENT]
            re.compile(r'^AKSC\d{6}[A-Z]?', re.IGNORECASE)                 # AKSCYYMMDD[SEGMENT]
        ]
    
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
        
        # Load default config if available
        default_paths = [
            'config/config.json',
            os.path.join(os.path.dirname(__file__), 'config/config.json'),
            os.path.join(os.path.expanduser('~'), '.lower_thirds_extractor', 'config.json')
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
        
        logger.error("No configuration file found.")
        return {}
    
    def login(self):
        """Login to the EVO system and get a session ID"""
        evo_address = self.config.get('evo_settings', {}).get('evo_address', 'http://192.168.1.66')
        username = self.config.get('evo_settings', {}).get('username', 'api')
        password = self.config.get('evo_settings', {}).get('password', '')
        
        login_url = f"{evo_address}/sb-api/api/public/v1.0/login"
        login_data = {
            "username": username,
            "password": password
        }
        
        try:
            response = requests.post(login_url, json=login_data)
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get('status') == 'success':
                    self.session_id = response_data.get('data', {}).get('session_id')
                    logger.info(f"Successfully logged in with session ID: {self.session_id}")
                    return True
                else:
                    logger.error(f"Login failed: {response_data.get('message', 'Unknown error')}")
            else:
                logger.error(f"Login request failed with status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
        
        return False
    
    def check_shares(self):
        """Check if the configured shares exist"""
        shares_to_monitor = self.config.get('evo_shares', [])
        if not shares_to_monitor:
            logger.error("No shares configured for monitoring")
            return False
        
        evo_address = self.config.get('evo_settings', {}).get('evo_address', 'http://192.168.1.66')
        shares_url = f"{evo_address}/sb-api/api/public/v1.0/volumes"
        
        if not self.session_id:
            if not self.login():
                return False
        
        headers = {
            'session_id': self.session_id
        }
        
        try:
            response = requests.get(shares_url, headers=headers)
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get('status') == 'success':
                    available_shares = response_data.get('data', [])
                    
                    # Check if all configured shares exist
                    available_uuids = [share.get('uuid') for share in available_shares]
                    
                    for share in shares_to_monitor:
                        if share.get('volumeUuid') not in available_uuids:
                            logger.warning(f"Share with UUID {share.get('volumeUuid')} not found")
                    
                    return True
                else:
                    logger.error(f"Failed to get shares: {response_data.get('message', 'Unknown error')}")
            else:
                logger.error(f"Shares request failed with status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error checking shares: {str(e)}")
        
        return False
    
    def scan_files_in_share(self, share):
        """Scan a single share for files matching the patterns"""
        if not self.session_id:
            if not self.login():
                return []
        
        evo_address = self.config.get('evo_settings', {}).get('evo_address', 'http://192.168.1.66')
        files_url = f"{evo_address}/sb-api/api/public/v1.0/files"
        
        headers = {
            'session_id': self.session_id
        }
        
        params = {
            'volume_id': share.get('volumeUuid'),
            'path': share.get('path', '/'),
            'recursive': 'true'
        }
        
        matching_files = []
        
        try:
            response = requests.get(files_url, headers=headers, params=params)
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get('status') == 'success':
                    files = response_data.get('data', [])
                    
                    # Filter files matching the patterns
                    for file in files:
                        file_name = file.get('name', '')
                        if any(pattern.match(file_name) for pattern in self.patterns):
                            matching_files.append({
                                'id': file.get('id'),
                                'name': file_name,
                                'path': file.get('path'),
                                'volume_uuid': share.get('volumeUuid'),
                                'share_name': share.get('name')
                            })
                    
                    logger.info(f"Found {len(matching_files)} matching files in share {share.get('name')}")
                else:
                    logger.error(f"Failed to get files: {response_data.get('message', 'Unknown error')}")
            else:
                logger.error(f"Files request failed with status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error scanning files in share {share.get('name')}: {str(e)}")
        
        return matching_files
    
    def get_proxy_url(self, file_id):
        """Get the proxy URL for a file"""
        if not self.session_id:
            if not self.login():
                return None
        
        evo_address = self.config.get('evo_settings', {}).get('evo_address', 'http://192.168.1.66')
        proxy_url_endpoint = f"{evo_address}/sb-api/api/public/v1.0/preview/proxy_url"
        
        headers = {
            'session_id': self.session_id
        }
        
        params = {
            'file_id': file_id
        }
        
        try:
            response = requests.get(proxy_url_endpoint, headers=headers, params=params)
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get('status') == 'success':
                    proxy_url = response_data.get('data', {}).get('video_proxy_url')
                    return proxy_url
                else:
                    logger.error(f"Failed to get proxy URL: {response_data.get('message', 'Unknown error')}")
            else:
                logger.error(f"Proxy URL request failed with status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error getting proxy URL for file {file_id}: {str(e)}")
        
        return None
    
    def download_proxy(self, proxy_url, output_path):
        """Download a proxy file to the specified path"""
        if not proxy_url:
            return False
        
        try:
            response = requests.get(proxy_url, stream=True)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Downloaded proxy to {output_path}")
                return True
            else:
                logger.error(f"Proxy download failed with status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error downloading proxy: {str(e)}")
        
        return False
    
    def process_proxy(self, proxy_path, file_info):
        """Process a proxy file to extract lower thirds"""
        # Create an instance of the lower thirds extractor
        extractor = LowerThirdsExtractor(self.config.get('config_path'))
        
        # Process the video
        lower_thirds = extractor.process_video(proxy_path)
        
        if lower_thirds:
            # Save metadata
            output_dir = self.config.get('output_settings', {}).get('default_output_dir', './output')
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(proxy_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_metadata.json")
            
            metadata = {
                'video': os.path.basename(proxy_path),
                'original_file': file_info.get('name'),
                'share_name': file_info.get('share_name'),
                'path': file_info.get('path'),
                'extraction_date': datetime.now().isoformat(),
                'lower_thirds': lower_thirds
            }
            
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata saved to {output_path}")
            
            # Upload to EVO if requested
            if self.config.get('upload_to_evo', False):
                self.upload_to_evo(metadata, file_info)
            
            # Export to CSV if requested
            if self.config.get('export_csv', False):
                self.export_to_csv(metadata, output_dir)
            
            return True
        else:
            logger.info(f"No lower thirds extracted from {proxy_path}")
            return False
    
    def upload_to_evo(self, metadata, file_info):
        """Upload extracted metadata to EVO via Slingshot API"""
        logger.info(f"Uploading metadata for {file_info.get('name')} to EVO...")
        
        # Implementation will depend on the Slingshot API
        # This is a placeholder that would be implemented based on your API documentation
        return False
    
    def export_to_csv(self, metadata, output_dir):
        """Export metadata to CSV files"""
        # Implementation similar to what you have in lower_thirds_extractor.py
        return False
    
    def run(self, process_files=True, list_only=False, limit=None, single_file=False):
        """
        Run the proxy finder to scan shares and process files
        
        Args:
            process_files (bool): Whether to process the found files
            list_only (bool): Only list matching files without processing
            limit (int): Limit processing to the specified number of files (for testing)
            single_file (bool): Process only the first file found (for testing)
        """
        # Check if shares are configured and accessible
        if not self.check_shares():
            logger.error("Failed to access EVO shares")
            return False
        
        shares_to_monitor = self.config.get('evo_shares', [])
        all_matching_files = []
        
        # Scan each share for matching files
        for share in shares_to_monitor:
            matching_files = self.scan_files_in_share(share)
            all_matching_files.extend(matching_files)
        
        # Sort files by name for consistent processing order
        all_matching_files.sort(key=lambda f: f.get('name', ''))
        
        total_files = len(all_matching_files)
        logger.info(f"Found {total_files} total matching files across all shares")
        
        # Apply limits if specified
        if single_file and total_files > 0:
            logger.info("Single file mode: Processing only the first file")
            all_matching_files = all_matching_files[:1]
        elif limit is not None and limit > 0 and total_files > limit:
            logger.info(f"Limited mode: Processing only the first {limit} files")
            all_matching_files = all_matching_files[:limit]
        
        if list_only:
            # Just print the list of files
            print("\nMatching files:")
            print("--------------")
            for i, file in enumerate(all_matching_files, 1):
                print(f"{i}. {file.get('share_name')}: {file.get('path')}/{file.get('name')}")
            return True
        
        if not process_files:
            return True
        
        # Process each file
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        processed_count = 0
        max_workers = 1  # Default to 1 for single processing
        
        # Only use multiple workers if specifically requested and not in single_file mode
        if not single_file and self.config.get('max_workers', 1) > 1:
            max_workers = self.config.get('max_workers')
        
        logger.info(f"Processing with {max_workers} worker{'s' if max_workers > 1 else ''}")
        
        if max_workers > 1:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for file_info in all_matching_files:
                    # Get the proxy URL
                    proxy_url = self.get_proxy_url(file_info.get('id'))
                    
                    if proxy_url:
                        # Download the proxy
                        proxy_path = os.path.join(temp_dir, f"{file_info.get('name')}_proxy.mp4")
                        
                        if self.download_proxy(proxy_url, proxy_path):
                            # Submit the processing job to the executor
                            future = executor.submit(self.process_proxy, proxy_path, file_info)
                            futures.append((future, proxy_path))
                
                # Process results as they complete
                for future, proxy_path in futures:
                    try:
                        if future.result():
                            processed_count += 1
                        
                        # Clean up the temporary proxy file
                        if os.path.exists(proxy_path):
                            os.remove(proxy_path)
                    except Exception as e:
                        logger.error(f"Error processing proxy {proxy_path}: {str(e)}")
        else:
            # Process files sequentially (better for VPN connections)
            for i, file_info in enumerate(all_matching_files, 1):
                logger.info(f"Processing file {i} of {len(all_matching_files)}: {file_info.get('name')}")
                
                # Get the proxy URL
                proxy_url = self.get_proxy_url(file_info.get('id'))
                
                if proxy_url:
                    # Download the proxy
                    proxy_path = os.path.join(temp_dir, f"{file_info.get('name')}_proxy.mp4")
                    
                    if self.download_proxy(proxy_url, proxy_path):
                        try:
                            if self.process_proxy(proxy_path, file_info):
                                processed_count += 1
                        except Exception as e:
                            logger.error(f"Error processing proxy {proxy_path}: {str(e)}")
                        finally:
                            # Clean up the temporary proxy file
                            if os.path.exists(proxy_path):
                                os.remove(proxy_path)
                else:
                    logger.warning(f"No proxy URL found for {file_info.get('name')}")
        
        logger.info(f"Successfully processed {processed_count} out of {len(all_matching_files)} files")
        return True


def main():
    parser = argparse.ArgumentParser(description='Scan SNS EVO shares for Gavel Alaska videos')
    parser.add_argument('--config', '-c', help='Path to configuration JSON file')
    parser.add_argument('--process', '-p', action='store_true', help='Process the found files')
    parser.add_argument('--list-only', '-l', action='store_true', help='Only list matching files without processing')
    parser.add_argument('--limit', '-n', type=int, help='Limit processing to N files (for testing)')
    parser.add_argument('--single', '-s', action='store_true', help='Process only the first file (for testing)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create the proxy finder
    finder = EVOProxyFinder(args.config)
    
    # Run the process
    finder.run(
        process_files=args.process, 
        list_only=args.list_only,
        limit=args.limit,
        single_file=args.single
    )


if __name__ == "__main__":
    main()