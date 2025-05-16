#!/usr/bin/env python3
"""
Configuration Manager for Gavel Lower Thirds Project

Handles loading, validation, and access to configuration settings
"""

import os
import json
import logging
from pathlib import Path

class ConfigManager:
    """Manages configuration settings for the lower thirds extractor"""
    
    def __init__(self, config_path=None):
        """
        Initialize the configuration manager
        
        Args:
            config_path (str, optional): Path to a configuration JSON file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path):
        """
        Load configuration from a JSON file with fallback to defaults
        
        Args:
            config_path (str): Path to the config file
            
        Returns:
            dict: Configuration dictionary
        """
        # Try loading from the specified path
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                self.logger.error(f"Error loading config from {config_path}: {str(e)}")
        
        # Try default paths
        default_paths = [
            'config.json',
            'config/config.json',
            os.path.join(os.path.dirname(__file__), 'config.json'),
            os.path.join(os.path.dirname(__file__), 'config', 'config.json'),
            os.path.join(os.path.expanduser('~'), '.gavel-lower-thirds', 'config.json')
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        config = json.load(f)
                    self.logger.info(f"Loaded configuration from {path}")
                    return config
                except Exception as e:
                    self.logger.error(f"Error loading config from {path}: {str(e)}")
        
        # No config found, return default configuration
        self.logger.warning("No configuration file found. Using default configuration.")
        return self._get_default_config()
    
    def _get_default_config(self):
        """
        Return default configuration values
        
        Returns:
            dict: Default configuration dictionary
        """
        return {
            "video_settings": {
                "sampling_rate": 1,
                "min_text_confidence": 60,
                "tesseract_path": "/usr/bin/tesseract",
                "max_workers": 1
            },
            
            "lower_third_detection": {
                "blue_hsv_lower": [90, 70, 70],
                "blue_hsv_upper": [130, 255, 255],
                "blue_density_threshold": 0.1,
                "edge_density_threshold": 0.05
            },
               
            "regions": {
                "modern_widescreen": [
                    {"name": "full_lower_third", "y1": 0.75, "y2": 1.0, "x1": 0, "x2": 1.0},
                    {"name": "top_bar", "y1": 0.75, "y2": 0.85, "x1": 0, "x2": 1.0},
                    {"name": "bottom_bar", "y1": 0.85, "y2": 1.0, "x1": 0, "x2": 1.0}
                ]
            },
            
            "markers": {
                "speaker": ["Sen.", "Rep.", "Speaker", "President", "Chairman", "Chair"],
                "bill": ["HB", "SB", "SCR", "HCR", "Bill", "CS", "Resolution", "Amendment", "Motion"],
                "event": ["Presentation", "Pledge", "Introduction", "At Ease", "Colors", "Allegiance", "Recess", "Adjourned", "Concluded"],
                "voice": ["Voice of:"],
                "next": ["Next"]
            },
            
            "ocr_settings": {
                "config": "--oem 1 --psm 6 -l eng --dpi 300",
                "preprocessor": "adaptive_threshold"
            },
            
            "database_settings": {
                "use_database": False,
                "database_path": "lower_thirds.db"
            },
            
            "export_csv": True,
            "debug_mode": False,
            
            "output_settings": {
                "default_output_dir": "./output",
                "csv_output_dir": "./output/csv",
                "json_output_dir": "./output/json"
            }
        }
    
    def get(self, key, default=None):
        """
        Get a configuration value with dot notation support
        
        Args:
            key (str): The configuration key (supports dot notation for nested dicts)
            default: Default value if key is not found
            
        Returns:
            The configuration value or default if not found
        """
        # Support nested keys using dot notation (e.g., "video_settings.sampling_rate")
        if '.' in key:
            parts = key.split('.')
            current = self.config
            for part in parts:
                if part in current:
                    current = current[part]
                else:
                    return default
            return current
        else:
            return self.config.get(key, default)
    
    def get_output_path(self, output_type='json', output_dir=None):
        """
        Get the appropriate output directory path
        
        Args:
            output_type (str): Type of output ('json', 'csv', etc.)
            output_dir (str, optional): Base output directory to override config
            
        Returns:
            str: Path to the output directory
        """
        # If output_dir is provided, use it as the base
        if output_dir:
            base_dir = output_dir
        else:
            base_dir = self.get('output_settings.default_output_dir', './output')
            
        # Create the base directory
        os.makedirs(base_dir, exist_ok=True)
        
        # Get type-specific subdirectory
        if output_type == 'json':
            subdir = self.get('output_settings.json_output_dir')
            if not subdir:
                subdir = os.path.join(base_dir, 'json')
        elif output_type == 'csv':
            subdir = self.get('output_settings.csv_output_dir')
            if not subdir:
                subdir = os.path.join(base_dir, 'csv')
        else:
            subdir = base_dir
            
        # Create the subdirectory
        os.makedirs(subdir, exist_ok=True)
        
        return subdir
    
    def update(self, key, value):
        """
        Update a configuration value
        
        Args:
            key (str): The configuration key (supports dot notation)
            value: The new value
        """
        if '.' in key:
            # Handle nested keys
            parts = key.split('.')
            current = self.config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            # Top level key
            self.config[key] = value
            
    def save(self, path=None):
        """
        Save the current configuration to a file
        
        Args:
            path (str, optional): Path to save the config file
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if not path:
            path = os.path.join(os.path.dirname(__file__), 'config', 'config.json')
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            self.logger.info(f"Configuration saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration to {path}: {str(e)}")
            return False
    
    def merge(self, config_dict):
        """
        Merge another configuration dictionary into this one
        
        Args:
            config_dict (dict): Configuration dictionary to merge
        """
        def deep_merge(source, destination):
            for key, value in source.items():
                if key in destination and isinstance(destination[key], dict) and isinstance(value, dict):
                    deep_merge(value, destination[key])
                else:
                    destination[key] = value
                    
        deep_merge(config_dict, self.config)
        
    def get_detection_params(self):
        """
        Get detection parameters adjusted for the specific type of lower thirds
        
        Returns:
            dict: Detection parameters
        """
        detection_params = self.get('lower_third_detection', {})
        
        # Enhanced detection parameters specifically for Alaska Legislature blue lower thirds
        if not detection_params:
            detection_params = {
                "blue_hsv_lower": [90, 70, 70],
                "blue_hsv_upper": [130, 255, 255],
                "blue_density_threshold": 0.1,
                "edge_density_threshold": 0.05
            }
            
        return detection_params
    
    def get_all(self):
        """
        Get the entire configuration dictionary
        
        Returns:
            dict: The entire configuration
        """
        return self.config
