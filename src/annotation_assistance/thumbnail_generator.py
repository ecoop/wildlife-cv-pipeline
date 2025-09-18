#!/usr/bin/env python3
"""
Thumbnail generation tool for wildlife detection video analysis.

This module generates image thumbnails from wildlife detection videos stored in Minio,
with options for annotated (with bounding boxes) and clean (unannotated) versions.
Thumbnails are extracted from key frames and can be used for quick visual verification
of detection accuracy and for training dataset generation.

Classes:
    VideoThumbnailGenerator: Main class for extracting and generating thumbnails

Main Functions:
    main(): Entry point with argument parsing and thumbnail generation
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import tempfile
import cv2
import numpy as np
from minio import Minio
from minio.error import S3Error
import re
from typing import List, Dict, Optional, Tuple
import yaml

class VideoThumbnailGenerator:
    def __init__(self, config_path: str = "config/cameras.yaml"):
        """Initialize with Minio connection from config"""
        self.logger = logging.getLogger(__name__)
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        minio_config = config.get('minio', {})
        
        # Connect to Minio
        self.client = Minio(
            minio_config.get('endpoint', 'localhost:9000'),
            access_key=minio_config.get('access_key', 'minioadmin'),
            secret_key=minio_config.get('secret_key', 'minioadmin'),
            secure=minio_config.get('secure', False)
        )
        
        self.bucket = f"recordings-{minio_config.get('environment', 'dev')}"
        self.logger.info(f"Connected to Minio bucket: {self.bucket}")

    def search_recordings(self, camera: Optional[str] = None, 
                         date: Optional[str] = None,
                         date_range: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Search for video/JSON pairs in raw-video prefix.
        
        Args:
            camera: Camera name to filter (e.g., 'east_fence')
            date: Specific date YYYY-MM-DD
            date_range: Date range YYYY-MM-DD:YYYY-MM-DD
            
        Returns:
            List of {video_path, json_path, camera, timestamp} dicts
        """
        recordings = []
        
        try:
            # List all objects in raw-video prefix
            objects = self.client.list_objects(self.bucket, prefix='raw-video/', recursive=True)
            
            # Group by video/json pairs
            video_files = {}
            json_files = {}
            
            for obj in objects:
                path = obj.object_name
                
                # Parse path: raw-video/YYYY/MM/camera/filename
                match = re.match(r'raw-video/(\d{4})/(\d{2})/([^/]+)/(.+)', path)
                if not match:
                    continue
                    
                year, month, cam, filename = match.groups()
                
                # Apply camera filter
                if camera and cam != camera:
                    continue
                
                # Apply date filters
                file_date = f"{year}-{month}"
                if date:
                    target_date = date[:7]  # YYYY-MM
                    if file_date != target_date:
                        continue
                        
                if date_range:
                    start_date, end_date = date_range.split(':')
                    start_month = start_date[:7]
                    end_month = end_date[:7]
                    if not (start_month <= file_date <= end_month):
                        continue
                
                # Extract timestamp from filename
                timestamp_match = re.search(r'(\d{8})_(\d{6})', filename)
                if not timestamp_match:
                    continue
                    
                date_part, time_part = timestamp_match.groups()
                timestamp = f"{date_part}_{time_part}"
                
                # Key format: camera_YYYYMMDD_HHMMSS
                parts = [cam, timestamp]
                key = '_'.join(parts)
                
                if filename.startswith('video_'):
                    video_files[key] = path
                elif filename.startswith('detections_'):
                    json_files[key] = path
            
            # Match video/json pairs
            for key in video_files:
                if key in json_files:
                    parts = key.split('_')
                    # Find where the date starts (8 consecutive digits)
                    for i, part in enumerate(parts):
                        if len(part) == 8 and part.isdigit():
                            cam = '_'.join(parts[:i])
                            timestamp = '_'.join(parts[i:])
                            break
                    else:
                        continue
                    
                    recordings.append({
                        'video_path': video_files[key],
                        'json_path': json_files[key],
                        'camera': cam,
                        'timestamp': timestamp
                    })
            
            self.logger.info(f"Found {len(recordings)} video/JSON pairs")
            return recordings
            
        except S3Error as e:
            self.logger.error(f"Error searching recordings: {e}")
            return []

    def download_file(self, object_path: str) -> bytes:
        """Download file from Minio and return bytes"""
        try:
            response = self.client.get_object(self.bucket, object_path)
            data = response.data
            response.close()
            return data
        except S3Error as e:
            self.logger.error(f"Error downloading {object_path}: {e}")
            raise

    def parse_detections(self, json_data: bytes) -> Tuple[List[Dict], Dict]:
        """Parse JSON detection data and return detections with video info"""
        try:
            data = json.loads(json_data.decode('utf-8'))
            
            # Extract video info and detections
            video_info = data.get('video_info', {})
            detections = data.get('detections', [])
            
            return detections, video_info
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON: {e}")
            return [], {}

    def extract_thumbnails(self, video_data: bytes, detections: List[Dict], 
                          video_info: Dict, output_dir: str, base_filename: str,
                          annotated: bool = True, size: Optional[Tuple[int, int]] = None,
                          max_thumbnails: int = 10, save_local: bool = True,
                          upload_minio: bool = True) -> List[str]:
        """
        Extract thumbnail images from video.
        
        Args:
            video_data: Raw video bytes
            detections: List of detection objects
            video_info: Video metadata
            output_dir: Directory to save thumbnails (local or Minio prefix)
            base_filename: Base name for output files (matches video name)
            annotated: Whether to draw bounding boxes
            size: Output size (width, height), None for original size
            max_thumbnails: Maximum number of thumbnails to extract
            save_local: Whether to save to local filesystem
            upload_minio: Whether to upload to Minio
            
        Returns:
            List of generated thumbnail file paths (local) or object names (Minio)
        """
        # Save video data to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video.write(video_data)
            temp_video_path = temp_video.name

        thumbnail_paths = []
        
        try:
            # Open video
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                self.logger.error("Could not open video")
                return []

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0:
                fps = 5.0  # Default
            
            # Group detections by timestamp for frame mapping
            detections_by_time = {}
            start_time = video_info.get('start_timestamp', 0)
            end_time = video_info.get('end_timestamp', start_time + 30)
            video_duration = end_time - start_time
            
            for detection in detections:
                timestamp = detection.get('timestamp', start_time)
                if timestamp not in detections_by_time:
                    detections_by_time[timestamp] = []
                detections_by_time[timestamp].append(detection)
            
            # Select frames to extract (evenly distributed + detection frames)
            frame_indices = set()
            
            # Add evenly distributed frames
            if total_frames > 0:
                step = max(1, total_frames // max_thumbnails)
                for i in range(0, total_frames, step):
                    frame_indices.add(i)
            
            # Add frames with detections
            for timestamp in detections_by_time.keys():
                if video_duration > 0:
                    relative_time = timestamp - start_time
                    frame_idx = int((relative_time / video_duration) * total_frames)
                    frame_indices.add(max(0, min(frame_idx, total_frames - 1)))
            
            # Limit to max_thumbnails
            frame_indices = sorted(list(frame_indices))[:max_thumbnails]
            
            # Extract thumbnails
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Calculate timestamp for this frame
                if video_duration > 0:
                    current_time = start_time + (frame_idx / total_frames) * video_duration
                else:
                    current_time = start_time
                
                # Find detections near this frame (within 0.5 seconds)
                frame_detections = []
                for timestamp, dets in detections_by_time.items():
                    if abs(timestamp - current_time) <= 0.5:
                        frame_detections.extend(dets)
                
                # Draw annotations if requested
                if annotated and frame_detections:
                    for detection in frame_detections:
                        bbox = detection.get('bbox', [])
                        if len(bbox) >= 4:
                            x, y, w, h = bbox[:4]
                            x, y, w, h = int(x), int(y), int(w), int(h)
                            
                            confidence = detection.get('confidence', 0.0)
                            label = detection.get('class_name', 'unknown')
                            
                            # Draw rectangle
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                            # Draw label with confidence
                            text = f"{label} {confidence:.2f}"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(frame, (x, y - text_size[1] - 10), 
                                        (x + text_size[0], y), (0, 255, 0), -1)
                            cv2.putText(frame, text, (x, y - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Resize if requested
                if size:
                    frame = cv2.resize(frame, size)
                
                # Generate filename: basefilename_frameXXXXX_[annotated|clean].jpg
                annotation_suffix = "annotated" if annotated else "clean"
                thumbnail_name = f"{base_filename}_frame{frame_idx:05d}_{annotation_suffix}.jpg"
                
                # Encode image to bytes
                _, img_buffer = cv2.imencode('.jpg', frame)
                img_bytes = img_buffer.tobytes()
                
                # Save locally if requested
                if save_local:
                    thumbnail_path = Path(output_dir) / thumbnail_name
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(thumbnail_path), frame)
                    thumbnail_paths.append(str(thumbnail_path))
                    self.logger.debug(f"Saved local thumbnail: {thumbnail_path}")
                
                # Upload to Minio if requested
                if upload_minio:
                    # Determine Minio path structure based on original video path
                    # Extract year/month/camera from base_filename
                    timestamp_match = re.search(r'(\d{4})(\d{2})(\d{2})_(\d{6})', base_filename)
                    if timestamp_match:
                        year, month, day, time_part = timestamp_match.groups()
                        # Get camera name from the recording context (need to pass this)
                        camera = getattr(self, '_current_camera', 'unknown')
                        
                        # Generate Minio object path: thumbnails/YYYY/MM/camera/filename
                        minio_object_name = f"thumbnails/{year}/{month}/{camera}/{thumbnail_name}"
                        
                        try:
                            from io import BytesIO
                            img_stream = BytesIO(img_bytes)
                            
                            self.client.put_object(
                                bucket_name=self.bucket,
                                object_name=minio_object_name,
                                data=img_stream,
                                length=len(img_bytes),
                                content_type='image/jpeg'
                            )
                            thumbnail_paths.append(minio_object_name)
                            self.logger.debug(f"Uploaded thumbnail: {minio_object_name}")
                            
                        except S3Error as e:
                            self.logger.error(f"Error uploading thumbnail to Minio: {e}")
                    else:
                        self.logger.warning(f"Could not parse timestamp from {base_filename} for Minio path")

            cap.release()
            Path(temp_video_path).unlink()  # Clean up temp file
            
            self.logger.info(f"Generated {len(thumbnail_paths)} thumbnails from {base_filename}")
            return thumbnail_paths
            
        except Exception as e:
            self.logger.error(f"Error extracting thumbnails: {e}")
            Path(temp_video_path).unlink(missing_ok=True)
            return []

    def process_recording(self, recording: Dict[str, str], output_dir: str,
                         annotated: bool = True, size: Optional[Tuple[int, int]] = None,
                         max_thumbnails: int = 10, save_local: bool = True,
                         upload_minio: bool = True) -> List[str]:
        """Process a single video/JSON pair and generate thumbnails"""
        video_path = recording['video_path']
        json_path = recording['json_path']
        camera = recording['camera']
        timestamp = recording['timestamp']
        
        self.logger.info(f"Processing {camera} {timestamp}")
        
        try:
            # Store camera for Minio path generation
            self._current_camera = camera
            
            # Download files
            video_data = self.download_file(video_path)
            json_data = self.download_file(json_path)
            
            # Parse detections
            detections, video_info = self.parse_detections(json_data)
            
            # Generate base filename from video path
            video_filename = Path(video_path).stem  # e.g., "video_20250911_135036"
            
            # Extract thumbnails
            thumbnails = self.extract_thumbnails(
                video_data=video_data,
                detections=detections,
                video_info=video_info,
                output_dir=output_dir,
                base_filename=video_filename,
                annotated=annotated,
                size=size,
                max_thumbnails=max_thumbnails,
                save_local=save_local,
                upload_minio=upload_minio
            )
            
            return thumbnails
            
        except Exception as e:
            self.logger.error(f"Error processing {video_path}: {e}")
            return []

def main():
    parser = argparse.ArgumentParser(description='Generate thumbnails from wildlife detection videos')
    parser.add_argument('--config', default='config/cameras.yaml', 
                       help='YAML config file path')
    parser.add_argument('--camera', help='Camera name to filter')
    parser.add_argument('--date', help='Specific date YYYY-MM-DD')
    parser.add_argument('--date-range', help='Date range YYYY-MM-DD:YYYY-MM-DD')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    parser.add_argument('--output-dir', default='thumbnails', 
                       help='Output directory for thumbnails')
    parser.add_argument('--size', help='Output size WIDTHxHEIGHT (e.g., 640x480), default=original')
    parser.add_argument('--max-thumbnails', type=int, default=10,
                       help='Maximum thumbnails per video')
    parser.add_argument('--annotated', action='store_true', default=True,
                       help='Generate annotated thumbnails (default)')
    parser.add_argument('--clean', action='store_true',
                       help='Generate clean (unannotated) thumbnails')
    parser.add_argument('--both', action='store_true',
                       help='Generate both annotated and clean thumbnails')
    parser.add_argument('--save-local', action='store_true', default=True,
                       help='Save thumbnails to local filesystem (default)')
    parser.add_argument('--no-local', action='store_true',
                       help='Skip saving to local filesystem')
    parser.add_argument('--upload-minio', action='store_true', default=True,
                       help='Upload thumbnails to Minio (default)')
    parser.add_argument('--no-minio', action='store_true',
                       help='Skip uploading to Minio')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse size argument
    size = None
    if args.size:
        try:
            width, height = map(int, args.size.split('x'))
            size = (width, height)
        except ValueError:
            print(f"Invalid size format: {args.size}. Use WIDTHxHEIGHT (e.g., 640x480)")
            return
    
    # Determine what to generate
    generate_annotated = args.annotated or args.both
    generate_clean = args.clean or args.both
    
    if not generate_annotated and not generate_clean:
        generate_annotated = True  # Default to annotated
    
    # Determine storage options
    save_local = args.save_local and not args.no_local
    upload_minio = args.upload_minio and not args.no_minio
    
    # Create thumbnail generator
    generator = VideoThumbnailGenerator(args.config)
    
    # Search for recordings
    recordings = generator.search_recordings(
        camera=args.camera,
        date=args.date, 
        date_range=args.date_range
    )
    
    if not recordings:
        print("No matching recordings found")
        return
    
    print(f"Found {len(recordings)} recordings to process")
    
    # Process each recording
    total_thumbnails = 0
    for recording in recordings:
        if generate_annotated:
            thumbnails = generator.process_recording(
                recording, args.output_dir, 
                annotated=True, size=size, max_thumbnails=args.max_thumbnails,
                save_local=save_local, upload_minio=upload_minio
            )
            total_thumbnails += len(thumbnails)
        
        if generate_clean:
            thumbnails = generator.process_recording(
                recording, args.output_dir,
                annotated=False, size=size, max_thumbnails=args.max_thumbnails,
                save_local=save_local, upload_minio=upload_minio
            )
            total_thumbnails += len(thumbnails)
    
    print(f"Generated {total_thumbnails} thumbnails in {args.output_dir}")

if __name__ == "__main__":
    main()