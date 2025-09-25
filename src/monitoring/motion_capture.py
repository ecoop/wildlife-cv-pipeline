"""
Multi-camera wildlife monitoring system with real-time detection and recording.

This module provides a comprehensive wildlife monitoring solution that connects to RTSP camera 
streams, performs real-time animal detection using YOLO models, and automatically records 
video clips when wildlife is detected. The system uses ring buffers to capture footage 
before and after detection events, implements dual sampling rates for efficient monitoring 
vs high-quality recording, and stores results both locally and in Minio object storage.

Classes:
    DetectionConfig: Configuration for detection parameters and sampling rates
    MinioConfig: Configuration for Minio object storage connection  
    SystemConfig: Overall system configuration including camera settings
    CameraConfig: Individual camera configuration and connection details
    FrameBuffer: Ring buffer implementation for pre/post detection video capture
    WildlifeDetector: YOLO-based animal detection with configurable confidence thresholds
    MinioVideoUploader: Handles uploading videos and metadata to Minio storage
    CameraMonitor: Manages individual camera monitoring, detection, and recording
    MultiCameraWildlifeSystem: Orchestrates multiple camera monitors and system lifecycle

Main Functions:
    main(): Entry point with argument parsing and system startup
"""

import cv2
import numpy as np
import threading
import queue
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, List
import logging
from pathlib import Path
import json
import yaml
import argparse
from datetime import datetime
from minio import Minio
from minio.error import S3Error
import tempfile
import os


@dataclass
class DetectionConfig:
    """Configuration for detection system"""
    confidence_threshold: float = 0.2
    monitor_sampling_rate: int = 10  # Check every Nth frame during monitoring
    capture_sampling_rate: int = 2  # Check every Nth frame during recording
    timeout: float = 2.0
    device: str = "CPU"
    batch_size: int = 1
    model_path: Optional[str] = None  # Path to YOLO model file
    # Ring buffer specific settings
    buffer_before_seconds: float = 5.0
    buffer_after_seconds: float = 20.0
    persistence_detections: int = 2
    persistence_frames: int = 5


@dataclass
class MinioConfig:
    """Minio storage configuration"""
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin" 
    secret_key: str = "minioadmin"
    secure: bool = False
    environment: str = "dev"  # 'dev' or 'prod'


@dataclass
class SystemConfig:
    """Overall system configuration"""
    output_dir: str = "recordings"
    fps: int = 5  # Camera FPS
    detection_fps: float = 2.0  # Processing FPS (derived from fps/sampling_rate)
    resolution_width: int = 1280
    resolution_height: int = 720
    minio: Optional[MinioConfig] = None


@dataclass
class CameraConfig:
    """Configuration for individual camera"""
    name: str
    rtsp_url: str
    enabled: bool = True
    # Detection configuration
    detection_type: str = "model"  # "model", "motion", or "none"
    model_path: Optional[str] = None
    confidence_threshold: Optional[float] = None
    # Motion detection parameters
    motion_threshold: float = 0.02
    pixel_threshold: int = 30
    min_area: int = 500
    blur_kernel: int = 5
    # Persistence parameters
    persistence_detections: Optional[int] = None
    persistence_frames: Optional[int] = None


@dataclass
class Detection:
    """Single detection result"""
    timestamp: float
    confidence: float
    bbox: tuple  # (x, y, w, h)
    class_name: str
    frame_number: int
    camera_name: str


class FrameBuffer:
    """Ring buffer for storing video frames with timestamps"""
    
    def __init__(self, max_seconds: float, fps: float):
        self.max_frames = int(max_seconds * fps)
        self.frames = deque(maxlen=self.max_frames)
        self.fps = fps
        self.lock = threading.Lock()
        
    def add_frame(self, frame: np.ndarray, timestamp: float):
        """Add frame with timestamp to buffer"""
        with self.lock:
            self.frames.append((frame.copy(), timestamp))
        
    def get_frames_since(self, start_timestamp: float) -> list:
        """Get all frames since given timestamp"""
        with self.lock:
            return [(frame, ts) for frame, ts in self.frames if ts >= start_timestamp]
        
    def get_all_frames(self) -> list:
        """Get all frames in buffer"""
        with self.lock:
            return list(self.frames)


class MinioVideoUploader:
    """Handle video and metadata uploads to Minio with multiple buckets"""
    
    # Buckets for motion_capture.py to create/use
    BUCKETS = [
        'recordings-dev',      # Development video recordings  
        'recordings-prod',     # Production video recordings
        'images-dev',         # Development detection images
        'images-prod'         # Production detection images
    ]
    
    def __init__(self, minio_config: Optional[MinioConfig]):
        self.config = minio_config
        self.client = None
        self.logger = logging.getLogger(__name__)
        
        if minio_config:
            try:
                self.client = Minio(
                    endpoint=minio_config.endpoint,
                    access_key=minio_config.access_key,
                    secret_key=minio_config.secret_key,
                    secure=minio_config.secure
                )
                
                # Ensure all required buckets exist
                self._ensure_buckets_exist()
                
                self.logger.info(f"Connected to Minio: {minio_config.endpoint}")
            except Exception as e:
                self.logger.error(f"Failed to connect to Minio: {e}")
                self.client = None
    
    def _ensure_buckets_exist(self):
        """Create all required buckets if they don't exist"""
        if not self.client:
            return
        
        for bucket_name in self.BUCKETS:
            try:
                if not self.client.bucket_exists(bucket_name):
                    self.client.make_bucket(bucket_name)
                    self.logger.info(f"Created Minio bucket: {bucket_name}")
            except S3Error as e:
                self.logger.error(f"Failed to create bucket {bucket_name}: {e}")
    
    def _get_bucket_name(self, content_type: str) -> str:
        """Get appropriate bucket name based on content type and environment"""
        env = self.config.environment if self.config else "dev"
        
        if content_type == "recordings":
            return f"recordings-{env}"
        elif content_type == "images":
            return f"images-{env}"
        else:
            # Fallback to recordings bucket
            return f"recordings-{env}"
    
    def upload_video_and_metadata(self, video_path: Path, metadata: dict, camera_name: str, start_timestamp: float):
        """Upload video and metadata to appropriate bucket with timestamp-based structure"""
        if not self.client or not self.config:
            self.logger.warning("Minio not configured, skipping upload")
            return
        
        try:
            # Get appropriate bucket for recordings
            bucket_name = self._get_bucket_name("recordings")
            
            # Create new hybrid path structure: raw-video/YYYY/MM/camera/
            dt = datetime.fromtimestamp(start_timestamp)
            base_path = f"raw-video/{dt.year:04d}/{dt.month:02d}/{camera_name}"
            
            # Clean filename format (matches local naming convention)
            video_filename = f"video_{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}.mp4"
            metadata_filename = f"detections_{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}.json"
            
            video_object_path = f"{base_path}/{video_filename}"
            metadata_object_path = f"{base_path}/{metadata_filename}"
            
            self.logger.info(f"Uploading video: {video_path} (type: {type(video_path)})")
            
            # Upload video file
            self.client.fput_object(
                bucket_name=bucket_name,
                object_name=video_object_path,
                file_path=str(video_path),
                content_type='video/mp4'
            )
            
            self.logger.info("Video upload successful, uploading metadata...")
            
            # Upload metadata as JSON
            metadata_json = json.dumps(metadata, indent=2)
            metadata_bytes = metadata_json.encode('utf-8')
            
            # Create BytesIO stream for metadata upload
            from io import BytesIO
            metadata_stream = BytesIO(metadata_bytes)
            
            self.client.put_object(
                bucket_name=bucket_name,
                object_name=metadata_object_path,
                data=metadata_stream,
                length=len(metadata_bytes),
                content_type='application/json'
            )
            
            self.logger.info(f"Uploaded to Minio bucket '{bucket_name}': {video_object_path}")
            self.logger.info(f"Uploaded metadata: {metadata_object_path}")
            
            # Optionally remove local files after successful upload
            # video_path.unlink()
            # (video_path.parent / f"{video_path.stem}_detections.json").unlink()
            
        except S3Error as e:
            self.logger.error(f"Minio upload failed: {e}")
        except Exception as e:
            self.logger.error(f"Upload error: {e}")
    
    def upload_detection_image(self, image_frame: np.ndarray, metadata: dict, camera_name: str, timestamp: float):
        """Upload individual detection frame to images bucket"""
        if not self.client or not self.config:
            self.logger.warning("Minio not configured, skipping image upload")
            return
        
        try:
            # Get appropriate bucket for images
            bucket_name = self._get_bucket_name("images")
            
            # Create new hybrid path structure for individual detection images
            dt = datetime.fromtimestamp(timestamp)
            base_path = f"detection-images/{dt.year:04d}/{dt.month:02d}/{camera_name}"
            
            # Clean filename format for individual detection images  
            image_filename = f"detection_{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}.jpg"
            metadata_filename = f"detection_{dt.year:04d}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}.json"
            
            image_object_path = f"{base_path}/{image_filename}"
            metadata_object_path = f"{base_path}/{metadata_filename}"
            
            # Encode image to bytes
            _, image_buffer = cv2.imencode('.jpg', image_frame)
            image_bytes = image_buffer.tobytes()
            
            # Upload image
            self.client.put_object(
                bucket_name=bucket_name,
                object_name=image_object_path,
                data=image_bytes,
                length=len(image_bytes),
                content_type='image/jpeg'
            )
            
            # Upload metadata
            metadata_json = json.dumps(metadata, indent=2)
            self.client.put_object(
                bucket_name=bucket_name,
                object_name=metadata_object_path,
                data=metadata_json.encode('utf-8'),
                length=len(metadata_json.encode('utf-8')),
                content_type='application/json'
            )
            
            self.logger.info(f"Uploaded detection image to bucket '{bucket_name}': {image_object_path}")
            
        except S3Error as e:
            self.logger.error(f"Minio image upload failed: {e}")
        except Exception as e:
            self.logger.error(f"Image upload error: {e}")
    
    def test_upload(self, existing_video_path: str):
        """Test upload with an existing video file"""
        import time
        
        test_metadata = {
            'camera_name': 'test_camera',
            'start_timestamp': time.time(),
            'end_timestamp': time.time(),
            'total_detections': 2,
            'detections': [
                {'bbox': [100, 100, 50, 50], 'confidence': 0.85, 'class_name': 'test'},
                {'bbox': [200, 200, 60, 60], 'confidence': 0.92, 'class_name': 'test'}
            ]
        }
        
        video_path = Path(existing_video_path)
        if not video_path.exists():
            self.logger.error(f"Test video not found: {video_path}")
            return
        
        self.logger.info(f"Testing upload with: {video_path}")
        self.upload_video_and_metadata(
            video_path=video_path,
            metadata=test_metadata,
            camera_name='test_camera',
            start_timestamp=time.time()
        )


class MotionDetector:
    """Frame-difference based motion detection"""
    
    def __init__(self, motion_threshold: float = 0.02, pixel_threshold: int = 30, 
                 min_area: int = 500, blur_kernel: int = 5):
        self.motion_threshold = motion_threshold
        self.pixel_threshold = pixel_threshold
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.previous_frame = None
        self.logger = logging.getLogger(__name__)
        
    def detect(self, frame: np.ndarray, confidence_threshold: float = None) -> list:
        """
        Detect motion in frame using frame differencing
        
        Returns:
            List of Detection objects (motion areas as bounding boxes)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # If no previous frame, store and return empty
        if self.previous_frame is None:
            self.previous_frame = gray
            return []
        
        # Calculate absolute difference
        diff = cv2.absdiff(self.previous_frame, gray)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, self.pixel_threshold, 255, cv2.THRESH_BINARY)
        
        # Calculate motion ratio
        motion_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_ratio = motion_pixels / total_pixels
        
        detections = []
        
        if motion_ratio > self.motion_threshold:
            # Find contours for bounding boxes
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Create detection with motion confidence based on area
                    motion_confidence = min(0.95, area / (total_pixels * 0.1))
                    
                    detections.append(Detection(
                        timestamp=time.time(),
                        confidence=motion_confidence,
                        bbox=(x, y, w, h),
                        class_name="motion",
                        frame_number=0,
                        camera_name=""  # Will be set by caller
                    ))
        
        # Update previous frame
        self.previous_frame = gray
        
        return detections


class WildlifeDetector:
    """YOLO-based wildlife detection model"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        if model_path and Path(model_path).exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.logger.info(f"Loaded YOLO model from: {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load YOLO model: {e}")
                self.logger.warning("Falling back to placeholder detector")
        else:
            self.logger.warning(f"Model path not found: {model_path}")
            self.logger.warning("Using PLACEHOLDER detector - provide valid model path!")
        
    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> list:
        """
        Detect animals in frame
        
        Returns:
            List of Detection objects (without camera_name - added by caller)
        """
        if self.model is not None:
            try:
                # Run YOLO inference
                results = self.model(frame, verbose=False)
                detections = []
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            confidence = float(box.conf[0])
                            if confidence >= confidence_threshold:
                                # Get bounding box coordinates (xyxy format)
                                xyxy = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = xyxy
                                
                                # Convert to (x, y, w, h) format
                                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                                
                                # Get class name
                                class_id = int(box.cls[0])
                                class_name = self.model.names.get(class_id, f"class_{class_id}")
                                
                                detections.append(Detection(
                                    timestamp=time.time(),
                                    confidence=confidence,
                                    bbox=bbox,
                                    class_name=class_name,
                                    frame_number=0,
                                    camera_name=""  # Will be set by caller
                                ))
                
                return detections
                
            except Exception as e:
                self.logger.error(f"Detection error: {e}")
                return []
        else:
            # Fallback to placeholder for testing
            import random
            if random.random() < 0.05:  # Reduced frequency for placeholder
                h, w = frame.shape[:2]
                confidence = random.uniform(0.4, 0.95)
                return [Detection(
                    timestamp=time.time(),
                    confidence=confidence,
                    bbox=(random.randint(0, w//2), random.randint(0, h//2), 
                          random.randint(50, 200), random.randint(50, 200)),
                    class_name=random.choice(['mammal', 'bird', 'unknown']),
                    frame_number=0,
                    camera_name=""  # Will be set by caller
                )]
            return []


class CameraMonitor:
    """Monitor for a single camera"""
    
    def __init__(self, camera_config: CameraConfig, system_config: SystemConfig, detection_config: DetectionConfig, save_local: bool = True, upload_minio: bool = True):
        self.camera_config = camera_config
        self.system_config = system_config
        self.detection_config = detection_config
        self.save_local = save_local
        self.upload_minio = upload_minio
        self.logger = logging.getLogger(f"Camera-{camera_config.name}")
        
        # Create detector based on camera configuration
        self.detector = self._create_detector()
        
        # Initialize Minio uploader
        self.minio_uploader = MinioVideoUploader(system_config.minio)
        
        # Calculate effective FPS for both monitoring and recording
        self.monitoring_fps = system_config.fps / detection_config.monitor_sampling_rate
        self.capture_fps = system_config.fps / detection_config.capture_sampling_rate
        
        # Use monitoring FPS as default detection FPS for ring buffer
        self.detection_fps = self.monitoring_fps
        
        # Ring buffer for this camera
        total_buffer_time = detection_config.buffer_before_seconds + detection_config.buffer_after_seconds
        self.frame_buffer = FrameBuffer(total_buffer_time, self.detection_fps)
        
        # Detection state tracking - use camera-specific persistence settings with fallbacks
        self.persistence_frames = (self.camera_config.persistence_frames or 
                                  detection_config.persistence_frames)
        self.persistence_detections = (self.camera_config.persistence_detections or 
                                     detection_config.persistence_detections)
        self.recent_detections = deque(maxlen=self.persistence_frames)
        self.recording = False
        self.recording_start_time = None
        self.last_detection_time = None
        self.recording_detections = []  # Store detections during recording
        
        # Threading
        self.running = False
        self.capture_thread = None
        self.detection_thread = None
        self.frame_queue = queue.Queue(maxsize=30)
    
    def _create_detector(self):
        """Create appropriate detector based on camera configuration"""
        detection_type = self.camera_config.detection_type.lower()
        
        if detection_type == "motion":
            self.logger.info(f"Using motion detection (threshold: {self.camera_config.motion_threshold})")
            return MotionDetector(
                motion_threshold=self.camera_config.motion_threshold,
                pixel_threshold=self.camera_config.pixel_threshold,
                min_area=self.camera_config.min_area,
                blur_kernel=self.camera_config.blur_kernel
            )
        elif detection_type == "model":
            # Use camera-specific model path if provided, otherwise fall back to global
            model_path = self.camera_config.model_path or self.detection_config.model_path
            self.logger.info(f"Using model detection (model: {model_path})")
            return WildlifeDetector(model_path)
        elif detection_type == "none":
            self.logger.info("Detection disabled for this camera")
            return None
        else:
            self.logger.warning(f"Unknown detection type '{detection_type}', using model detection")
            model_path = self.camera_config.model_path or self.detection_config.model_path
            return WildlifeDetector(model_path)
        
    def connect_camera(self) -> Optional[cv2.VideoCapture]:
        """Connect to RTSP camera stream"""
        self.logger.info(f"Connecting to: {self.camera_config.rtsp_url}")
        
        cap = cv2.VideoCapture(self.camera_config.rtsp_url)
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.system_config.resolution_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.system_config.resolution_height)
        
        # Set buffer size to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Change from 2 to 1

        if not cap.isOpened():
            self.logger.error("Failed to connect")
            return None
            
        # Test read
        ret, frame = cap.read()
        if not ret:
            self.logger.error("Failed to read test frame")
            cap.release()
            return None
            
        self.logger.info(f"Connected. Frame size: {frame.shape}")
        return cap
        
    def capture_frames(self):
        """Capture frames from camera"""
        cap = self.connect_camera()
        if not cap:
            return
            
        last_capture_time = 0
        last_recording_state = False
        
        try:
            while self.running:
                current_time = time.time()
                
                # Use different frame rates for monitoring vs recording
                current_fps = self.capture_fps if self.recording else self.monitoring_fps
                frame_interval = 1.0 / current_fps
                
                # Log frame rate changes
                if self.recording != last_recording_state:
                    state_name = "Recording" if self.recording else "Monitoring"
                    self.logger.info(f"Switched to {state_name} mode: {current_fps:.1f} fps")
                    last_recording_state = self.recording
                
                if current_time - last_capture_time < frame_interval:
                    time.sleep(0.01)
                    continue
                    
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame, reconnecting...")
                    cap.release()
                    time.sleep(5)
                    cap = self.connect_camera()
                    if not cap:
                        break
                    continue
                
                last_capture_time = current_time
                
                # Add to ring buffer
                self.frame_buffer.add_frame(frame, current_time)
                
                # Add to detection queue
                try:
                    self.frame_queue.put((frame, current_time), block=False)
                except queue.Full:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Capture error: {e}")
        finally:
            cap.release()
            
    def process_detections(self):
        """Process frames for detection"""
        while self.running:
            try:
                frame, timestamp = self.frame_queue.get(timeout=1.0)
                
                # Run detection if detector is available
                detections = []
                if self.detector:
                    # Use camera-specific confidence threshold if provided, otherwise global
                    confidence_threshold = (self.camera_config.confidence_threshold or 
                                          self.detection_config.confidence_threshold)
                    detections = self.detector.detect(frame, confidence_threshold)
                    
                    # Add camera name to detections
                    for detection in detections:
                        detection.camera_name = self.camera_config.name
                        detection.timestamp = timestamp  # Ensure consistent timestamp
                
                # Process detections
                if detections:
                    self.handle_detections(detections, timestamp)
                
                # Update persistence tracking
                self.recent_detections.append(len(detections) > 0)
                
                # Check recording state
                self.update_recording_state(timestamp)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Detection error: {e}")
                
    def handle_detections(self, detections: list, timestamp: float):
        """Handle detected animals"""
        for detection in detections:
            x, y, w, h = detection.bbox
            area = w * h
            self.logger.info(f"Detection: {detection.class_name} "
                           f"(confidence: {detection.confidence:.3f}) "
                           f"bbox: {detection.bbox} "
                           f"area: {area} pixels")
            
            # Store detections during recording
            if self.recording:
                detection_dict = {
                    'timestamp': detection.timestamp,
                    'confidence': detection.confidence,
                    'bbox': detection.bbox,
                    'class_name': detection.class_name,
                    'camera_name': detection.camera_name
                }
                self.recording_detections.append(detection_dict)
        
        if self.should_start_recording():
            self.start_recording(timestamp)
        
        self.last_detection_time = timestamp
        
    def should_start_recording(self) -> bool:
        """Check if detection persistence criteria is met"""
        if len(self.recent_detections) < self.persistence_frames:
            return False
            
        recent_positive = sum(1 for has_detection in self.recent_detections 
                            if has_detection)
        
        return (recent_positive >= self.persistence_detections and 
                not self.recording)
    
    def start_recording(self, detection_timestamp: float):
        """Start recording clip with pre-buffer"""
        if self.recording:
            return
            
        self.logger.info("Starting recording")
        self.recording = True
        self.recording_start_time = detection_timestamp
        self.recording_detections = []  # Reset detection storage
        
        # Get pre-buffer frames
        buffer_start_time = detection_timestamp - self.detection_config.buffer_before_seconds
        pre_buffer_frames = self.frame_buffer.get_frames_since(buffer_start_time)
        
        # Start saving clip
        self.save_clip_async(pre_buffer_frames, detection_timestamp)
        
    def update_recording_state(self, current_timestamp: float):
        """Update recording state"""
        if not self.recording:
            return
            
        if (self.last_detection_time and 
            current_timestamp - self.last_detection_time > self.detection_config.buffer_after_seconds):
            self.stop_recording()
            
    def stop_recording(self):
        """Stop current recording"""
        if not self.recording:
            return
            
        self.logger.info(f"Stopping recording ({len(self.recording_detections)} detections captured)")
        self.recording = False
        self.recording_start_time = None
        
    def draw_detections(self, frame: np.ndarray, detections: list, frame_timestamp: float) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        if not detections:
            return frame
            
        annotated_frame = frame.copy()
        
        # Find detections for this timestamp (within 0.5 seconds)
        frame_detections = [d for d in detections 
                          if abs(d['timestamp'] - frame_timestamp) < 0.5]
        
        for detection in frame_detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Convert to integers
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Draw bounding box (green)
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Create label with confidence
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw label background (green)
            label_y = y - 10 if y - 10 > text_height else y + h + text_height + 10
            cv2.rectangle(annotated_frame, 
                         (x, label_y - text_height - baseline), 
                         (x + text_width, label_y + baseline), 
                         (0, 255, 0), -1)
            
            # Draw label text (black)
            cv2.putText(annotated_frame, label, (x, label_y - baseline), 
                       font, font_scale, (0, 0, 0), font_thickness)
        
        return annotated_frame
        
    def save_clip_async(self, initial_frames: list, start_timestamp: float):
        """Save video clip with conditional storage (local/Minio based on flags)"""
        def save_clip():
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_timestamp))
            
            # Determine file paths
            if self.save_local:
                video_path = Path(self.system_config.output_dir) / f"{self.camera_config.name}_{timestamp_str}.mp4"
                metadata_path = Path(self.system_config.output_dir) / f"{self.camera_config.name}_{timestamp_str}_detections.json"
            else:
                # Use temporary files for Minio-only uploads
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                    video_path = Path(temp_video.name)
                with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_metadata:
                    metadata_path = Path(temp_metadata.name)
            
            try:
                if initial_frames:
                    h, w = initial_frames[0][0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(str(video_path), fourcc, self.detection_fps, (w, h))
                    
                    # Write initial frames (clean, no annotations)
                    for frame, frame_timestamp in initial_frames:
                        out.write(frame)
                    
                    # Continue writing frames while recording
                    last_frame_time = start_timestamp
                    while self.recording:
                        new_frames = self.frame_buffer.get_frames_since(last_frame_time)
                        for frame, frame_timestamp in new_frames:
                            if frame_timestamp > last_frame_time:
                                out.write(frame)  # Write clean frame without annotations
                                last_frame_time = frame_timestamp
                        time.sleep(0.1)
                    
                    out.release()
                    
                    # Create detection metadata
                    metadata = {
                        'video_info': {
                            'camera_name': self.camera_config.name,
                            'start_timestamp': start_timestamp,
                            'end_timestamp': time.time(),
                            'frame_dimensions': {
                                'width': w,
                                'height': h,
                                'full_frame_bbox': [0, 0, w, h]
                            }
                        },
                        'detection_summary': self._create_detection_summary(),
                        'detections': self.recording_detections
                    }
                    
                    # Save metadata locally if enabled
                    if self.save_local:
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        self.logger.info(f"Saved video: {video_path}")
                        self.logger.info(f"Saved metadata: {metadata_path}")
                    
                    # Upload to Minio if enabled
                    if self.upload_minio:
                        self.minio_uploader.upload_video_and_metadata(
                            video_path=video_path,
                            metadata=metadata,
                            camera_name=self.camera_config.name,
                            start_timestamp=start_timestamp
                        )
                    
                    # Clean up temporary files if not saving locally
                    if not self.save_local:
                        video_path.unlink(missing_ok=True)
                        metadata_path.unlink(missing_ok=True)
                    
            except Exception as e:
                self.logger.error(f"Error saving clip: {e}")
                # Clean up temp files on error
                if not self.save_local:
                    video_path.unlink(missing_ok=True)
                    metadata_path.unlink(missing_ok=True)
        
        save_thread = threading.Thread(target=save_clip, daemon=True)
        save_thread.start()
        
    def _create_detection_summary(self) -> dict:
        """Create summary of detections in this recording"""
        if not self.recording_detections:
            return {
                'max_detections_per_frame': 0,
                'detection_distribution': {'0': 'N/A'},
                'class_counts': {},
                'average_confidence_by_class': {},
                'total_detections': 0
            }
        
        # Count by class
        class_counts = {}
        confidence_by_class = {}
        
        # Group detections by frame timestamp for distribution analysis
        detections_by_frame = {}
        
        for detection in self.recording_detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            frame_timestamp = detection.get('timestamp', detection.get('frame_timestamp', 0))
            
            # Group by class
            if class_name not in class_counts:
                class_counts[class_name] = 0
                confidence_by_class[class_name] = []
            
            class_counts[class_name] += 1
            confidence_by_class[class_name].append(confidence)
            
            # Group by frame for distribution analysis
            if frame_timestamp not in detections_by_frame:
                detections_by_frame[frame_timestamp] = 0
            detections_by_frame[frame_timestamp] += 1
        
        # Calculate average confidence by class
        avg_confidence_by_class = {}
        for class_name, confidences in confidence_by_class.items():
            avg_confidence_by_class[class_name] = sum(confidences) / len(confidences)
        
        # Calculate detection distribution analysis
        detection_counts = list(detections_by_frame.values())
        max_detections_per_frame = max(detection_counts) if detection_counts else 0
        
        # Create distribution histogram
        detection_distribution = {}
        for count in detection_counts:
            count_str = str(count)
            if count_str not in detection_distribution:
                detection_distribution[count_str] = 0
            detection_distribution[count_str] += 1
        
        return {
            'max_detections_per_frame': max_detections_per_frame,
            'detection_distribution': detection_distribution,
            'class_counts': class_counts,
            'average_confidence_by_class': avg_confidence_by_class,
            'total_detections': len(self.recording_detections)
        }
        
    def start(self):
        """Start monitoring this camera"""
        if not self.camera_config.enabled:
            self.logger.info("Camera disabled, skipping")
            return
            
        self.logger.info("Starting camera monitor")
        self.running = True
        
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.detection_thread = threading.Thread(target=self.process_detections, daemon=True)
        
        self.capture_thread.start()
        self.detection_thread.start()
        
    def stop(self):
        """Stop monitoring this camera"""
        self.logger.info("Stopping camera monitor")
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
        if self.detection_thread:
            self.detection_thread.join(timeout=5)


class MultiCameraWildlifeSystem:
    """Main system managing multiple cameras"""
    
    def __init__(self, config_path: str, save_local: bool = True, upload_minio: bool = True):
        self.config_path = config_path
        self.save_local = save_local
        self.upload_minio = upload_minio
        self.logger = logging.getLogger("WildlifeSystem")
        
        # Load configuration
        self.load_config()
        
        # Create camera monitors (each creates its own detector)
        self.camera_monitors: Dict[str, CameraMonitor] = {}
        for camera_config in self.camera_configs:
            monitor = CameraMonitor(camera_config, self.system_config, self.detection_config, 
                                  save_local=save_local, upload_minio=upload_minio)
            self.camera_monitors[camera_config.name] = monitor
        
        # Setup output directory if local saving enabled
        if save_local:
            Path(self.system_config.output_dir).mkdir(exist_ok=True)
        
    def load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load system-level config from recording namespace
        recording_settings = config.get('recording', {})
        system_settings = {
            'output_dir': recording_settings.get('output_dir', 'recordings'),
            'fps': recording_settings.get('fps', 5),
            'resolution_width': recording_settings.get('resolution_width', 1280),
            'resolution_height': recording_settings.get('resolution_height', 720)
        }
        
        # Load Minio config if present
        minio_config_data = config.get('minio')
        if minio_config_data:
            system_settings['minio'] = MinioConfig(**minio_config_data)
        
        self.system_config = SystemConfig(**system_settings)
        
        # Load detection config
        detection_settings = config.get('detection', {})
        # Add default ring buffer settings if not specified
        detection_defaults = {
            'buffer_before_seconds': 5.0,
            'buffer_after_seconds': 20.0,
            'persistence_detections': 2,
            'persistence_frames': 5
        }
        detection_settings = {**detection_defaults, **detection_settings}
        self.detection_config = DetectionConfig(**detection_settings)
        
        # Load camera configs
        self.camera_configs = []
        for camera_data in config.get('cameras', []):
            camera_config = CameraConfig(
                name=camera_data['name'],
                rtsp_url=camera_data['rtsp_url'],
                enabled=camera_data.get('enabled', True),
                # Detection configuration
                detection_type=camera_data.get('detection_type', 'model'),
                model_path=camera_data.get('model_path'),
                confidence_threshold=camera_data.get('confidence_threshold'),
                # Motion detection parameters
                motion_threshold=camera_data.get('motion_threshold', 0.02),
                pixel_threshold=camera_data.get('pixel_threshold', 30),
                min_area=camera_data.get('min_area', 500),
                blur_kernel=camera_data.get('blur_kernel', 5),
                # Persistence parameters
                persistence_detections=camera_data.get('persistence_detections'),
                persistence_frames=camera_data.get('persistence_frames')
            )
            self.camera_configs.append(camera_config)
        
        self.logger.info(f"Loaded config for {len(self.camera_configs)} cameras")
        monitoring_fps = self.system_config.fps / self.detection_config.monitor_sampling_rate
        capture_fps = self.system_config.fps / self.detection_config.capture_sampling_rate
        self.logger.info(f"Frame rates - Monitoring: {monitoring_fps:.1f} fps (every {self.detection_config.monitor_sampling_rate} frames)")
        self.logger.info(f"Frame rates - Recording: {capture_fps:.1f} fps (every {self.detection_config.capture_sampling_rate} frames)")
        self.logger.info(f"Base camera FPS: {self.system_config.fps}")
        
    def start(self):
        """Start all camera monitors"""
        self.logger.info("Starting multi-camera wildlife monitoring system")
        
        # Start each camera monitor
        for name, monitor in self.camera_monitors.items():
            try:
                monitor.start()
                self.logger.info(f"Started monitor for camera: {name}")
            except Exception as e:
                self.logger.error(f"Failed to start camera {name}: {e}")
        
        enabled_cameras = [name for name, monitor in self.camera_monitors.items() 
                          if monitor.camera_config.enabled]
        self.logger.info(f"System started with {len(enabled_cameras)} cameras: {enabled_cameras}")
        
    def stop(self):
        """Stop all camera monitors"""
        self.logger.info("Stopping multi-camera system")
        
        for name, monitor in self.camera_monitors.items():
            try:
                monitor.stop()
            except Exception as e:
                self.logger.error(f"Error stopping camera {name}: {e}")
        
        self.logger.info("System stopped")
        
    def get_system_status(self) -> dict:
        """Get status of all cameras"""
        status = {
            'active_cameras': 0,
            'recording_cameras': 0,
            'cameras': {}
        }
        
        for name, monitor in self.camera_monitors.items():
            camera_status = {
                'enabled': monitor.camera_config.enabled,
                'running': monitor.running,
                'recording': monitor.recording,
                'queue_size': monitor.frame_queue.qsize(),
                'total_detections_this_recording': len(monitor.recording_detections) if monitor.recording else 0
            }
            status['cameras'][name] = camera_status
            
            if monitor.running:
                status['active_cameras'] += 1
            if monitor.recording:
                status['recording_cameras'] += 1
                
        return status


def main():
    parser = argparse.ArgumentParser(description='Multi-Camera Wildlife Detection System with Real-time Annotation')
    parser.add_argument('--config', default='config/cameras.yaml', help='YAML config file path')
    parser.add_argument('--log_level', default='INFO', help='Logging level')
    parser.add_argument('--save-local', action='store_true', default=True,
                       help='Save recordings to local filesystem (default)')
    parser.add_argument('--no-local', action='store_true',
                       help='Skip saving to local filesystem')
    parser.add_argument('--upload-minio', action='store_true', default=True,
                       help='Upload recordings to Minio (default)')
    parser.add_argument('--no-minio', action='store_true',
                       help='Skip uploading to Minio')
    
    args = parser.parse_args()
    
    # Validate storage arguments
    if args.save_local and args.no_local:
        print("Error: Cannot specify both --save-local and --no-local")
        return
    if args.upload_minio and args.no_minio:
        print("Error: Cannot specify both --upload-minio and --no-minio")
        return
    
    # Determine storage options
    save_local = args.save_local and not args.no_local
    upload_minio = (args.upload_minio or not args.save_local) and not args.no_minio
    
    if not save_local and not upload_minio:
        print("Error: Must save somewhere - specify --save-local or --upload-minio")
        return
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not Path(args.config).exists():
        print(f"Config file not found: {args.config}")
        print("Create a YAML config file like:")
        print("""
output_dir: recordings
fps: 5

detection:
  confidence_threshold: 0.2
  monitor_sampling_rate: 10  # Monitoring: every 10th frame
  capture_sampling_rate: 2   # Recording: every 2nd frame (higher rate during capture)
  timeout: 2.0
  device: CPU
  batch_size: 1
  model_path: runs/detect/wildlife_detector_yolov8n/weights/best.pt  # Path to trained YOLO model
  buffer_before_seconds: 5.0
  buffer_after_seconds: 20.0

cameras:
  - name: west_alley
    rtsp_url: rtsp://user:pass@192.168.1.140/stream1
  - name: east_fence
    rtsp_url: rtsp://user:pass@192.168.1.179/stream1
    enabled: false  # Optional, defaults to true
""")
        return
    
    try:
        # Create and start system
        system = MultiCameraWildlifeSystem(args.config, save_local=save_local, upload_minio=upload_minio)
        system.start()
        
        # Status reporting loop
        while True:
            time.sleep(30)  # Status every 30 seconds
            status = system.get_system_status()
            total_recording_detections = sum(
                cam['total_detections_this_recording'] 
                for cam in status['cameras'].values()
            )
            logging.getLogger("Status").info(
                f"Active: {status['active_cameras']}, "
                f"Recording: {status['recording_cameras']} cameras, "
                f"Current recording detections: {total_recording_detections}"
            )
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        logging.error(f"System error: {e}")
    finally:
        system.stop()


if __name__ == "__main__":
    main()