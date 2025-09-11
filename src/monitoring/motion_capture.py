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
    sampling_rate: int = 10  # Check every Nth frame
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
    
    def __init__(self, camera_config: CameraConfig, system_config: SystemConfig, detection_config: DetectionConfig, detector: WildlifeDetector):
        self.camera_config = camera_config
        self.system_config = system_config
        self.detection_config = detection_config
        self.detector = detector
        self.logger = logging.getLogger(f"Camera-{camera_config.name}")
        
        # Initialize Minio uploader
        self.minio_uploader = MinioVideoUploader(system_config.minio)
        
        # Calculate effective detection FPS from sampling rate
        self.detection_fps = system_config.fps / detection_config.sampling_rate
        
        # Ring buffer for this camera
        total_buffer_time = detection_config.buffer_before_seconds + detection_config.buffer_after_seconds
        self.frame_buffer = FrameBuffer(total_buffer_time, self.detection_fps)
        
        # Detection state tracking
        self.recent_detections = deque(maxlen=detection_config.persistence_frames)
        self.recording = False
        self.recording_start_time = None
        self.last_detection_time = None
        self.recording_detections = []  # Store detections during recording
        
        # Threading
        self.running = False
        self.capture_thread = None
        self.detection_thread = None
        self.frame_queue = queue.Queue(maxsize=30)
        
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
            
        frame_interval = 1.0 / self.detection_fps
        last_capture_time = 0
        
        try:
            while self.running:
                current_time = time.time()
                
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
                
                # Run detection
                detections = self.detector.detect(frame, self.detection_config.confidence_threshold)
                
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
            self.logger.info(f"Detection: {detection.class_name} "
                           f"(confidence: {detection.confidence:.3f}) "
                           f"bbox: {detection.bbox}")
            
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
        if len(self.recent_detections) < self.detection_config.persistence_frames:
            return False
            
        recent_positive = sum(1 for has_detection in self.recent_detections 
                            if has_detection)
        
        return (recent_positive >= self.detection_config.persistence_detections and 
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
        """Save video clip with real-time annotation overlay"""
        def save_clip():
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_timestamp))
            
            # File paths
            video_path = Path(self.system_config.output_dir) / f"{self.camera_config.name}_{timestamp_str}.mp4"
            metadata_path = Path(self.system_config.output_dir) / f"{self.camera_config.name}_{timestamp_str}_detections.json"
            
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
                    
                    # Save detection metadata
                    metadata = {
                        'camera_name': self.camera_config.name,
                        'start_timestamp': start_timestamp,
                        'end_timestamp': time.time(),
                        'total_detections': len(self.recording_detections),
                        'detections': self.recording_detections,
                        'detection_summary': self._create_detection_summary()
                    }
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    self.logger.info(f"Saved video: {video_path}")
                    self.logger.info(f"Saved metadata: {metadata_path}")
                    
                    # Upload to Minio if configured
                    self.minio_uploader.upload_video_and_metadata(
                        video_path=video_path,
                        metadata=metadata,
                        camera_name=self.camera_config.name,
                        start_timestamp=start_timestamp
                    )
                    
            except Exception as e:
                self.logger.error(f"Error saving clip: {e}")
        
        save_thread = threading.Thread(target=save_clip, daemon=True)
        save_thread.start()
        
    def _create_detection_summary(self) -> dict:
        """Create summary of detections in this recording"""
        if not self.recording_detections:
            return {}
        
        # Count by class
        class_counts = {}
        confidence_by_class = {}
        
        for detection in self.recording_detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            if class_name not in class_counts:
                class_counts[class_name] = 0
                confidence_by_class[class_name] = []
            
            class_counts[class_name] += 1
            confidence_by_class[class_name].append(confidence)
        
        # Calculate average confidence by class
        avg_confidence_by_class = {}
        for class_name, confidences in confidence_by_class.items():
            avg_confidence_by_class[class_name] = sum(confidences) / len(confidences)
        
        return {
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
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.logger = logging.getLogger("WildlifeSystem")
        
        # Load configuration
        self.load_config()
        
        # Initialize detector (shared across cameras)
        self.detector = WildlifeDetector(self.detection_config.model_path)
        
        # Create camera monitors
        self.camera_monitors: Dict[str, CameraMonitor] = {}
        for camera_config in self.camera_configs:
            monitor = CameraMonitor(camera_config, self.system_config, self.detection_config, self.detector)
            self.camera_monitors[camera_config.name] = monitor
        
        # Setup output directory
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
                enabled=camera_data.get('enabled', True)
            )
            self.camera_configs.append(camera_config)
        
        self.logger.info(f"Loaded config for {len(self.camera_configs)} cameras")
        self.logger.info(f"Detection FPS: {self.system_config.fps / self.detection_config.sampling_rate:.1f} "
                        f"(camera {self.system_config.fps} fps / sampling rate {self.detection_config.sampling_rate})")
        
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
    
    args = parser.parse_args()
    
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
  sampling_rate: 10
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
        system = MultiCameraWildlifeSystem(args.config)
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