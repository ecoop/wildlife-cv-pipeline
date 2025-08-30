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


@dataclass
class DetectionConfig:
    """Configuration for detection system"""
    confidence_threshold: float = 0.2
    sampling_rate: int = 10  # Check every Nth frame
    timeout: float = 2.0
    device: str = "CPU"
    batch_size: int = 1
    # Ring buffer specific settings
    buffer_before_seconds: float = 5.0
    buffer_after_seconds: float = 20.0
    persistence_detections: int = 2
    persistence_frames: int = 5


@dataclass
class SystemConfig:
    """Overall system configuration"""
    output_dir: str = "recordings"
    fps: int = 5  # Camera FPS
    detection_fps: float = 2.0  # Processing FPS (derived from fps/sampling_rate)
    resolution_width: int = 1280
    resolution_height: int = 720


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


class WildlifeDetector:
    """Placeholder for actual detection model"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        # TODO: Load actual detection model (YOLOv8, etc.)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading detection model from: {model_path}")
        
    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> list:
        """
        Detect animals in frame
        
        Returns:
            List of Detection objects (without camera_name - added by caller)
        """
        # PLACEHOLDER: Replace with actual model inference
        import random
        
        if random.random() < 0.15:  # 15% chance of detection for testing
            h, w = frame.shape[:2]
            return [Detection(
                timestamp=time.time(),
                confidence=random.uniform(0.4, 0.95),
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
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
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
                           f"(conf: {detection.confidence:.2f}) "
                           f"at {detection.bbox}")
        
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
            
        self.logger.info("Stopping recording")
        self.recording = False
        self.recording_start_time = None
        
    def save_clip_async(self, initial_frames: list, start_timestamp: float):
        """Save video clip in background"""
        def save_clip():
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_timestamp))
            output_path = Path(self.system_config.output_dir) / f"{self.camera_config.name}_{timestamp_str}.mp4"
            
            if initial_frames:
                h, w = initial_frames[0][0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, self.detection_fps, (w, h))
                
                # Write initial frames
                for frame, _ in initial_frames:
                    out.write(frame)
                
                # Continue writing while recording
                last_frame_time = start_timestamp
                while self.recording:
                    new_frames = self.frame_buffer.get_frames_since(last_frame_time)
                    for frame, frame_timestamp in new_frames:
                        if frame_timestamp > last_frame_time:
                            out.write(frame)
                            last_frame_time = frame_timestamp
                    time.sleep(0.1)
                
                out.release()
                self.logger.info(f"Saved: {output_path}")
        
        save_thread = threading.Thread(target=save_clip, daemon=True)
        save_thread.start()
        
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
        self.detector = WildlifeDetector()
        
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
        
        # Load system-level config
        system_settings = {
            'output_dir': config.get('output_dir', 'recordings'),
            'fps': config.get('fps', 5),
            'resolution_width': config.get('resolution_width', 1280),
            'resolution_height': config.get('resolution_height', 720)
        }
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
                'queue_size': monitor.frame_queue.qsize()
            }
            status['cameras'][name] = camera_status
            
            if monitor.running:
                status['active_cameras'] += 1
            if monitor.recording:
                status['recording_cameras'] += 1
                
        return status


def main():
    parser = argparse.ArgumentParser(description='Multi-Camera Wildlife Detection System')
    parser.add_argument('--config', default='cameras.yaml', help='YAML config file path')
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
detection:
  confidence_threshold: 0.55
  detection_fps: 2.0
  buffer_before_seconds: 5.0
  buffer_after_seconds: 20.0
  output_directory: "wildlife_clips"

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
            logging.getLogger("Status").info(
                f"Active: {status['active_cameras']}, "
                f"Recording: {status['recording_cameras']} cameras"
            )
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        logging.error(f"System error: {e}")
    finally:
        system.stop()


if __name__ == "__main__":
    main()