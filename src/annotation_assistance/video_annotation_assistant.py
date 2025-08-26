import json
import os
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
import hashlib
import time


class VideoAnnotationAssistant:
    """
    Pipeline for processing new videos with model-assisted annotation suggestions.
    """
    
    def __init__(self, classifier_model_path, existing_annotations_path, output_dir="new_video_analysis"):
        self.classifier_model_path = classifier_model_path
        self.existing_annotations_path = existing_annotations_path
        self.output_dir = output_dir
        
        # Setup device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon MPS")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        
        # Load existing annotations for duplicate checking
        self.existing_videos = self.load_existing_video_names()
        
        # Load classifier
        self.load_classifier()
        
        # Create output structure
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "crops"), exist_ok=True)
        
    def load_existing_video_names(self):
        """Extract video identifiers from existing annotations to avoid duplicates"""
        print("Loading existing video names...")
        
        if not os.path.exists(self.existing_annotations_path):
            print(f"No existing annotations found at {self.existing_annotations_path}")
            return set()
        
        with open(self.existing_annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        video_names = set()
        for img in coco_data['images']:
            filename = img['file_name']
            # Extract video identifier (everything before _mp4-)
            if '_mp4-' in filename:
                video_id = filename.split('_mp4-')[0]
                video_names.add(video_id)
        
        print(f"Found {len(video_names)} existing video sessions")
        return video_names
    
    def check_video_duplicates(self, video_list):
        """Check which videos are new vs already processed"""
        new_videos = []
        duplicate_videos = []
        
        for video_path in video_list:
            video_name = Path(video_path).stem
            
            # Extract identifier that matches your naming convention
            # Adjust this pattern based on your video naming
            video_id = self.extract_video_identifier(video_name)
            
            if video_id in self.existing_videos:
                duplicate_videos.append(video_path)
            else:
                new_videos.append(video_path)
        
        print(f"Video duplicate check:")
        print(f"  New videos: {len(new_videos)}")
        print(f"  Already processed: {len(duplicate_videos)}")
        
        return new_videos, duplicate_videos
    
    def extract_video_identifier(self, video_filename):
        """
        Extract video identifier from filename to match existing annotations.
        
        Adjust this method based on your video naming convention.
        E.g., if video is "FF0A_2023_06_29_11-36-45_3.mp4", 
        return "FF0A_2023_06_29_11-36-45_3"
        """
        # Remove extension
        base_name = os.path.splitext(video_filename)[0]
        
        # If your videos follow the same pattern as images, return as-is
        # Otherwise, customize this extraction logic
        return base_name
    
    def load_classifier(self):
        """Load trained classifier model"""
        print(f"Loading classifier from {self.classifier_model_path}")
        
        checkpoint = torch.load(self.classifier_model_path, map_location=self.device)
        
        self.backbone = checkpoint['backbone']
        self.num_classes = checkpoint['num_classes']
        self.class_names = checkpoint['class_names']
        
        # Create model
        self.classifier = timm.create_model(
            self.backbone,
            pretrained=False,
            num_classes=self.num_classes
        )
        
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        print(f"Loaded {self.backbone} classifier with classes: {self.class_names}")
    
    def extract_frames(self, video_path, frame_interval=30):
        """
        Extract frames from video at specified intervals.
        
        Args:
            video_path: Path to video file
            frame_interval: Extract every N frames (default 30 = ~6 seconds at 5fps)
        
        Returns:
            List of extracted frame paths
        """
        print(f"Extracting frames from {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        video_name = Path(video_path).stem
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  Video info: {fps:.1f} fps, {total_frames} frames")
        
        extracted_frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at intervals
            if frame_count % frame_interval == 0:
                frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
                frame_path = os.path.join(self.output_dir, "frames", frame_filename)
                
                # Save frame
                cv2.imwrite(frame_path, frame)
                extracted_frames.append(frame_path)
            
            frame_count += 1
        
        cap.release()
        print(f"  Extracted {len(extracted_frames)} frames")
        return extracted_frames
    
    def simple_background_subtraction_detection(self, frame_path, min_area=500):
        """
        Simple motion/change detection as a basic animal detector.
        
        This is a placeholder - you'll want to replace this with:
        1. Your trained Geti detection model, or
        2. A proper object detection model (YOLO, etc.), or  
        3. A more sophisticated background subtraction method
        """
        # For now, return dummy bounding boxes
        # In practice, you'd run actual detection here
        
        # Load frame to get dimensions
        frame = cv2.imread(frame_path)
        h, w = frame.shape[:2]
        
        # Dummy detections - replace with real detector
        dummy_detections = [
            {'bbox': [100, 100, 200, 150], 'confidence': 0.8},  # [x, y, w, h]
            {'bbox': [300, 200, 150, 100], 'confidence': 0.6},
        ]
        
        return dummy_detections
    
    def classify_crop(self, image_crop):
        """Classify a single bbox crop using the trained model"""
        
        # Transform crop for model input
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert to PIL and transform
        if isinstance(image_crop, np.ndarray):
            image_crop = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        
        input_tensor = transform(image_crop).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.classifier(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        predicted_class_name = self.class_names[predicted_class.item()]
        confidence_score = confidence.item()
        
        # Get top 3 predictions for context
        top_probs, top_indices = torch.topk(probabilities, 3, dim=1)
        top_predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            top_predictions.append({
                'class': self.class_names[idx.item()],
                'confidence': prob.item()
            })
        
        return {
            'predicted_class': predicted_class_name,
            'confidence': confidence_score,
            'top_predictions': top_predictions
        }
    
    def process_video(self, video_path, frame_interval=30):
        """Process a single video through the full pipeline"""
        print(f"\n=== Processing {Path(video_path).name} ===")
        
        # Extract frames
        frame_paths = self.extract_frames(video_path, frame_interval)
        if not frame_paths:
            return None
        
        video_results = {
            'video_path': video_path,
            'video_name': Path(video_path).stem,
            'total_frames_extracted': len(frame_paths),
            'frame_analysis': []
        }
        
        # Process each frame
        for frame_path in frame_paths:
            print(f"Processing frame: {Path(frame_path).name}")
            
            # Detect potential animals (replace with real detector)
            detections = self.simple_background_subtraction_detection(frame_path)
            
            frame_result = {
                'frame_path': frame_path,
                'detections': []
            }
            
            if detections:
                # Load frame for cropping
                frame = cv2.imread(frame_path)
                
                for i, detection in enumerate(detections):
                    bbox = detection['bbox']  # [x, y, w, h]
                    x, y, w, h = bbox
                    
                    # Extract crop
                    crop = frame[y:y+h, x:x+w]
                    
                    # Save crop
                    crop_filename = f"{Path(frame_path).stem}_crop_{i}.jpg"
                    crop_path = os.path.join(self.output_dir, "crops", crop_filename)
                    cv2.imwrite(crop_path, crop)
                    
                    # Classify crop
                    classification = self.classify_crop(crop)
                    
                    detection_result = {
                        'bbox': bbox,
                        'detection_confidence': detection['confidence'],
                        'crop_path': crop_path,
                        'classification': classification
                    }
                    
                    frame_result['detections'].append(detection_result)
            
            video_results['frame_analysis'].append(frame_result)
        
        return video_results
    
    def process_video_batch(self, video_paths, frame_interval=30, max_videos=None):
        """Process multiple videos"""
        
        # Check for duplicates first
        new_videos, duplicate_videos, camera_stats = self.check_video_duplicates(video_paths)
        
        if duplicate_videos:
            print(f"\nSkipping {len(duplicate_videos)} already processed videos:")
            for vid in duplicate_videos[:5]:  # Show first 5
                print(f"  {Path(vid).name}")
            if len(duplicate_videos) > 5:
                print(f"  ... and {len(duplicate_videos) - 5} more")
        
        # Process new videos (no prioritization needed since both cameras are trained)
        videos_to_process = new_videos[:max_videos] if max_videos else new_videos
        
        all_results = []
        for i, video_path in enumerate(videos_to_process):
            print(f"\n[{i+1}/{len(videos_to_process)}]", end=" ")
            
            try:
                results = self.process_video(video_path, frame_interval)
                if results:
                    # Add camera info to results
                    video_name = Path(video_path).stem
                    _, camera_id = self.extract_video_identifier(video_name)
                    results['camera_id'] = camera_id
                    
                    all_results.append(results)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
        
        return all_results
    
    def generate_annotation_suggestions(self, results, confidence_threshold=0.7):
        """Generate human-reviewable annotation suggestions"""
        suggestions = {
            'high_confidence': [],    # Auto-accept candidates (>0.9)
            'medium_confidence': [],  # Human review needed (0.7-0.9)
            'low_confidence': [],     # Flagged for attention (<0.7)
            'summary': {}
        }
        
        for video_result in results:
            for frame_result in video_result['frame_analysis']:
                for detection in frame_result['detections']:
                    classification = detection['classification']
                    confidence = classification['confidence']
                    
                    suggestion = {
                        'video_name': video_result['video_name'],
                        'frame_path': frame_result['frame_path'],
                        'crop_path': detection['crop_path'],
                        'bbox': detection['bbox'],
                        'predicted_class': classification['predicted_class'],
                        'confidence': confidence,
                        'top_predictions': classification['top_predictions']
                    }
                    
                    if confidence > 0.9:
                        suggestions['high_confidence'].append(suggestion)
                    elif confidence > confidence_threshold:
                        suggestions['medium_confidence'].append(suggestion)
                    else:
                        suggestions['low_confidence'].append(suggestion)
        
        # Generate summary
        suggestions['summary'] = {
            'high_confidence_count': len(suggestions['high_confidence']),
            'medium_confidence_count': len(suggestions['medium_confidence']),
            'low_confidence_count': len(suggestions['low_confidence']),
            'class_distribution': {}
        }
        
        # Count predictions by class
        for category in ['high_confidence', 'medium_confidence', 'low_confidence']:
            for suggestion in suggestions[category]:
                class_name = suggestion['predicted_class']
                if class_name not in suggestions['summary']['class_distribution']:
                    suggestions['summary']['class_distribution'][class_name] = {'high': 0, 'medium': 0, 'low': 0}
                
                if category == 'high_confidence':
                    suggestions['summary']['class_distribution'][class_name]['high'] += 1
                elif category == 'medium_confidence':
                    suggestions['summary']['class_distribution'][class_name]['medium'] += 1
                else:
                    suggestions['summary']['class_distribution'][class_name]['low'] += 1
        
        return suggestions
    
    def save_results(self, results, suggestions):
        """Save analysis results and suggestions"""
        
        # Save detailed results
        results_path = os.path.join(self.output_dir, "analysis_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save annotation suggestions
        suggestions_path = os.path.join(self.output_dir, "annotation_suggestions.json")
        with open(suggestions_path, 'w') as f:
            json.dump(suggestions, f, indent=2)
        
        print(f"\n=== RESULTS SAVED ===")
        print(f"Analysis results: {results_path}")
        print(f"Annotation suggestions: {suggestions_path}")
        print(f"Frame crops: {self.output_dir}/crops/")
        
        # Print summary
        summary = suggestions['summary']
        print(f"\n=== ANNOTATION SUMMARY ===")
        print(f"High confidence (>90%): {summary['high_confidence_count']} detections")
        print(f"Medium confidence (70-90%): {summary['medium_confidence_count']} need review")
        print(f"Low confidence (<70%): {summary['low_confidence_count']} need attention")
        
        if summary['class_distribution']:
            print(f"\nClass distribution:")
            for class_name, counts in summary['class_distribution'].items():
                total = counts['high'] + counts['medium'] + counts['low']
                print(f"  {class_name}: {total} ({counts['high']} high, {counts['medium']} medium, {counts['low']} low)")


def main():
    parser = argparse.ArgumentParser(description='Process videos with model-assisted annotation')
    parser.add_argument('--videos', nargs='+', required=True,
                        help='List of video files to process')
    parser.add_argument('--model_path', required=True,
                        help='Path to trained classifier model (.pth)')
    parser.add_argument('--existing_annotations', 
                        default='~/datasets/wildlife_coco_dataset/annotations.json',
                        help='Path to existing annotations.json for duplicate checking')
    parser.add_argument('--output_dir', default='new_video_analysis',
                        help='Output directory for results')
    parser.add_argument('--frame_interval', type=int, default=30,
                        help='Extract every N frames (default: 30)')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Maximum number of videos to process')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                        help='Confidence threshold for human review')
    
    args = parser.parse_args()
    
    # Expand paths
    args.existing_annotations = os.path.expanduser(args.existing_annotations)
    
    # Verify model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    # Create assistant
    assistant = VideoAnnotationAssistant(
        classifier_model_path=args.model_path,
        existing_annotations_path=args.existing_annotations,
        output_dir=args.output_dir
    )
    
    # Process videos
    results = assistant.process_video_batch(
        args.videos, 
        frame_interval=args.frame_interval,
        max_videos=args.max_videos
    )
    
    if not results:
        print("No videos processed successfully.")
        return
    
    # Generate suggestions
    suggestions = assistant.generate_annotation_suggestions(
        results, 
        confidence_threshold=args.confidence_threshold
    )
    
    # Save results
    assistant.save_results(results, suggestions)
    
    print(f"\n=== NEXT STEPS ===")
    print(f"1. Review medium confidence predictions in: {args.output_dir}/annotation_suggestions.json")
    print(f"2. Manually verify crops in: {args.output_dir}/crops/")
    print(f"3. Create corrected annotations for training data expansion")
    print(f"4. Consider improving detection model if many false positives/negatives")


if __name__ == "__main__":
    main()