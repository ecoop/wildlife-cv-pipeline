#!/usr/bin/env python3
"""
Create YOLO labels from curated review frames

Takes the sorted animal/ and no_object/ folders from video review and creates
YOLO format labels using the original detection metadata.

Usage:
    python src/annotation_assistance/create_yolo_labels.py --review_dir path/to/review --output_dir path/to/yolo_dataset
"""

import json
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm


def convert_bbox_to_yolo(bbox, image_width, image_height):
    """Convert bbox from [x, y, w, h] to YOLO format [x_center, y_center, width, height] (normalized)"""
    x, y, w, h = bbox
    
    # Convert to center coordinates
    center_x = (x + w/2) / image_width
    center_y = (y + h/2) / image_height
    norm_width = w / image_width
    norm_height = h / image_height
    
    return center_x, center_y, norm_width, norm_height


def create_yolo_labels(review_dir, output_dir):
    """Create YOLO labels from curated review data"""
    
    review_path = Path(review_dir)
    output_path = Path(output_dir)
    
    # Paths
    animal_dir = review_path / 'animal'
    no_object_dir = review_path / 'no_object'
    metadata_dir = review_path / 'metadata'
    
    output_images_dir = output_path / 'images'
    output_labels_dir = output_path / 'labels'
    
    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    if not animal_dir.exists() or not no_object_dir.exists():
        print(f"Error: Expected animal/ and no_object/ folders in {review_dir}")
        return
    
    if not metadata_dir.exists():
        print(f"Error: metadata/ folder not found in {review_dir}")
        return
    
    processed_count = 0
    false_negatives = []
    
    # Process animal images (with labels)
    print("Processing animal images...")
    animal_images = list(animal_dir.glob('*.jpg'))
    
    for image_path in tqdm(animal_images):
        # Copy image to output
        dest_image_path = output_images_dir / image_path.name
        if not dest_image_path.exists():
            import shutil
            shutil.copy2(image_path, dest_image_path)
        
        # Find corresponding metadata
        metadata_filename = image_path.name.replace('.jpg', '.json')
        metadata_path = metadata_dir / metadata_filename
        
        if not metadata_path.exists():
            print(f"Warning: No metadata found for {image_path.name}")
            continue
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get image dimensions
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        
        image_height, image_width = img.shape[:2]
        
        # Create YOLO label file
        label_filename = image_path.stem + '.txt'
        label_path = output_labels_dir / label_filename
        
        yolo_annotations = []
        
        # Check if this is a false negative (animal image but no detections)
        detections = metadata.get('detections', [])
        
        if not detections:
            # False negative - model missed the animal but human found it
            false_negatives.append(image_path.name)
            print(f"  False negative found: {image_path.name} (needs manual annotation)")
            continue
        
        # Convert each detection to YOLO format
        for detection in detections:
            bbox = detection['bbox']  # [x, y, w, h] format
            
            # Convert to YOLO format (all animals are class 0)
            center_x, center_y, norm_width, norm_height = convert_bbox_to_yolo(
                bbox, image_width, image_height
            )
            
            yolo_line = f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
            yolo_annotations.append(yolo_line)
        
        # Write label file
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        processed_count += 1
    
    # Process no-object images (empty labels)
    print("Processing no-object images...")
    no_object_images = list(no_object_dir.glob('*.jpg'))
    
    for image_path in tqdm(no_object_images):
        # Copy image to output
        dest_image_path = output_images_dir / image_path.name
        if not dest_image_path.exists():
            import shutil
            shutil.copy2(image_path, dest_image_path)
        
        # Create empty label file
        label_filename = image_path.stem + '.txt'
        label_path = output_labels_dir / label_filename
        
        # Write empty label file
        with open(label_path, 'w') as f:
            f.write('')  # Empty file for no objects
        
        processed_count += 1
    
    print(f"\nProcessed {processed_count} images:")
    print(f"  Animal images with detections: {len(animal_images) - len(false_negatives)}")
    print(f"  No-object images: {len(no_object_images)}")
    print(f"  Output: {output_path}")
    
    if false_negatives:
        print(f"\n⚠️  FALSE NEGATIVES FOUND ({len(false_negatives)} images need manual annotation):")
        for fn in false_negatives:
            print(f"    {fn}")
        print(f"\nThese images are in animal/ but had no YOLO detections.")
        print(f"Use a tool like LabelImg to manually annotate bounding boxes.")
        print(f"After annotation, add them to {output_path}")
    
    return processed_count


def main():
    parser = argparse.ArgumentParser(description='Create YOLO labels from curated review data')
    parser.add_argument('--review_dir', required=True,
                       help='Directory containing animal/, no_object/, and metadata/ folders')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for YOLO dataset (will create images/ and labels/ subdirs)')
    
    args = parser.parse_args()
    
    if not Path(args.review_dir).exists():
        print(f"Error: Review directory {args.review_dir} does not exist")
        return
    
    processed_count = create_yolo_labels(args.review_dir, args.output_dir)
    
    if processed_count > 0:
        print(f"\nNext steps:")
        print(f"1. Update dataset.yaml to point to {args.output_dir}")
        print(f"2. Run training: python src/training/train_yolo_detector.py --output_dir {args.output_dir}")
    else:
        print("No images processed. Check your review directory structure.")


if __name__ == "__main__":
    main()