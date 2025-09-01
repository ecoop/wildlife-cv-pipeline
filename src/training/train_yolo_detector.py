import yaml
import json
import os
from pathlib import Path
from ultralytics import YOLO
import argparse
import torch


def create_yolo_config(coco_annotations_path, images_dir, output_dir):
    """Create YOLO dataset configuration"""
    
    # Load COCO data
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Map all animal categories to single 'animal' class
    categories = [cat for cat in coco_data['categories'] if cat['name'].lower() != 'animal']
    class_names = ['animal']  # Single generic class
    
    # Create YOLO dataset config
    dataset_config = {
        'path': str(Path(output_dir).absolute()),
        'train': 'images',  # All images in one directory for now
        'val': 'images',    # YOLO will create splits automatically
        'test': 'images',
        'nc': len(class_names),
        'names': class_names
    }
    
    # Save dataset config
    config_path = Path(output_dir) / 'dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Created YOLO dataset config: {config_path}")
    print(f"Classes: {class_names}")
    
    return config_path, class_names


def convert_coco_to_yolo_structure(coco_annotations_path, source_images_dir, output_dir):
    """Convert COCO dataset to YOLO directory structure"""
    
    # Load COCO data
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create category mapping - all animals map to single class (exclude 'Animal')
    categories = {cat['id']: cat for cat in coco_data['categories'] if cat['name'].lower() != 'animal'}
    category_id_to_yolo_id = {cat_id: 0 for cat_id in categories.keys()}  # All map to class 0 (animal)
    
    print(f"Category mapping (all species -> animal):")
    for cat_id in categories.keys():
        print(f"  {categories[cat_id]['name']}: COCO ID {cat_id} -> YOLO ID 0 (animal)")
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        if ann['category_id'] in categories:  # Skip 'Animal' category
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
    
    # Process each image
    processed_images = 0
    processed_annotations = 0
    
    for image_info in coco_data['images']:
        image_id = image_info['id']
        
        # Skip images without valid annotations
        if image_id not in annotations_by_image:
            continue
        
        # Copy image file
        source_image_path = Path(source_images_dir) / image_info['file_name']
        if not source_image_path.exists():
            print(f"Warning: Image not found: {source_image_path}")
            continue
        
        # Use same filename for consistency
        target_image_path = images_dir / image_info['file_name']
        if not target_image_path.exists():
            import shutil
            shutil.copy2(source_image_path, target_image_path)
        
        # Convert annotations to YOLO format
        image_width = image_info['width']
        image_height = image_info['height']
        
        yolo_annotations = []
        for ann in annotations_by_image[image_id]:
            category_id = ann['category_id']
            if category_id not in category_id_to_yolo_id:
                continue
                
            # Convert COCO bbox to YOLO format
            x, y, w, h = ann['bbox']  # COCO: [x, y, width, height]
            
            # Convert to YOLO format: [class_id, center_x, center_y, width, height] (normalized)
            center_x = (x + w/2) / image_width
            center_y = (y + h/2) / image_height
            norm_width = w / image_width
            norm_height = h / image_height
            
            yolo_class_id = category_id_to_yolo_id[category_id]
            yolo_annotations.append(f"{yolo_class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
            processed_annotations += 1
        
        # Save YOLO label file
        label_filename = Path(image_info['file_name']).stem + '.txt'
        label_path = labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        processed_images += 1
        
        if processed_images % 500 == 0:
            print(f"Processed {processed_images} images...")
    
    print(f"Conversion complete:")
    print(f"  Images: {processed_images}")
    print(f"  Annotations: {processed_annotations}")
    
    return processed_images, processed_annotations


def train_yolo_detector(dataset_config_path, model_size='n', epochs=100):
    """Train YOLOv8 detection model"""
    
    print(f"Starting YOLOv8{model_size} training...")
    
    # Load pre-trained model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train the model
    results = model.train(
        data=dataset_config_path,
        epochs=epochs,
        imgsz=640,
        batch=16,
        name=f'wildlife_detector_yolov8{model_size}',
        device='mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    )
    
    print(f"Training complete. Model saved to: runs/detect/wildlife_detector_yolov8{model_size}/weights/best.pt")
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 wildlife detection model')
    parser.add_argument('--coco_annotations', 
                        default='~/datasets/wildlife_coco_dataset/annotations.json',
                        help='Path to COCO annotations file')
    parser.add_argument('--images_dir',
                        default='~/datasets/wildlife_coco_dataset/images', 
                        help='Path to images directory')
    parser.add_argument('--output_dir',
                        default='~/datasets/yolo_wildlife_dataset',
                        help='Output directory for YOLO dataset')
    parser.add_argument('--model_size', default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--convert_only', action='store_true',
                        help='Only convert dataset, do not train')
    
    args = parser.parse_args()
    
    # Expand paths
    args.coco_annotations = os.path.expanduser(args.coco_annotations)
    args.images_dir = os.path.expanduser(args.images_dir)
    
    # Verify inputs
    if not os.path.exists(args.coco_annotations):
        print(f"COCO annotations not found: {args.coco_annotations}")
        return
        
    if not os.path.exists(args.images_dir):
        print(f"Images directory not found: {args.images_dir}")
        return
    
    # Convert COCO to YOLO format
    print("Converting COCO dataset to YOLO format...")
    processed_images, processed_annotations = convert_coco_to_yolo_structure(
        args.coco_annotations, 
        args.images_dir, 
        args.output_dir
    )
    
    if processed_images == 0:
        print("No images processed. Check your COCO file and image paths.")
        return
    
    # Create YOLO dataset config
    dataset_config_path, class_names = create_yolo_config(
        args.coco_annotations,
        args.images_dir, 
        args.output_dir
    )
    
    if args.convert_only:
        print(f"Dataset conversion complete. YOLO dataset ready at: {args.output_dir}")
        return
    
    # Train model
    print(f"Training YOLOv8{args.model_size} on {processed_images} images...")
    results = train_yolo_detector(dataset_config_path, args.model_size, args.epochs)
    
    print(f"\nNext steps:")
    print(f"1. Test model: yolo predict model=runs/detect/wildlife_detector_yolov8{args.model_size}/weights/best.pt source=path/to/test/images")
    print(f"2. Update wildlife_detection_ringbuffer.py to use: runs/detect/wildlife_detector_yolov8{args.model_size}/weights/best.pt")


if __name__ == "__main__":
    main()