# v44 - COCO Annotation Tracer and Editor
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image
import shutil

logger = logging.getLogger(__name__)


class COCOAnnotationTracer:
    """Trace cropped images back to original annotations and edit labels"""
    
    def __init__(self, coco_json_path: str, images_dir: str, cropped_dataset_dir: str):
        """
        Initialize tracer
        
        Args:
            coco_json_path: Path to COCO annotations.json
            images_dir: Directory with original images
            cropped_dataset_dir: Directory with cropped classification dataset
        """
        self.coco_json_path = coco_json_path
        self.images_dir = images_dir
        self.cropped_dataset_dir = cropped_dataset_dir
        
        # Load COCO data
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create lookup mappings
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.category_name_to_id = {cat['name']: cat['id'] for cat in self.coco_data['categories']}
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.annotation_id_to_ann = {ann['id']: ann for ann in self.coco_data['annotations']}
        
        logger.info(f"Loaded COCO dataset with {len(self.coco_data['annotations'])} annotations")
    
    def trace_cropped_image(self, cropped_filename: str) -> Optional[Dict]:
        """
        Trace a cropped image back to its original annotation
        
        Args:
            cropped_filename: Name like "squirrel_003783.jpg"
            
        Returns:
            Dictionary with tracing info or None if not found
        """
        # Extract annotation ID from filename
        if '_' not in cropped_filename:
            logger.error(f"Invalid cropped filename format: {cropped_filename}")
            return None
        
        try:
            # Parse "squirrel_003783.jpg" -> category="squirrel", ann_id=3783
            parts = cropped_filename.replace('.jpg', '').split('_')
            if len(parts) != 2:
                logger.error(f"Cannot parse filename: {cropped_filename}")
                return None
            
            category_name = parts[0]
            annotation_id = int(parts[1])
            
            # Find the annotation
            if annotation_id not in self.annotation_id_to_ann:
                logger.error(f"Annotation ID {annotation_id} not found in COCO data")
                return None
            
            annotation = self.annotation_id_to_ann[annotation_id]
            
            # Get image info
            image_info = self.image_id_to_info[annotation['image_id']]
            
            # Get current category
            current_category = self.category_id_to_name[annotation['category_id']]
            
            trace_info = {
                'cropped_filename': cropped_filename,
                'annotation_id': annotation_id,
                'current_category': current_category,
                'expected_category': category_name,
                'is_mislabeled': current_category != category_name,
                'original_image': image_info['file_name'],
                'bbox': annotation['bbox'],
                'annotation': annotation,
                'image_info': image_info
            }
            
            return trace_info
            
        except Exception as e:
            logger.error(f"Error tracing {cropped_filename}: {e}")
            return None
    
    def find_annotation_errors(self, cropped_dir: str = None) -> List[Dict]:
        """
        Find all mislabeled annotations in cropped dataset
        
        Args:
            cropped_dir: Specific directory to check (e.g., "train/squirrel")
            
        Returns:
            List of error information
        """
        errors = []
        
        # Determine which directories to check
        if cropped_dir:
            check_dirs = [cropped_dir]
        else:
            # Check all category directories in train/val/test
            check_dirs = []
            for split in ['train', 'val', 'test']:
                split_dir = os.path.join(self.cropped_dataset_dir, split)
                if os.path.exists(split_dir):
                    for category in os.listdir(split_dir):
                        category_path = os.path.join(split_dir, category)
                        if os.path.isdir(category_path):
                            check_dirs.append(os.path.join(split, category))
        
        # Check each directory
        for dir_path in check_dirs:
            full_path = os.path.join(self.cropped_dataset_dir, dir_path)
            if not os.path.exists(full_path):
                continue
            
            expected_category = os.path.basename(dir_path)
            logger.info(f"Checking {dir_path} for mislabeled {expected_category} images...")
            
            for filename in os.listdir(full_path):
                if filename.endswith('.jpg'):
                    trace_info = self.trace_cropped_image(filename)
                    if trace_info and trace_info['is_mislabeled']:
                        trace_info['found_in_directory'] = dir_path
                        errors.append(trace_info)
        
        return errors
    
    def correct_annotation(self, annotation_id: int, new_category: str, backup: bool = True) -> bool:
        """
        Correct an annotation's category in the COCO JSON
        
        Args:
            annotation_id: ID of annotation to correct
            new_category: New category name
            backup: Whether to backup original file
            
        Returns:
            True if successful
        """
        try:
            if new_category not in self.category_name_to_id:
                logger.error(f"Category '{new_category}' not found in COCO categories")
                return False
            
            if annotation_id not in self.annotation_id_to_ann:
                logger.error(f"Annotation ID {annotation_id} not found")
                return False
            
            # Backup original file
            if backup:
                backup_path = self.coco_json_path + '.backup'
                shutil.copy2(self.coco_json_path, backup_path)
                logger.info(f"Backed up original to {backup_path}")
            
            # Update the annotation
            old_category_id = self.annotation_id_to_ann[annotation_id]['category_id']
            old_category_name = self.category_id_to_name[old_category_id]
            new_category_id = self.category_name_to_id[new_category]
            
            # Find and update in the main data structure
            for i, ann in enumerate(self.coco_data['annotations']):
                if ann['id'] == annotation_id:
                    self.coco_data['annotations'][i]['category_id'] = new_category_id
                    break
            
            # Update our lookup
            self.annotation_id_to_ann[annotation_id]['category_id'] = new_category_id
            
            # Save updated COCO file
            with open(self.coco_json_path, 'w') as f:
                json.dump(self.coco_data, f, indent=2)
            
            logger.info(f"✅ Corrected annotation {annotation_id}: {old_category_name} → {new_category}")
            return True
            
        except Exception as e:
            logger.error(f"Error correcting annotation {annotation_id}: {e}")
            return False
    
    def batch_correct_annotations(self, corrections: List[Tuple[int, str]]) -> Dict[str, int]:
        """
        Apply multiple corrections at once
        
        Args:
            corrections: List of (annotation_id, new_category) tuples
            
        Returns:
            Dictionary with success/failure counts
        """
        results = {'success': 0, 'failed': 0}
        
        # Backup once before batch
        backup_path = self.coco_json_path + '.backup'
        shutil.copy2(self.coco_json_path, backup_path)
        logger.info(f"Backed up original to {backup_path}")
        
        for annotation_id, new_category in corrections:
            if self.correct_annotation(annotation_id, new_category, backup=False):
                results['success'] += 1
            else:
                results['failed'] += 1
        
        logger.info(f"Batch correction complete: {results['success']} success, {results['failed']} failed")
        return results
    
    def create_correction_report(self, output_file: str = "annotation_errors.txt"):
        """Create a report of all annotation errors for manual review"""
        errors = self.find_annotation_errors()
        
        with open(output_file, 'w') as f:
            f.write("ANNOTATION ERROR REPORT\n")
            f.write("="*50 + "\n\n")
            
            for error in errors:
                f.write(f"MISLABELED IMAGE: {error['cropped_filename']}\n")
                f.write(f"  Found in directory: {error['found_in_directory']}\n")
                f.write(f"  Current label: {error['current_category']}\n")
                f.write(f"  Expected label: {error['expected_category']}\n")
                f.write(f"  Original image: {error['original_image']}\n")
                f.write(f"  Annotation ID: {error['annotation_id']}\n")
                f.write(f"  Bbox: {error['bbox']}\n")
                f.write("\n")
        
        logger.info(f"Created error report: {output_file}")
        logger.info(f"Found {len(errors)} mislabeled images")
        
        return errors


def trace_and_fix_annotations(coco_json_path: str, images_dir: str, cropped_dataset_dir: str):
    """
    Main function to find and optionally fix annotation errors
    """
    tracer = COCOAnnotationTracer(coco_json_path, images_dir, cropped_dataset_dir)
    
    # Example usage:
    
    # 1. Trace specific cropped images
    problem_images = ["squirrel_003783.jpg", "cat_002772.jpg"]
    
    print("TRACING SPECIFIC PROBLEM IMAGES:")
    print("="*50)
    for filename in problem_images:
        trace_info = tracer.trace_cropped_image(filename)
        if trace_info:
            print(f"\n🔍 {filename}:")
            print(f"  Original image: {trace_info['original_image']}")
            print(f"  Current label: {trace_info['current_category']}")
            print(f"  Expected label: {trace_info['expected_category']}")
            print(f"  Annotation ID: {trace_info['annotation_id']}")
            print(f"  Bbox: {trace_info['bbox']}")
            print(f"  Mislabeled: {trace_info['is_mislabeled']}")
    
    # 2. Find all annotation errors
    print(f"\n\nSCANNING ALL CROPPED IMAGES FOR ERRORS:")
    print("="*50)
    errors = tracer.find_annotation_errors()
    
    if errors:
        print(f"Found {len(errors)} mislabeled images:")
        for error in errors[:10]:  # Show first 10
            print(f"  • {error['cropped_filename']}: {error['current_category']} → should be {error['expected_category']}")
        
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        
        # Create detailed report
        tracer.create_correction_report()
        
        # Example corrections (uncomment to apply):
        # corrections = [
        #     (3783, "cat"),      # squirrel_003783.jpg is actually a cat
        #     (2772, "jay")       # cat_002772.jpg is actually a jay  
        # ]
        # tracer.batch_correct_annotations(corrections)
        
    else:
        print("✅ No mislabeled images found!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Trace and fix annotations
    trace_and_fix_annotations(
        coco_json_path="wildlife_coco_dataset/annotations.json",
        images_dir="wildlife_coco_dataset/images", 
        cropped_dataset_dir="wildlife_classification_dataset"
    )