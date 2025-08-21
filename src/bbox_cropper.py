# v1
import os
import json
import random
import logging
from PIL import Image
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class BboxCropper:
    """Extract individual animal patches from COCO dataset for classification training"""
    
    def __init__(self, coco_json_path: str, images_dir: str, output_dir: str = "classification_dataset"):
        """
        Initialize bbox cropper
        
        Args:
            coco_json_path: Path to COCO annotations.json
            images_dir: Directory containing the full images
            output_dir: Output directory for cropped classification dataset
        """
        self.coco_json_path = coco_json_path
        self.images_dir = images_dir  
        self.output_dir = output_dir
        
        # Load COCO data
        logger.info(f"Loading COCO data from {coco_json_path}")
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create category mappings
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        
        # Create output directories
        self.train_dir = os.path.join(output_dir, 'train')
        self.val_dir = os.path.join(output_dir, 'val') 
        self.test_dir = os.path.join(output_dir, 'test')
        
        logger.info(f"Initialized bbox cropper - Output: {output_dir}")
        logger.info(f"Found {len(self.coco_data['images'])} images, {len(self.coco_data['annotations'])} annotations")
    
    def create_classification_dataset(self, train_split: float = 0.7, val_split: float = 0.2, 
                                    test_split: float = 0.1, exclude_categories: List[str] = None,
                                    min_size: int = 10):
        """
        Create classification dataset by cropping bboxes
        
        Args:
            train_split: Proportion for training set (default: 70%)
            val_split: Proportion for validation set (default: 20%)  
            test_split: Proportion for test set (default: 10%)
            exclude_categories: Categories to exclude (e.g., ['Animal'])
            min_size: Minimum bbox size in pixels (default: 10)
        """
        if exclude_categories is None:
            exclude_categories = ['Animal']  # Exclude generic detection labels
        
        logger.info("Creating classification dataset from COCO bboxes...")
        logger.info(f"Splits: {train_split:.0%} train, {val_split:.0%} val, {test_split:.0%} test")
        logger.info(f"Excluding categories: {exclude_categories}")
        
        # Group annotations by category
        annotations_by_category = self._group_annotations_by_category(exclude_categories, min_size)
        
        # Create train/val/test splits
        splits = self._create_splits(annotations_by_category, train_split, val_split, test_split)
        
        # Create directory structure
        self._create_directories(annotations_by_category.keys())
        
        # Crop and save images
        total_cropped = self._crop_and_save_images(splits)
        
        # Print summary
        self._print_summary(annotations_by_category, splits, total_cropped)
        
        return total_cropped
    
    def _group_annotations_by_category(self, exclude_categories: List[str], min_size: int) -> Dict[str, List]:
        """Group annotations by category and filter"""
        annotations_by_category = defaultdict(list)
        
        excluded_count = 0
        small_bbox_count = 0
        
        for ann in self.coco_data['annotations']:
            category_name = self.category_id_to_name[ann['category_id']]
            
            # Skip excluded categories
            if category_name in exclude_categories:
                excluded_count += 1
                continue
            
            # Skip very small bboxes
            bbox = ann['bbox']  # [x, y, width, height]
            if bbox[2] < min_size or bbox[3] < min_size:
                small_bbox_count += 1
                continue
            
            annotations_by_category[category_name].append(ann)
        
        logger.info(f"Filtered annotations:")
        logger.info(f"  - Excluded categories: {excluded_count}")
        logger.info(f"  - Too small bboxes: {small_bbox_count}")
        logger.info(f"  - Valid for classification: {sum(len(anns) for anns in annotations_by_category.values())}")
        
        return annotations_by_category
    
    def _create_splits(self, annotations_by_category: Dict[str, List], 
                      train_split: float, val_split: float, test_split: float) -> Dict[str, Dict[str, List]]:
        """Create train/val/test splits for each category"""
        splits = {'train': defaultdict(list), 'val': defaultdict(list), 'test': defaultdict(list)}
        
        logger.info("Creating train/val/test splits:")
        
        for category, annotations in annotations_by_category.items():
            # Shuffle annotations for random split
            random.shuffle(annotations)
            
            # Calculate split indices
            n_total = len(annotations)
            n_train = int(n_total * train_split)
            n_val = int(n_total * val_split)
            
            # Split annotations
            splits['train'][category] = annotations[:n_train]
            splits['val'][category] = annotations[n_train:n_train + n_val]
            splits['test'][category] = annotations[n_train + n_val:]
            
            logger.info(f"  {category}: {n_train} train, {len(splits['val'][category])} val, {len(splits['test'][category])} test")
        
        return splits
    
    def _create_directories(self, categories: List[str]):
        """Create directory structure for classification dataset"""
        # Create main directories
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(split_dir, exist_ok=True)
            
            # Create category subdirectories
            for category in categories:
                category_dir = os.path.join(split_dir, category)
                os.makedirs(category_dir, exist_ok=True)
        
        logger.info(f"Created directory structure with {len(categories)} categories")
    
    def _crop_and_save_images(self, splits: Dict[str, Dict[str, List]]) -> Dict[str, int]:
        """Crop bboxes and save to classification dataset structure"""
        total_cropped = {'train': 0, 'val': 0, 'test': 0}
        
        for split_name, split_data in splits.items():
            logger.info(f"Processing {split_name} split...")
            split_dir = getattr(self, f"{split_name}_dir")
            
            for category, annotations in split_data.items():
                category_dir = os.path.join(split_dir, category)
                cropped_count = 0
                
                for i, ann in enumerate(annotations):
                    try:
                        # Get image info
                        image_info = self.image_id_to_info[ann['image_id']]
                        image_path = os.path.join(self.images_dir, image_info['file_name'])
                        
                        # Skip if image doesn't exist
                        if not os.path.exists(image_path):
                            logger.warning(f"Image not found: {image_path}")
                            continue
                        
                        # Load and crop image
                        with Image.open(image_path) as img:
                            bbox = ann['bbox']  # [x, y, width, height]
                            
                            # Convert to PIL crop format (left, upper, right, lower)
                            left = max(0, int(bbox[0]))
                            upper = max(0, int(bbox[1])) 
                            right = min(img.width, int(bbox[0] + bbox[2]))
                            lower = min(img.height, int(bbox[1] + bbox[3]))
                            
                            # Skip if bbox is invalid
                            if right <= left or lower <= upper:
                                logger.warning(f"Invalid bbox for annotation {ann['id']}: {bbox}")
                                continue
                            
                            # Crop image
                            cropped = img.crop((left, upper, right, lower))
                            
                            # Save cropped image
                            crop_filename = f"{category}_{ann['id']:06d}.jpg"
                            crop_path = os.path.join(category_dir, crop_filename)
                            cropped.save(crop_path, 'JPEG', quality=90)
                            
                            cropped_count += 1
                    
                    except Exception as e:
                        logger.error(f"Error cropping annotation {ann['id']}: {e}")
                        continue
                
                total_cropped[split_name] += cropped_count
                if len(annotations) > 0:
                    logger.info(f"  {category}: {cropped_count}/{len(annotations)} cropped successfully")
        
        return total_cropped
    
    def _print_summary(self, annotations_by_category: Dict[str, List], 
                      splits: Dict[str, Dict[str, List]], total_cropped: Dict[str, int]):
        """Print dataset creation summary"""
        logger.info("\n" + "="*60)
        logger.info("CLASSIFICATION DATASET CREATED")
        logger.info("="*60)
        
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"📊 Total cropped images: {sum(total_cropped.values())}")
        logger.info(f"   • Train: {total_cropped['train']}")
        logger.info(f"   • Val: {total_cropped['val']}")
        logger.info(f"   • Test: {total_cropped['test']}")
        
        logger.info(f"\n📈 Category breakdown:")
        for category in annotations_by_category.keys():
            train_count = len(splits['train'][category])
            val_count = len(splits['val'][category])
            test_count = len(splits['test'][category])
            total = train_count + val_count + test_count
            logger.info(f"   • {category}: {total} total ({train_count}/{val_count}/{test_count})")
        
        logger.info(f"\n🚀 Next steps:")
        logger.info(f"   1. Train classification model on: {self.output_dir}")
        logger.info(f"   2. Categories: {list(annotations_by_category.keys())}")
        logger.info(f"   3. Perfect for EfficientNet or ResNet fine-tuning on Mac Studio M2!")
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the cropped dataset"""
        stats = {
            'train': {},
            'val': {},
            'test': {},
            'total': {}
        }
        
        # Count files in each directory
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.output_dir, split)
            if os.path.exists(split_dir):
                for category in os.listdir(split_dir):
                    category_dir = os.path.join(split_dir, category)
                    if os.path.isdir(category_dir):
                        count = len([f for f in os.listdir(category_dir) if f.endswith('.jpg')])
                        stats[split][category] = count
                        
                        # Add to total
                        if category not in stats['total']:
                            stats['total'][category] = 0
                        stats['total'][category] += count
        
        return stats


def create_classification_dataset(coco_json_path: str, images_dir: str, 
                                output_dir: str = "wildlife_classification_dataset"):
    """
    Main function to create classification dataset from COCO annotations
    
    Args:
        coco_json_path: Path to your COCO annotations.json
        images_dir: Directory with full-size images  
        output_dir: Where to save the cropped classification dataset
    """
    cropper = BboxCropper(coco_json_path, images_dir, output_dir)
    
    # Create the dataset (excludes 'Animal' category by default)
    total_cropped = cropper.create_classification_dataset(
        train_split=0.7,
        val_split=0.2, 
        test_split=0.1,
        exclude_categories=['Animal'],  # Focus on species classification
        min_size=10  # Skip very small bboxes
    )
    
    return total_cropped


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create classification dataset from your COCO export
    logger.info("🚀 Starting bbox cropping for classification training...")
    
    total_cropped = create_classification_dataset(
        coco_json_path="wildlife_coco_dataset/annotations.json",
        images_dir="wildlife_coco_dataset/images",
        output_dir="wildlife_classification_dataset"
    )
    
    print(f"\n✅ Classification dataset ready!")
    print(f"📁 Location: wildlife_classification_dataset/")
    print(f"🎯 Ready for training on Mac Studio M2!")
    print(f"📊 Total images cropped: {sum(total_cropped.values())}")