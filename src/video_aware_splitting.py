import json
import os
import random
from collections import defaultdict, Counter
from PIL import Image
import shutil
import argparse
import re


class VideoAwareDatasetSplitter:
    """
    Create proper train/val/test splits by grouping video frames together
    to prevent temporal data leakage.
    """
    
    def __init__(self, coco_file, images_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        self.coco_file = coco_file
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Ensure ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            print(f"Warning: Ratios sum to {total_ratio}, normalizing...")
            self.train_ratio = train_ratio / total_ratio
            self.val_ratio = val_ratio / total_ratio
            self.test_ratio = test_ratio / total_ratio
            
        print(f"Split ratios: {self.train_ratio:.1%} train, {self.val_ratio:.1%} val, {self.test_ratio:.1%} test")
        
    def extract_video_id(self, filename):
        """
        Extract video identifier from filename.
        
        Example: '249_2023_06_29_11-36-45_3_mp4-497_jpg-rf-hash.jpg' 
        Returns: '249_2023_06_29_11-36-45_3'
        """
        # Pattern: everything before '_mp4-{frame_number}'
        match = re.match(r'^(.+)_mp4-\d+_jpg-rf-.+\.jpg$', filename)
        if match:
            return match.group(1)
        
        # Fallback: use first part before frame numbers if no _mp4 pattern
        # Remove common suffixes and frame-like patterns
        base = filename.replace('.jpg', '').replace('.png', '')
        
        # Try to find frame number patterns
        frame_patterns = [
            r'-\d+_jpg-rf-.+$',  # -497_jpg-rf-hash
            r'_frame_?\d+.*$',   # _frame123
            r'_\d{3,}.*$',       # _001234
        ]
        
        for pattern in frame_patterns:
            base = re.sub(pattern, '', base)
            
        return base
    
    def analyze_video_distribution(self, coco_data):
        """Analyze video sources and their annotations"""
        print("\n=== ANALYZING VIDEO SOURCES ===")
        
        # Group annotations by video
        video_annotations = defaultdict(list)
        video_images = defaultdict(set)
        
        # Create image id to filename mapping
        image_info = {img['id']: img['file_name'] for img in coco_data['images']}
        
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            filename = image_info[image_id]
            video_id = self.extract_video_id(filename)
            
            video_annotations[video_id].append(ann)
            video_images[video_id].add(filename)
        
        # Analyze distribution
        video_stats = []
        for video_id, annotations in video_annotations.items():
            category_counts = Counter(ann['category_id'] for ann in annotations)
            video_stats.append({
                'video_id': video_id,
                'num_annotations': len(annotations),
                'num_images': len(video_images[video_id]),
                'categories': dict(category_counts)
            })
        
        # Sort by number of annotations
        video_stats.sort(key=lambda x: x['num_annotations'], reverse=True)
        
        print(f"Found {len(video_stats)} unique video sources")
        print(f"Total annotations: {sum(len(anns) for anns in video_annotations.values())}")
        
        # Show top videos
        print(f"\nTop 10 videos by annotation count:")
        for i, stats in enumerate(video_stats[:10]):
            print(f"  {i+1}. {stats['video_id']}: {stats['num_annotations']} annotations, {stats['num_images']} images")
        
        # Category distribution check
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        print(f"\nChecking category balance across videos...")
        
        for cat_id, cat_name in categories.items():
            if cat_name.lower() == 'animal':
                continue
                
            videos_with_category = [v for v in video_stats if cat_id in v['categories']]
            total_instances = sum(v['categories'][cat_id] for v in videos_with_category)
            print(f"  {cat_name}: {total_instances} instances across {len(videos_with_category)} videos")
        
        return video_annotations, video_images, video_stats
    
    def create_stratified_splits(self, video_stats, categories):
        """Create stratified splits ensuring each category is represented in each split"""
        print(f"\n=== CREATING STRATIFIED SPLITS ===")
        
        # Filter out 'Animal' category
        target_categories = {cid: cname for cid, cname in categories.items() 
                           if cname.lower() != 'animal'}
        
        # Group videos by their primary categories
        category_videos = defaultdict(list)
        
        for video_stat in video_stats:
            video_categories = video_stat['categories']
            
            # Find the category with most annotations for this video
            if video_categories:
                # Exclude 'Animal' category (id=1 typically)
                filtered_cats = {cid: count for cid, count in video_categories.items() 
                               if cid in target_categories}
                
                if filtered_cats:
                    primary_cat = max(filtered_cats.items(), key=lambda x: x[1])[0]
                    category_videos[primary_cat].append(video_stat)
        
        print(f"Videos grouped by primary category:")
        for cat_id, videos in category_videos.items():
            cat_name = categories[cat_id]
            total_annotations = sum(v['num_annotations'] for v in videos)
            print(f"  {cat_name}: {len(videos)} videos, {total_annotations} annotations")
        
        # Split each category proportionally
        splits = {'train': [], 'val': [], 'test': []}
        
        for cat_id, videos in category_videos.items():
            cat_name = categories[cat_id]
            
            # Shuffle videos for random split
            random.shuffle(videos)
            
            # Calculate split indices
            n_videos = len(videos)
            train_end = int(n_videos * self.train_ratio)
            val_end = train_end + int(n_videos * self.val_ratio)
            
            # Split videos
            train_videos = videos[:train_end]
            val_videos = videos[train_end:val_end]
            test_videos = videos[val_end:]
            
            splits['train'].extend([v['video_id'] for v in train_videos])
            splits['val'].extend([v['video_id'] for v in val_videos])
            splits['test'].extend([v['video_id'] for v in test_videos])
            
            print(f"  {cat_name}: {len(train_videos)} train, {len(val_videos)} val, {len(test_videos)} test videos")
        
        print(f"\nFinal split sizes:")
        print(f"  Train: {len(splits['train'])} videos")
        print(f"  Val: {len(splits['val'])} videos")
        print(f"  Test: {len(splits['test'])} videos")
        
        return splits
    
    def create_split_annotations(self, video_annotations, splits, coco_data):
        """Create separate annotation files for each split"""
        print(f"\n=== CREATING SPLIT ANNOTATIONS ===")
        
        # Base structure for each split
        base_structure = {
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', []),
            'categories': coco_data['categories']
        }
        
        split_data = {}
        
        for split_name, video_ids in splits.items():
            split_annotations = []
            split_images = []
            image_id_mapping = {}  # old_id -> new_id
            
            new_ann_id = 1
            new_img_id = 1
            
            # Collect all annotations for this split
            for video_id in video_ids:
                for ann in video_annotations[video_id]:
                    # Map image if not already mapped
                    old_image_id = ann['image_id']
                    if old_image_id not in image_id_mapping:
                        # Find original image info
                        original_img = next(img for img in coco_data['images'] 
                                          if img['id'] == old_image_id)
                        
                        # Create new image entry
                        new_img = original_img.copy()
                        new_img['id'] = new_img_id
                        split_images.append(new_img)
                        
                        image_id_mapping[old_image_id] = new_img_id
                        new_img_id += 1
                    
                    # Create new annotation
                    new_ann = ann.copy()
                    new_ann['id'] = new_ann_id
                    new_ann['image_id'] = image_id_mapping[old_image_id]
                    split_annotations.append(new_ann)
                    new_ann_id += 1
            
            # Create split dataset
            split_data[split_name] = {
                **base_structure,
                'images': split_images,
                'annotations': split_annotations
            }
            
            print(f"  {split_name}: {len(split_images)} images, {len(split_annotations)} annotations")
        
        return split_data
    
    def crop_and_organize_images(self, split_data):
        """Crop bboxes and organize into classification dataset structure"""
        print(f"\n=== CROPPING AND ORGANIZING IMAGES ===")
        
        # Create output directory structure
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get category info
        categories = {cat['id']: cat['name'] for cat in split_data['train']['categories']}
        
        total_cropped = 0
        split_stats = {}
        
        for split_name, data in split_data.items():
            print(f"\nProcessing {split_name} split...")
            
            split_dir = os.path.join(self.output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            # Create category directories
            for cat_name in categories.values():
                if cat_name.lower() != 'animal':  # Skip 'Animal' category
                    os.makedirs(os.path.join(split_dir, cat_name), exist_ok=True)
            
            # Process annotations
            category_counts = Counter()
            
            for ann in data['annotations']:
                try:
                    # Skip 'Animal' category
                    if categories[ann['category_id']].lower() == 'animal':
                        continue
                    
                    # Find corresponding image
                    image_info = next(img for img in data['images'] 
                                    if img['id'] == ann['image_id'])
                    
                    # Load original image
                    image_path = os.path.join(self.images_dir, image_info['file_name'])
                    if not os.path.exists(image_path):
                        print(f"    Warning: Image not found: {image_path}")
                        continue
                    
                    image = Image.open(image_path).convert('RGB')
                    
                    # Extract bbox
                    bbox = ann['bbox']  # [x, y, width, height]
                    x, y, w, h = bbox
                    
                    # Crop image
                    crop = image.crop((x, y, x + w, y + h))
                    
                    # Save crop
                    category_name = categories[ann['category_id']]
                    crop_filename = f"{ann['id']}.jpg"
                    crop_path = os.path.join(split_dir, category_name, crop_filename)
                    crop.save(crop_path)
                    
                    category_counts[category_name] += 1
                    total_cropped += 1
                    
                except Exception as e:
                    print(f"    Error processing annotation {ann['id']}: {e}")
                    continue
            
            # Store split stats
            split_stats[split_name] = dict(category_counts)
            
            print(f"  {split_name} completed: {sum(category_counts.values())} crops")
            for cat_name, count in category_counts.items():
                print(f"    {cat_name}: {count}")
        
        print(f"\nTotal cropped images: {total_cropped}")
        
        return split_stats
    
    def run_video_aware_split(self):
        """Run the complete video-aware splitting process"""
        print(f"Starting video-aware dataset splitting...")
        print(f"Input COCO file: {self.coco_file}")
        print(f"Input images: {self.images_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Load COCO data
        print(f"\n=== LOADING COCO DATA ===")
        with open(self.coco_file, 'r') as f:
            coco_data = json.load(f)
        
        print(f"Loaded {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
        
        # Analyze video distribution
        video_annotations, video_images, video_stats = self.analyze_video_distribution(coco_data)
        
        # Create stratified splits
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        splits = self.create_stratified_splits(video_stats, categories)
        
        # Create split annotations
        split_data = self.create_split_annotations(video_annotations, splits, coco_data)
        
        # Save split annotation files
        annotations_dir = os.path.join(self.output_dir, 'annotations')
        os.makedirs(annotations_dir, exist_ok=True)
        
        for split_name, data in split_data.items():
            output_file = os.path.join(annotations_dir, f'{split_name}_annotations.json')
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {split_name} annotations to {output_file}")
        
        # Crop and organize images
        split_stats = self.crop_and_organize_images(split_data)
        
        # Create summary
        self.create_summary_report(split_stats, video_stats, len(coco_data['annotations']))
        
        return split_stats
    
    def create_summary_report(self, split_stats, video_stats, original_annotations):
        """Create summary report of the splitting process"""
        summary = {
            'original_dataset': {
                'total_annotations': original_annotations,
                'total_videos': len(video_stats)
            },
            'new_dataset': {
                'output_directory': self.output_dir,
                'split_ratios': {
                    'train': self.train_ratio,
                    'val': self.val_ratio, 
                    'test': self.test_ratio
                },
                'splits': split_stats
            },
            'improvements': [
                "Eliminated temporal data leakage",
                "Proper video-aware train/test separation", 
                "Stratified splitting by category",
                "Ready for realistic evaluation"
            ]
        }
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'split_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n=== SUMMARY ===")
        print(f"Video-aware dataset created: {self.output_dir}")
        print(f"Original: {original_annotations} annotations from {len(video_stats)} videos")
        print(f"New dataset:")
        
        total_new = 0
        for split_name, stats in split_stats.items():
            split_total = sum(stats.values())
            total_new += split_total
            print(f"  {split_name}: {split_total} samples")
        
        print(f"Total: {total_new} samples (excluded 'Animal' category)")
        print(f"Ready for training: python src/train_classifier.py --data_dir {self.output_dir}")
        print(f"Expected accuracy drop: ~15-20% (but now it's REAL performance!)")


def main():
    parser = argparse.ArgumentParser(description='Create video-aware train/test splits to eliminate temporal data leakage')
    parser.add_argument('--coco_file', 
                        default='~/datasets/wildlife_coco_dataset/annotations.json',
                        help='Path to COCO annotations file')
    parser.add_argument('--images_dir', 
                        default='~/datasets/wildlife_coco_dataset/images',
                        help='Path to images directory')
    parser.add_argument('--output_dir', 
                        default='~/datasets/wildlife_classification_video_aware',
                        help='Output directory for video-aware dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Proportion for training set')
    parser.add_argument('--val_ratio', type=float, default=0.2, 
                        help='Proportion for validation set')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Proportion for test set')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits')
    
    args = parser.parse_args()
    
    # Expand paths
    args.coco_file = os.path.expanduser(args.coco_file)
    args.images_dir = os.path.expanduser(args.images_dir)
    args.output_dir = os.path.expanduser(args.output_dir)
    
    # Set random seed
    random.seed(args.seed)
    
    # Verify inputs
    if not os.path.exists(args.coco_file):
        print(f"COCO file not found: {args.coco_file}")
        return
        
    if not os.path.exists(args.images_dir):
        print(f"Images directory not found: {args.images_dir}")
        return
    
    # Create splitter and run
    splitter = VideoAwareDatasetSplitter(
        coco_file=args.coco_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    split_stats = splitter.run_video_aware_split()
    
    print(f"\nNext steps:")
    print(f"1. Train new model: python src/train_classifier.py --data_dir {args.output_dir}")
    print(f"2. Evaluate performance: python src/evaluate_classifier.py --model_path [new_model] --data_dir {args.output_dir}")
    print(f"3. Compare with old results (expect 15-20% accuracy drop, but it's REAL!)")


if __name__ == "__main__":
    main()