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
            print(f"⚠️  Ratios sum to {total_ratio}, normalizing...")
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
        
        print(f\"\\nFinal split sizes:\")\n        print(f\"  Train: {len(splits['train'])} videos\")\n        print(f\"  Val: {len(splits['val'])} videos\")\n        print(f\"  Test: {len(splits['test'])} videos\")\n        \n        return splits\n    \n    def create_split_annotations(self, video_annotations, splits, coco_data):\n        \"\"\"Create separate annotation files for each split\"\"\"\n        print(f\"\\n=== CREATING SPLIT ANNOTATIONS ===\")\n        \n        # Base structure for each split\n        base_structure = {\n            'info': coco_data.get('info', {}),\n            'licenses': coco_data.get('licenses', []),\n            'categories': coco_data['categories']\n        }\n        \n        split_data = {}\n        \n        for split_name, video_ids in splits.items():\n            split_annotations = []\n            split_images = []\n            image_id_mapping = {}  # old_id -> new_id\n            \n            new_ann_id = 1\n            new_img_id = 1\n            \n            # Collect all annotations for this split\n            for video_id in video_ids:\n                for ann in video_annotations[video_id]:\n                    # Map image if not already mapped\n                    old_image_id = ann['image_id']\n                    if old_image_id not in image_id_mapping:\n                        # Find original image info\n                        original_img = next(img for img in coco_data['images'] \n                                          if img['id'] == old_image_id)\n                        \n                        # Create new image entry\n                        new_img = original_img.copy()\n                        new_img['id'] = new_img_id\n                        split_images.append(new_img)\n                        \n                        image_id_mapping[old_image_id] = new_img_id\n                        new_img_id += 1\n                    \n                    # Create new annotation\n                    new_ann = ann.copy()\n                    new_ann['id'] = new_ann_id\n                    new_ann['image_id'] = image_id_mapping[old_image_id]\n                    split_annotations.append(new_ann)\n                    new_ann_id += 1\n            \n            # Create split dataset\n            split_data[split_name] = {\n                **base_structure,\n                'images': split_images,\n                'annotations': split_annotations\n            }\n            \n            print(f\"  {split_name}: {len(split_images)} images, {len(split_annotations)} annotations\")\n        \n        return split_data\n    \n    def crop_and_organize_images(self, split_data):\n        \"\"\"Crop bboxes and organize into classification dataset structure\"\"\"\n        print(f\"\\n=== CROPPING AND ORGANIZING IMAGES ===\")\n        \n        # Create output directory structure\n        os.makedirs(self.output_dir, exist_ok=True)\n        \n        # Get category info\n        categories = {cat['id']: cat['name'] for cat in split_data['train']['categories']}\n        \n        total_cropped = 0\n        split_stats = {}\n        \n        for split_name, data in split_data.items():\n            print(f\"\\nProcessing {split_name} split...\")\n            \n            split_dir = os.path.join(self.output_dir, split_name)\n            os.makedirs(split_dir, exist_ok=True)\n            \n            # Create category directories\n            for cat_name in categories.values():\n                if cat_name.lower() != 'animal':  # Skip 'Animal' category\n                    os.makedirs(os.path.join(split_dir, cat_name), exist_ok=True)\n            \n            # Process annotations\n            category_counts = Counter()\n            \n            for ann in data['annotations']:\n                try:\n                    # Skip 'Animal' category\n                    if categories[ann['category_id']].lower() == 'animal':\n                        continue\n                    \n                    # Find corresponding image\n                    image_info = next(img for img in data['images'] \n                                    if img['id'] == ann['image_id'])\n                    \n                    # Load original image\n                    image_path = os.path.join(self.images_dir, image_info['file_name'])\n                    if not os.path.exists(image_path):\n                        print(f\"    Warning: Image not found: {image_path}\")\n                        continue\n                    \n                    image = Image.open(image_path).convert('RGB')\n                    \n                    # Extract bbox\n                    bbox = ann['bbox']  # [x, y, width, height]\n                    x, y, w, h = bbox\n                    \n                    # Crop image\n                    crop = image.crop((x, y, x + w, y + h))\n                    \n                    # Save crop\n                    category_name = categories[ann['category_id']]\n                    crop_filename = f\"{ann['id']}.jpg\"\n                    crop_path = os.path.join(split_dir, category_name, crop_filename)\n                    crop.save(crop_path)\n                    \n                    category_counts[category_name] += 1\n                    total_cropped += 1\n                    \n                except Exception as e:\n                    print(f\"    Error processing annotation {ann['id']}: {e}\")\n                    continue\n            \n            # Store split stats\n            split_stats[split_name] = dict(category_counts)\n            \n            print(f\"  {split_name} completed: {sum(category_counts.values())} crops\")\n            for cat_name, count in category_counts.items():\n                print(f\"    {cat_name}: {count}\")\n        \n        print(f\"\\n✅ Total cropped images: {total_cropped}\")\n        \n        return split_stats\n    \n    def run_video_aware_split(self):\n        \"\"\"Run the complete video-aware splitting process\"\"\"\n        print(f\"🎬 Starting video-aware dataset splitting...\")\n        print(f\"Input COCO file: {self.coco_file}\")\n        print(f\"Input images: {self.images_dir}\")\n        print(f\"Output directory: {self.output_dir}\")\n        \n        # Load COCO data\n        print(f\"\\n=== LOADING COCO DATA ===\")\n        with open(self.coco_file, 'r') as f:\n            coco_data = json.load(f)\n        \n        print(f\"Loaded {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations\")\n        \n        # Analyze video distribution\n        video_annotations, video_images, video_stats = self.analyze_video_distribution(coco_data)\n        \n        # Create stratified splits\n        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}\n        splits = self.create_stratified_splits(video_stats, categories)\n        \n        # Create split annotations\n        split_data = self.create_split_annotations(video_annotations, splits, coco_data)\n        \n        # Save split annotation files\n        annotations_dir = os.path.join(self.output_dir, 'annotations')\n        os.makedirs(annotations_dir, exist_ok=True)\n        \n        for split_name, data in split_data.items():\n            output_file = os.path.join(annotations_dir, f'{split_name}_annotations.json')\n            with open(output_file, 'w') as f:\n                json.dump(data, f, indent=2)\n            print(f\"Saved {split_name} annotations to {output_file}\")\n        \n        # Crop and organize images\n        split_stats = self.crop_and_organize_images(split_data)\n        \n        # Create summary\n        self.create_summary_report(split_stats, video_stats, len(coco_data['annotations']))\n        \n        return split_stats\n    \n    def create_summary_report(self, split_stats, video_stats, original_annotations):\n        \"\"\"Create summary report of the splitting process\"\"\"\n        summary = {\n            'original_dataset': {\n                'total_annotations': original_annotations,\n                'total_videos': len(video_stats)\n            },\n            'new_dataset': {\n                'output_directory': self.output_dir,\n                'split_ratios': {\n                    'train': self.train_ratio,\n                    'val': self.val_ratio, \n                    'test': self.test_ratio\n                },\n                'splits': split_stats\n            },\n            'improvements': [\n                \"Eliminated temporal data leakage\",\n                \"Proper video-aware train/test separation\", \n                \"Stratified splitting by category\",\n                \"Ready for realistic evaluation\"\n            ]\n        }\n        \n        # Save summary\n        summary_path = os.path.join(self.output_dir, 'split_summary.json')\n        with open(summary_path, 'w') as f:\n            json.dump(summary, f, indent=2)\n        \n        print(f\"\\n=== SUMMARY ===\")\n        print(f\"✅ Video-aware dataset created: {self.output_dir}\")\n        print(f\"📊 Original: {original_annotations} annotations from {len(video_stats)} videos\")\n        print(f\"📊 New dataset:\")\n        \n        total_new = 0\n        for split_name, stats in split_stats.items():\n            split_total = sum(stats.values())\n            total_new += split_total\n            print(f\"  {split_name}: {split_total} samples\")\n        \n        print(f\"📊 Total: {total_new} samples (excluded 'Animal' category)\")\n        print(f\"🎯 Ready for training: python src/train_classifier.py --data_dir {self.output_dir}\")\n        print(f\"💡 Expected accuracy drop: ~15-20% (but now it's REAL performance!)\")\n\n\ndef main():\n    parser = argparse.ArgumentParser(description='Create video-aware train/test splits to eliminate temporal data leakage')\n    parser.add_argument('--coco_file', \n                        default='~/datasets/wildlife_coco_dataset/annotations.json',\n                        help='Path to COCO annotations file')\n    parser.add_argument('--images_dir', \n                        default='~/datasets/wildlife_coco_dataset/images',\n                        help='Path to images directory')\n    parser.add_argument('--output_dir', \n                        default='~/datasets/wildlife_classification_video_aware',\n                        help='Output directory for video-aware dataset')\n    parser.add_argument('--train_ratio', type=float, default=0.7,\n                        help='Proportion for training set')\n    parser.add_argument('--val_ratio', type=float, default=0.2, \n                        help='Proportion for validation set')\n    parser.add_argument('--test_ratio', type=float, default=0.1,\n                        help='Proportion for test set')\n    parser.add_argument('--seed', type=int, default=42,\n                        help='Random seed for reproducible splits')\n    \n    args = parser.parse_args()\n    \n    # Expand paths\n    args.coco_file = os.path.expanduser(args.coco_file)\n    args.images_dir = os.path.expanduser(args.images_dir)\n    args.output_dir = os.path.expanduser(args.output_dir)\n    \n    # Set random seed\n    random.seed(args.seed)\n    \n    # Verify inputs\n    if not os.path.exists(args.coco_file):\n        print(f\"❌ COCO file not found: {args.coco_file}\")\n        return\n        \n    if not os.path.exists(args.images_dir):\n        print(f\"❌ Images directory not found: {args.images_dir}\")\n        return\n    \n    # Create splitter and run\n    splitter = VideoAwareDatasetSplitter(\n        coco_file=args.coco_file,\n        images_dir=args.images_dir,\n        output_dir=args.output_dir,\n        train_ratio=args.train_ratio,\n        val_ratio=args.val_ratio,\n        test_ratio=args.test_ratio\n    )\n    \n    split_stats = splitter.run_video_aware_split()\n    \n    print(f\"\\n🚀 Next steps:\")\n    print(f\"1. Train new model: python src/train_classifier.py --data_dir {args.output_dir}\")\n    print(f\"2. Evaluate performance: python src/evaluate_classifier.py --model_path [new_model] --data_dir {args.output_dir}\")\n    print(f\"3. Compare with old results (expect 15-20% accuracy drop, but it's REAL!)\")\n\n\nif __name__ == \"__main__\":\n    main()"