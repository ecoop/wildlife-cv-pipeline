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
        
        print(f"\nFinal split sizes:")
        print(f"  Train: {len(splits['train'])} videos")
        print(f"  Val: {len(splits['val'])} videos")
        print(f"  Test: {len(splits['test'])} videos")
        
        return splits