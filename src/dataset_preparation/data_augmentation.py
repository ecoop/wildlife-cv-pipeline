import os
import random
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as transforms
import numpy as np
from collections import Counter
import argparse


class AdvancedAugmenter:
    """Advanced augmentation specifically for underrepresented classes"""
    
    def __init__(self, target_samples_per_class=1500):
        self.target_samples = target_samples_per_class
        
        # Heavy augmentation for small classes
        self.heavy_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=20),
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ])
        
        # Moderate augmentation for medium classes
        self.moderate_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ])
        
        # Light augmentation for large classes
        self.light_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])
    
    def analyze_class_distribution(self, data_dir):
        """Analyze current class distribution"""
        print("=== ANALYZING CLASS DISTRIBUTION ===")
        
        train_dir = os.path.join(data_dir, 'train')
        class_counts = {}
        
        for class_name in os.listdir(train_dir):
            class_path = os.path.join(train_dir, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                class_counts[class_name] = count
        
        # Sort by count
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
        
        print(f"Current distribution:")
        for class_name, count in sorted_classes:
            print(f"  {class_name}: {count} samples")
        
        print(f"\nTarget: {self.target_samples} samples per class")
        
        return class_counts
    
    def determine_augmentation_strategy(self, class_counts):
        """Determine augmentation strategy based on class sizes"""
        strategies = {}
        
        for class_name, count in class_counts.items():
            if count < 300:
                # Very small classes - heavy augmentation
                multiplier = max(2, self.target_samples // count)
                strategies[class_name] = {
                    'strategy': 'heavy',
                    'multiplier': min(multiplier, 8),  # Cap at 8x
                    'transforms': self.heavy_transforms
                }
            elif count < 800:
                # Medium classes - moderate augmentation  
                multiplier = max(1, self.target_samples // count)
                strategies[class_name] = {
                    'strategy': 'moderate', 
                    'multiplier': min(multiplier, 4),  # Cap at 4x
                    'transforms': self.moderate_transforms
                }
            else:
                # Large classes - light augmentation or none
                strategies[class_name] = {
                    'strategy': 'light',
                    'multiplier': 1,
                    'transforms': self.light_transforms
                }
        
        return strategies
    
    def apply_custom_augmentations(self, image):
        """Apply additional custom augmentations"""
        augmented_images = []
        
        # Original
        augmented_images.append(image)
        
        # Noise addition
        img_array = np.array(image)
        noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
        noisy_image = Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8))
        augmented_images.append(noisy_image)
        
        # Gaussian blur
        blurred = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        augmented_images.append(blurred)
        
        # Brightness variations
        enhancer = ImageEnhance.Brightness(image)
        bright_image = enhancer.enhance(1.3)
        dark_image = enhancer.enhance(0.7)
        augmented_images.extend([bright_image, dark_image])
        
        # Contrast variations
        enhancer = ImageEnhance.Contrast(image)
        high_contrast = enhancer.enhance(1.3)
        low_contrast = enhancer.enhance(0.7)
        augmented_images.extend([high_contrast, low_contrast])
        
        return augmented_images
    
    def augment_class(self, class_dir, output_dir, strategy_info):
        """Augment a specific class"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all images in class
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"  Processing {len(image_files)} images with {strategy_info['strategy']} augmentation...")
        
        generated_count = 0
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Load image
                image = Image.open(img_path).convert('RGB')
                
                # Copy original
                base_name = os.path.splitext(img_file)[0]
                image.save(os.path.join(output_dir, f"{base_name}_orig.jpg"))
                generated_count += 1
                
                # Generate augmented versions
                for aug_idx in range(strategy_info['multiplier'] - 1):
                    # Apply transforms
                    if strategy_info['transforms']:
                        augmented = strategy_info['transforms'](image)
                        augmented.save(os.path.join(output_dir, f"{base_name}_aug_{aug_idx}.jpg"))
                        generated_count += 1
                    
                    # For very small classes, add custom augmentations
                    if strategy_info['strategy'] == 'heavy' and aug_idx < 2:
                        custom_augs = self.apply_custom_augmentations(image)
                        for i, custom_aug in enumerate(custom_augs[1:3]):  # Take 2 custom augs
                            custom_aug.save(os.path.join(output_dir, f"{base_name}_custom_{aug_idx}_{i}.jpg"))
                            generated_count += 1
                            
            except Exception as e:
                print(f"    Error processing {img_file}: {e}")
                continue
        
        print(f"    Generated {generated_count} total images")
        return generated_count
    
    def create_balanced_dataset(self, input_dir, output_dir):
        """Create balanced dataset with augmentation"""
        print("=== CREATING BALANCED DATASET ===")
        
        # Analyze current distribution
        class_counts = self.analyze_class_distribution(input_dir)
        
        # Determine strategies
        strategies = self.determine_augmentation_strategy(class_counts)
        
        print(f"\nAugmentation strategies:")
        for class_name, strategy_info in strategies.items():
            current_count = class_counts[class_name]
            expected_count = current_count * strategy_info['multiplier']
            print(f"  {class_name}: {strategy_info['strategy']} ({current_count} → ~{expected_count})")
        
        # Create output structure
        os.makedirs(output_dir, exist_ok=True)
        train_input = os.path.join(input_dir, 'train')
        train_output = os.path.join(output_dir, 'train')
        
        # Process each class
        final_counts = {}
        for class_name in class_counts.keys():
            print(f"\nProcessing class: {class_name}")
            
            class_input_dir = os.path.join(train_input, class_name)
            class_output_dir = os.path.join(train_output, class_name)
            
            count = self.augment_class(class_input_dir, class_output_dir, strategies[class_name])
            final_counts[class_name] = count
        
        # Copy val and test unchanged
        for split in ['val', 'test']:
            split_input = os.path.join(input_dir, split)
            split_output = os.path.join(output_dir, split)
            
            if os.path.exists(split_input):
                print(f"\nCopying {split} split...")
                os.system(f"cp -r {split_input} {split_output}")
        
        print(f"\n=== BALANCED DATASET CREATED ===")
        print(f"Output: {output_dir}")
        print(f"Final training distribution:")
        for class_name, count in sorted(final_counts.items(), key=lambda x: x[1]):
            print(f"  {class_name}: {count} samples")
        
        return final_counts


def main():
    parser = argparse.ArgumentParser(description='Create balanced dataset with class-specific augmentation')
    parser.add_argument('--input_dir', default='~/datasets/wildlife_classification_dataset',
                        help='Input dataset directory')
    parser.add_argument('--output_dir', default='~/datasets/wildlife_classification_balanced',
                        help='Output balanced dataset directory')
    parser.add_argument('--target_samples', type=int, default=1500,
                        help='Target number of samples per class')
    
    args = parser.parse_args()
    
    # Expand paths
    args.input_dir = os.path.expanduser(args.input_dir)
    args.output_dir = os.path.expanduser(args.output_dir)
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    # Create augmenter
    augmenter = AdvancedAugmenter(target_samples_per_class=args.target_samples)
    
    # Create balanced dataset
    final_counts = augmenter.create_balanced_dataset(args.input_dir, args.output_dir)
    
    print(f"\n✅ Balanced dataset ready!")
    print(f"📁 Location: {args.output_dir}")
    print(f"🎯 Train with: python src/train_classifier.py --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()