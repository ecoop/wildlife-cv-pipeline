import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import numpy as np
import json
import os
import time
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, confusion_matrix
import argparse


class ClassifierTrainer:
    def __init__(self, data_dir, backbone='resnet18', experiment_name='experiment', 
                 batch_size=32, learning_rate=0.001, num_epochs=50):
        
        # Device setup - works on both x86 MBP and M2 Studio
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🚀 Using Apple Silicon MPS")
        else:
            self.device = torch.device("cpu") 
            print("💻 Using CPU")
        
        self.data_dir = data_dir
        self.backbone = backbone
        self.experiment_name = experiment_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Create results directory
        self.results_dir = f"training_results/{experiment_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.setup_data()
        self.setup_model()
        
    def setup_data(self):
        """Setup data loaders with class balancing"""
        print("\n=== SETTING UP DATA ===")
        
        # Load datasets
        train_dataset = ImageFolder(
            os.path.join(self.data_dir, 'train'),
            transform=self.train_transform
        )
        
        val_dataset = ImageFolder(
            os.path.join(self.data_dir, 'val'),
            transform=self.val_transform
        )
        
        self.num_classes = len(train_dataset.classes)
        self.class_names = train_dataset.classes
        
        print(f"Classes: {self.class_names}")
        print(f"Number of classes: {self.num_classes}")
        
        # Calculate class weights for balancing
        class_counts = Counter(train_dataset.targets)
        total_samples = len(train_dataset)
        
        print(f"\nTraining set distribution:")
        for i, class_name in enumerate(self.class_names):
            count = class_counts[i]
            percentage = (count / total_samples) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Create weighted sampler for class balancing
        class_weights = []
        for i in range(self.num_classes):
            weight = total_samples / (self.num_classes * class_counts[i])
            class_weights.append(weight)
        
        sample_weights = [class_weights[target] for target in train_dataset.targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        pin_memory = self.device.type != 'mps' # turn off pin memory for Apple Silicon MPS

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        
    def setup_model(self):
        """Setup model, loss, and optimizer"""
        print(f"\n=== SETTING UP MODEL: {self.backbone} ===")
        
        # Create model using timm (easy backbone swapping)
        self.model = timm.create_model(
            self.backbone,
            pretrained=True,
            num_classes=self.num_classes
        )
        
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
            # Progress update
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_predictions
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = np.mean(np.array(all_predictions) == np.array(all_targets))
        
        return epoch_loss, epoch_acc, all_predictions, all_targets
    
    def train(self):
        """Main training loop"""
        print(f"\n=== STARTING TRAINING ===")
        print(f"Experiment: {self.experiment_name}")
        print(f"Backbone: {self.backbone}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_predictions, val_targets = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rates'].append(current_lr)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model.pth')
                print(f"🎉 New best validation accuracy: {val_acc:.4f}")
                
                # Save detailed classification report for best model
                self.save_classification_report(val_predictions, val_targets)
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
        
        total_time = time.time() - start_time
        print(f"\n🏁 Training completed in {total_time/60:.1f} minutes")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        # Save final results
        self.save_history(history)
        self.save_model('final_model.pth')
        
        return history, best_val_acc
    
    def save_model(self, filename):
        """Save model checkpoint"""
        filepath = os.path.join(self.results_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'backbone': self.backbone,
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }, filepath)
    
    def save_history(self, history):
        """Save training history"""
        filepath = os.path.join(self.results_dir, 'training_history.json')
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
    
    def save_classification_report(self, predictions, targets):
        """Save detailed classification report"""
        report = classification_report(
            targets, predictions, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        # Save as JSON
        filepath = os.path.join(self.results_dir, 'classification_report.json')
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save as readable text
        text_report = classification_report(
            targets, predictions, 
            target_names=self.class_names
        )
        
        filepath = os.path.join(self.results_dir, 'classification_report.txt')
        with open(filepath, 'w') as f:
            f.write(text_report)
        
        # Print summary
        print("\nPer-class results:")
        for class_name in self.class_names:
            if class_name in report:
                f1 = report[class_name]['f1-score']
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                print(f"  {class_name}: F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Train wildlife classifier')
    parser.add_argument('--data_dir', default='wildlife_coco_dataset/cropped_images', 
                        help='Path to cropped images directory')
    parser.add_argument('--backbone', default='resnet18', 
                        help='Model backbone (resnet18, resnet50, efficientnet_b0, etc.)')
    parser.add_argument('--experiment_name', default='resnet18_v1',
                        help='Experiment name for saving results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        print("Make sure you've run bbox_cropper.py first!")
        return
    
    # Create trainer
    trainer = ClassifierTrainer(
        data_dir=args.data_dir,
        backbone=args.backbone,
        experiment_name=args.experiment_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs
    )
    
    # Train
    history, best_acc = trainer.train()
    
    print(f"\n✅ Training complete!")
    print(f"Results saved to: {trainer.results_dir}")
    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()