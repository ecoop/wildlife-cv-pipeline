import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import json
import os
import argparse
from PIL import Image
import pandas as pd
from collections import Counter


class ClassifierEvaluator:
    def __init__(self, model_path, data_dir, results_dir="evaluation_results"):
        self.model_path = model_path
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🚀 Using Apple Silicon MPS")
        else:
            self.device = torch.device("cpu")
            print("💻 Using CPU")
        
        # Load model and setup data
        self.load_model()
        self.setup_data()
        
    def load_model(self):
        """Load trained model from checkpoint"""
        print(f"\n=== LOADING MODEL ===")
        print(f"Model path: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model info
        self.backbone = checkpoint['backbone']
        self.num_classes = checkpoint['num_classes']
        self.class_names = checkpoint['class_names']
        
        print(f"Backbone: {self.backbone}")
        print(f"Classes: {self.class_names}")
        
        # Create and load model
        self.model = timm.create_model(
            self.backbone,
            pretrained=False,  # Don't need pretrained weights
            num_classes=self.num_classes
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def setup_data(self):
        """Setup data loaders for evaluation"""
        print(f"\n=== SETTING UP DATA ===")
        
        # Transform (same as validation)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load test dataset
        self.test_dataset = ImageFolder(
            os.path.join(self.data_dir, 'test'),
            transform=self.transform
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2
        )
        
        print(f"Test images: {len(self.test_dataset)}")
        
        # Verify class alignment
        dataset_classes = self.test_dataset.classes
        if dataset_classes != self.class_names:
            print("⚠️  Warning: Class mismatch between model and dataset")
            print(f"Model classes: {self.class_names}")
            print(f"Dataset classes: {dataset_classes}")
        
    def evaluate_model(self):
        """Run full evaluation on test set"""
        print(f"\n=== EVALUATING MODEL ===")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_image_paths = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Get image paths for this batch
                start_idx = batch_idx * self.test_loader.batch_size
                end_idx = start_idx + inputs.size(0)
                batch_paths = [self.test_dataset.samples[i][0] for i in range(start_idx, min(end_idx, len(self.test_dataset)))]
                all_image_paths.extend(batch_paths)
                
                if batch_idx % 10 == 0:
                    print(f"  Processed {batch_idx * self.test_loader.batch_size}/{len(self.test_dataset)} images")
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = np.mean(predictions == targets)
        print(f"\n📊 Overall Test Accuracy: {accuracy:.4f}")
        
        return {
            'predictions': predictions,
            'targets': targets,
            'probabilities': probabilities,
            'image_paths': all_image_paths,
            'accuracy': accuracy
        }
    
    def create_confusion_matrix(self, results):
        """Create and save confusion matrix visualization"""
        print(f"\n=== CREATING CONFUSION MATRIX ===")
        
        cm = confusion_matrix(results['targets'], results['predictions'])
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Number of Predictions'})
        
        plt.title(f'Confusion Matrix - {self.backbone}\nOverall Accuracy: {results["accuracy"]:.3f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, 
                    annot=True, 
                    fmt='.3f', 
                    cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Proportion'})
        
        plt.title(f'Normalized Confusion Matrix - {self.backbone}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm, cm_norm
    
    def analyze_per_class_performance(self, results):
        """Analyze and visualize per-class performance"""
        print(f"\n=== PER-CLASS ANALYSIS ===")
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            results['targets'], results['predictions'], average=None
        )
        
        # Create DataFrame for easy visualization
        metrics_df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        print(metrics_df.round(3))
        
        # Save to CSV
        metrics_df.to_csv(os.path.join(self.results_dir, 'per_class_metrics.csv'), index=False)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Per-Class Performance Analysis - {self.backbone}', fontsize=16)
        
        # Precision
        axes[0,0].bar(self.class_names, precision)
        axes[0,0].set_title('Precision by Class')
        axes[0,0].set_ylim(0, 1)
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Recall
        axes[0,1].bar(self.class_names, recall)
        axes[0,1].set_title('Recall by Class')
        axes[0,1].set_ylim(0, 1)
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # F1-Score
        axes[1,0].bar(self.class_names, f1)
        axes[1,0].set_title('F1-Score by Class')
        axes[1,0].set_ylim(0, 1)
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Support (sample size)
        axes[1,1].bar(self.class_names, support)
        axes[1,1].set_title('Test Samples by Class')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'per_class_performance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return metrics_df
    
    def analyze_confidence_distribution(self, results):
        """Analyze confidence distributions"""
        print(f"\n=== CONFIDENCE ANALYSIS ===")
        
        # Get max probabilities (confidence scores)
        confidences = np.max(results['probabilities'], axis=1)
        predicted_classes = results['predictions']
        
        # Overall confidence distribution
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Overall confidence histogram
        plt.subplot(2, 2, 1)
        plt.hist(confidences, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Overall Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
        plt.legend()
        
        # Plot 2: Confidence by correctness
        plt.subplot(2, 2, 2)
        correct_mask = results['predictions'] == results['targets']
        correct_conf = confidences[correct_mask]
        incorrect_conf = confidences[~correct_mask]
        
        plt.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green', edgecolor='black')
        plt.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
        plt.title('Confidence: Correct vs Incorrect')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Plot 3: Confidence by class
        plt.subplot(2, 1, 2)
        confidence_by_class = []
        for i, class_name in enumerate(self.class_names):
            class_confidences = confidences[predicted_classes == i]
            confidence_by_class.append(class_confidences)
        
        plt.boxplot(confidence_by_class, labels=self.class_names)
        plt.title('Confidence Distribution by Predicted Class')
        plt.xlabel('Class')
        plt.ylabel('Confidence Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confidence_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'mean_confidence': np.mean(confidences),
            'correct_confidence': np.mean(correct_conf) if len(correct_conf) > 0 else 0,
            'incorrect_confidence': np.mean(incorrect_conf) if len(incorrect_conf) > 0 else 0
        }
    
    def find_misclassified_examples(self, results, num_examples=5):
        """Find and display worst misclassified examples"""
        print(f"\n=== MISCLASSIFIED EXAMPLES ===")
        
        # Find incorrect predictions
        incorrect_mask = results['predictions'] != results['targets']
        
        if not np.any(incorrect_mask):
            print("🎉 Perfect accuracy! No misclassified examples.")
            return
        
        # Get confidence scores for incorrect predictions
        incorrect_confidences = np.max(results['probabilities'][incorrect_mask], axis=1)
        incorrect_indices = np.where(incorrect_mask)[0]
        
        # Sort by confidence (highest confidence wrong predictions are most interesting)
        sorted_indices = np.argsort(-incorrect_confidences)
        
        # Create figure for top misclassifications
        fig, axes = plt.subplots(2, min(num_examples, 5), figsize=(15, 8))
        fig.suptitle('Most Confident Misclassifications', fontsize=16)
        
        if axes.ndim == 1:  # Handle case with only one row
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_examples, len(sorted_indices), 10)):
            if i >= axes.shape[1]:
                break
                
            # Get the actual index in the full results
            result_idx = incorrect_indices[sorted_indices[i]]
            
            # Load and display image
            img_path = results['image_paths'][result_idx]
            img = Image.open(img_path).convert('RGB')
            
            row = i // axes.shape[1] if i < 10 else 1
            col = i % axes.shape[1]
            
            if row < axes.shape[0]:
                axes[row, col].imshow(img)
                
                # Get prediction info
                true_class = self.class_names[results['targets'][result_idx]]
                pred_class = self.class_names[results['predictions'][result_idx]]
                confidence = incorrect_confidences[sorted_indices[i]]
                
                axes[row, col].set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.3f}', 
                                       fontsize=10)
                axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(num_examples, axes.shape[0] * axes.shape[1]):
            if i < 10:
                row = i // axes.shape[1]
                col = i % axes.shape[1]
                if row < axes.shape[0]:
                    axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'misclassified_examples.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary of misclassifications
        print(f"Total misclassifications: {np.sum(incorrect_mask)}")
        print(f"Misclassification rate: {np.mean(incorrect_mask):.3f}")
        
    def create_comprehensive_report(self, results, metrics_df, confidence_stats):
        """Create comprehensive evaluation report"""
        print(f"\n=== CREATING COMPREHENSIVE REPORT ===")
        
        report = {
            'model_info': {
                'backbone': self.backbone,
                'num_classes': self.num_classes,
                'class_names': self.class_names
            },
            'dataset_info': {
                'test_samples': len(self.test_dataset),
                'samples_per_class': dict(Counter([self.class_names[target] for target in results['targets']]))
            },
            'performance': {
                'overall_accuracy': float(results['accuracy']),
                'per_class_metrics': metrics_df.to_dict('records'),
                'confidence_stats': confidence_stats
            },
            'recommendations': self.generate_recommendations(results, metrics_df)
        }
        
        # Save report
        report_path = os.path.join(self.results_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Comprehensive report saved to: {report_path}")
        
        return report
    
    def generate_recommendations(self, results, metrics_df):
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Check for class imbalance issues
        class_f1_scores = dict(zip(self.class_names, metrics_df['F1-Score']))
        worst_class = min(class_f1_scores.keys(), key=lambda x: class_f1_scores[x])
        best_class = max(class_f1_scores.keys(), key=lambda x: class_f1_scores[x])
        
        if class_f1_scores[worst_class] < 0.7:
            recommendations.append(f"Consider data augmentation for '{worst_class}' (F1: {class_f1_scores[worst_class]:.3f})")
        
        if results['accuracy'] < 0.85:
            recommendations.append("Try larger backbone (ResNet50, EfficientNet-B2) for better accuracy")
        
        # Check confidence calibration
        incorrect_mask = results['predictions'] != results['targets']
        if len(results['probabilities'][incorrect_mask]) > 0:
            high_conf_wrong = np.sum(np.max(results['probabilities'][incorrect_mask], axis=1) > 0.8)
            if high_conf_wrong > 0:
                recommendations.append(f"{high_conf_wrong} high-confidence wrong predictions - consider confidence calibration")
        
        return recommendations
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print(f"🔍 Starting comprehensive evaluation...")
        print(f"Model: {self.model_path}")
        print(f"Data: {self.data_dir}")
        
        # Main evaluation
        results = self.evaluate_model()
        
        # Create visualizations
        cm, cm_norm = self.create_confusion_matrix(results)
        metrics_df = self.analyze_per_class_performance(results)
        confidence_stats = self.analyze_confidence_distribution(results)
        self.find_misclassified_examples(results)
        
        # Comprehensive report
        report = self.create_comprehensive_report(results, metrics_df, confidence_stats)
        
        print(f"\n✅ Evaluation complete!")
        print(f"📁 Results saved to: {self.results_dir}")
        print(f"🎯 Overall accuracy: {results['accuracy']:.4f}")
        
        # Print key recommendations
        if report['recommendations']:
            print(f"\n💡 Key recommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        return results, report


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained wildlife classifier')
    parser.add_argument('--model_path', required=True,
                        help='Path to trained model (.pth file)')
    parser.add_argument('--data_dir', default='~/datasets/wildlife_classification_dataset',
                        help='Path to classification dataset')
    parser.add_argument('--results_dir', default='evaluation_results',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Expand user path
    args.data_dir = os.path.expanduser(args.data_dir)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        return
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    # Create evaluator and run
    evaluator = ClassifierEvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )
    
    results, report = evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()