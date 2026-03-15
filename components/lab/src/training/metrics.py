from typing import Dict, Any, List, Tuple
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class MetricsCalculator:
    """Class for calculating and visualizing evaluation metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the metrics calculator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.metrics = {}
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, probs: torch.Tensor):
        """Update metrics with new batch of predictions.
        
        Args:
            pred: Predicted labels
            target: Ground truth labels
            probs: Predicted probabilities
        """
        self.predictions.extend(pred.cpu().numpy())
        self.targets.extend(target.cpu().numpy())
        self.probabilities.extend(probs.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary containing computed metrics
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)
        
        metrics = {}
        for metric in self.config["evaluation"]["metrics"]:
            if metric == "accuracy":
                metrics[metric] = accuracy_score(targets, predictions)
            elif metric == "precision":
                metrics[metric] = precision_score(targets, predictions, average="binary")
            elif metric == "recall":
                metrics[metric] = recall_score(targets, predictions, average="binary")
            elif metric == "f1_score":
                metrics[metric] = f1_score(targets, predictions, average="binary")
            elif metric == "auc_roc":
                metrics[metric] = roc_auc_score(targets, probabilities[:, 1])
            elif metric == "confusion_matrix":
                metrics[metric] = confusion_matrix(targets, predictions).tolist()
        
        self.metrics = metrics
        return metrics
    
    def visualize(self, save_dir: str):
        """Generate and save visualization plots.
        
        Args:
            save_dir: Directory to save plots
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot confusion matrix
        if "confusion_matrix" in self.metrics:
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                self.metrics["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"]
            )
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.savefig(save_dir / "confusion_matrix.png")
            plt.close()
        
        # Plot ROC curve
        if "auc_roc" in self.metrics:
            fpr, tpr, _ = roc_curve(self.targets, self.probabilities[:, 1])
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"AUC = {self.metrics['auc_roc']:.3f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.savefig(save_dir / "roc_curve.png")
            plt.close()
        
        # Plot precision-recall curve
        precision, recall, _ = precision_recall_curve(self.targets, self.probabilities[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.savefig(save_dir / "precision_recall_curve.png")
        plt.close()
        
        # Save metrics to file
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
    
    def analyze_errors(self, data_loader, model, device: str):
        """Analyze misclassified samples.
        
        Args:
            data_loader: Data loader containing test data
            model: Trained model
            device: Device to run inference on
        """
        if not self.config["evaluation"]["error_analysis"]["enabled"]:
            return
        
        misclassified = []
        confidence_threshold = self.config["evaluation"]["error_analysis"]["confidence_threshold"]
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                # Find misclassified samples
                mask = pred != target
                if mask.any():
                    for i in range(len(mask)):
                        if mask[i]:
                            confidence = probs[i, pred[i]].item()
                            if confidence >= confidence_threshold:
                                misclassified.append({
                                    "sample_idx": batch_idx * data_loader.batch_size + i,
                                    "true_label": target[i].item(),
                                    "predicted_label": pred[i].item(),
                                    "confidence": confidence
                                })
        
        # Save error analysis results
        if self.config["evaluation"]["error_analysis"]["save_misclassified"]:
            save_dir = Path(self.config["evaluation"]["visualization"]["save_dir"])
            with open(save_dir / "error_analysis.json", "w") as f:
                json.dump(misclassified, f, indent=2)
        
        return misclassified 