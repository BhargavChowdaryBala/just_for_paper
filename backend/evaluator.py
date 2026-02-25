import json
import warnings

class EvaluationModule:
    """
    IEEE-standard Evaluation Module for ML Classification.
    Computes Accuracy, Precision, Recall, and F1-Score based on confusion matrix.
    """
    
    def __init__(self):
        self.metrics = {}

    def compute_confusion_matrix(self, y_true, y_pred):
        """Computes TP, TN, FP, FN for binary/multiclass labels."""
        tp = tn = fp = fn = 0
        
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1: tp += 1
            elif yt == 0 and yp == 0: tn += 1
            elif yt == 0 and yp == 1: fp += 1
            elif yt == 1 and yp == 0: fn += 1
            
        return tp, tn, fp, fn

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculates IEEE-standard metrics strictly from TP, TN, FP, FN.
        Handles divide-by-zero safely.
        """
        tp, tn, fp, fn = self.compute_confusion_matrix(y_true, y_pred)
        
        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        # Precision = TP / (TP + FP)
        precision_denom = tp + fp
        precision = tp / precision_denom if precision_denom > 0 else 0.0
        if precision_denom == 0:
            warnings.warn("Precision calculation: Division by zero encountered (TP + FP = 0).")
            
        # Recall = TP / (TP + FN)
        recall_denom = tp + fn
        recall = tp / recall_denom if recall_denom > 0 else 0.0
        if recall_denom == 0:
            warnings.warn("Recall calculation: Division by zero encountered (TP + FN = 0).")
            
        # F1-Score = (2 * TP) / (2 * TP + FP + FN)
        f1_denom = (2 * tp) + fp + fn
        f1_score = (2 * tp) / f1_denom if f1_denom > 0 else 0.0
        if f1_denom == 0:
             warnings.warn("F1-Score calculation: Division by zero encountered (2*TP + FP + FN = 0).")

        self.metrics = {
            "confusion_matrix": {
                "TP": tp, "TN": tn, "FP": fp, "FN": fn
            },
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1_score, 4)
            }
        }
        return self.metrics

    def get_report(self, format="json"):
        """Returns the metrics in a structured format."""
        if format == "json":
            return json.dumps(self.metrics, indent=4)
        return self.metrics

if __name__ == "__main__":
    # Example validation run for IEEE documentation
    evaluator = EvaluationModule()
    # Mock data: 1=Correct Detection, 0=Failure
    y_true = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1] 
    y_pred = [1, 1, 0, 0, 1, 1, 1, 1, 1, 1]
    
    results = evaluator.calculate_metrics(y_true, y_pred)
    print("IEEE Standard Metrics Report:")
    print(evaluator.get_report())
