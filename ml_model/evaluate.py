from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pandas as pd
import ml_config as config

def evaluate_models(y_test, predictions):
    """Calculates F1-scores and confusion matrices for all predictions."""
    summary_results = []
    
    # Invert mapping to display text labels instead of integers 0, 1, 2
    target_names = [k for k, v in sorted(config.TARGET_MAP.items(), key=lambda item: item[1])]

    print("\n" + "="*60)
    print("           MODEL EVALUATION & METRICS REPORT           ")
    print("="*60)

    for name, y_pred in predictions.items():
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        
        summary_results.append({
            "Model": name,
            "Macro F1-Score": round(macro_f1, 4),
            "Weighted F1-Score": round(weighted_f1, 4)
        })
        
        print(f"\n--- Model: {name} ---")
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
        print(cm_df)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        print("-" * 50)
        
    print("\nSUMMARY COMPARISON TABLE:")
    summary_df = pd.DataFrame(summary_results)
    print(summary_df.to_string(index=False))
    return summary_df