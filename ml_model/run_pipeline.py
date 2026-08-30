from data_loader import load_and_preprocess_data, split_data_by_group
from train import train_and_predict_all
from evaluate import evaluate_models

def main():
    print("1. Loading dataset and building entry-level baseline feature set...")
    X, y, groups = load_and_preprocess_data()
    print(f"   Total Samples: {len(X)} | Features: {X.shape[1]} | Replication Groups: {groups.nunique()}")
    
    print("\n2. Splitting data strictly by replication seed (GroupShuffleSplit)...")
    X_train, X_test, y_train, y_test = split_data_by_group(X, y, groups, test_size=0.2, random_state=42)
    print(f"   Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    print("\n3. Training Logistic Regression, Random Forest, and XGBoost...")
    predictions = train_and_predict_all(X_train, y_train, X_test)
    
    print("\n4. Evaluating model performance...")
    evaluate_models(y_test, predictions)

if __name__ == "__main__":
    main()