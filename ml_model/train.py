from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import ml_config as config

def get_models():
    """Initializes and returns the 3 models for side-by-side comparison."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, 
            random_state=42, 
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, 
            random_state=42
        )
    }
    return models

def train_and_predict_all(X_train, y_train, X_test):
    """Trains each model on the training set and generates predictions on the test set."""
    models = get_models()
    predictions = {}
    
    for name, model in models.items():
        # Fit model on training set split by replication group
        model.fit(X_train, y_train)
        
        # Predict on hold-out replication test set
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
    return predictions