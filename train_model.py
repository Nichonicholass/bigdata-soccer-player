import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, 'soccer_player_performances.csv')
TEST_PATH = os.path.join(BASE_DIR, 'soccer_player_test.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'soccer_pipeline.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'position_encoder.pkl')
PRED_PATH = os.path.join(BASE_DIR, 'soccer_player_test_with_predictions.csv')

def add_features(df):
    """
    Feature Engineering to improve model performance.
    Adds ratio-based features to capture player efficiency.
    """
    df = df.copy()
    
    epsilon = 1e-6
    
    df['Tackle_Success_Rate'] = df['Tackle Won'] / (df['Tackle Attempt'] + epsilon)
    df['Shot_Accuracy'] = df['Shots on Target'] / (df['Shots'] + epsilon)
    df['Goal_Conversion'] = df['Goals'] / (df['Shots'] + epsilon)
    
    df['Pass_Accuracy'] = df['Pass Completed / 90 minutes'] / (df['Pass Attempt / 90 minutes'] + epsilon)
    
    df['Goals_Per_App'] = df['Goals'] / (df['Appearances'] + epsilon)
    df['Assists_Per_App'] = df['Assist'] / (df['Appearances'] + epsilon)
    
    return df

def train():
    print("SOCCER PLAYER PERFORMANCE PREDICTION")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = train_df[train_df['Performance'] != '-']

    print("  Applying Feature Engineering...")
    y = train_df['Performance']
    
    X_enhanced = add_features(train_df)
    
    cols_to_drop = ['Name', 'Performance', 'Club', 'Nationality']
    X = X_enhanced.drop(cols_to_drop, axis=1)
    
    le_position = LabelEncoder()
    X['Position'] = le_position.fit_transform(X['Position'])
    joblib.dump(le_position, ENCODER_PATH) 

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nStep 1: Baseline Model (Random Forest)")
    pipeline_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    
    pipeline_rf.fit(X_train, y_train)
    rf_preds = pipeline_rf.predict(X_val)
    rf_acc = accuracy_score(y_val, rf_preds)
    print(f"  Random Forest Validation Accuracy: {rf_acc:.2%}")

    print("\nStep 2: Final Model (Gradient Boosting)")
    pipeline_gb = Pipeline([
        ('scaler', StandardScaler()),
        ('gb', GradientBoostingClassifier(random_state=42))
    ])

    param_grid = {
        'gb__n_estimators': [100, 200],
        'gb__learning_rate': [0.05, 0.1],
        'gb__max_depth': [3, 4],
        'gb__subsample': [0.8, 1.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline_gb, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    best_gb = grid_search.best_estimator_

    print(f"  Best Params: {grid_search.best_params_}")
    
    gb_preds = best_gb.predict(X_val)
    gb_acc = accuracy_score(y_val, gb_preds)
    print(f"  Gradient Boosting Validation Accuracy: {gb_acc:.2%}")

    print("\nModel Comparison")
    print(f"  Random Forest (Baseline):   {rf_acc:.4f}")
    print(f"  Gradient Boosting (Final):  {gb_acc:.4f}")
    
    print("\nClassification Report (Final Model - Gradient Boosting):\n")
    print(classification_report(y_val, gb_preds))

    best_gb.fit(X, y)
    joblib.dump(best_gb, MODEL_PATH) 
    print("  Final Gradient Boosting Pipeline saved.")

    test_X_enhanced = add_features(test_df)
    test_X = test_X_enhanced.drop(['Name', 'Club', 'Nationality'], axis=1)
    
    test_X['Position'] = le_position.transform(test_X['Position'])
    
    test_preds = best_gb.predict(test_X)
    
    final_preds = []
    for i, row in test_df.iterrows():
        if row['Appearances'] == 0:
            final_preds.append('-')
        else:
            final_preds.append(test_preds[i])

    test_df['Performance'] = final_preds
    test_df.to_csv(PRED_PATH, index=False)
    print("  Predictions saved.")

if __name__ == "__main__":
    train()