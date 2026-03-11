import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'soccer_player_performances.csv')
df = pd.read_csv(DATA_PATH)

df = df[df['Performance'] != '-']

def add_features(df):
    df = df.copy()
    
    epsilon = 1e-6
    
    df['Tackle_Success_Rate'] = df['Tackle Won'] / (df['Tackle Attempt'] + epsilon)
    df['Shot_Accuracy'] = df['Shots on Target'] / (df['Shots'] + epsilon)
    df['Goal_Conversion'] = df['Goals'] / (df['Shots'] + epsilon)
    
    df['Pass_Accuracy'] = df['Pass Completed / 90 minutes'] / (df['Pass Attempt / 90 minutes'] + epsilon)
    
    df['Goals_Per_App'] = df['Goals'] / (df['Appearances'] + epsilon)
    df['Assists_Per_App'] = df['Assist'] / (df['Appearances'] + epsilon)
    
    return df

df = add_features(df)

positions = ['Striker', 'Defender', 'Midfielder', 'Goalkeeper']

plt.figure(figsize=(20, 10))
plt.suptitle('Top 5 Influential Features by Position (Gradient Boosting)', fontsize=16)

scaler = StandardScaler()

for i, pos in enumerate(positions):
    subset = df[df['Position'] == pos].copy()
    
    if len(subset) < 10:
        continue
        
    X = subset.drop(['Name', 'Performance', 'Club', 'Nationality', 'Position'], axis=1)
    y = subset['Performance']
    
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(5) 
    
    plt.subplot(2, 2, i+1)
    sns.barplot(data=importances, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)
    plt.title(f'{pos} (n={len(subset)})')
    plt.xlabel('Relative Importance')
    plt.ylabel('')

plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.4, wspace=0.4) 
plt.show()