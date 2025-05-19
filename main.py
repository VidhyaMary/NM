import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data/sample_data.csv')

# Preprocessing
df['match_result'] = df['match_result'].map({'win': 1, 'loss': 0})
X = df[['minutes_played', 'goals', 'assists', 'passes_completed']]
y = df['match_result']

# Visualize
sns.pairplot(df, hue='match_result')
plt.show()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Player Ranking
df['performance_score'] = (
    df['goals'] * 4 +
    df['assists'] * 3 +
    df['passes_completed'] * 0.1 +
    df['minutes_played'] * 0.01
)

print("\nTop Players:")
print(df[['player_name', 'performance_score']].sort_values(by='performance_score', ascending=False))
