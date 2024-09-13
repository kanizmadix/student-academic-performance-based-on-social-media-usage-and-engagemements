import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv('data.csv')

# Preprocess the data
le = LabelEncoder()
columns_to_encode = ['Level of education', 'Most used platform', 'Main activity', 'Positive enhancement',
                     'Checks social media', 'Frequency of Procrastination', 'Social media Breaks',
                     'Good balance between social media use and academic responsibilities', 'Experience guilt',
                     'Strategies used', 'Tools to manage']

for col in columns_to_encode:
    df[col] = le.fit_transform(df[col])

# Convert 'Time Spent' to numeric
time_spent_map = {'Less than 1 hour': 0.5, '1-2 hours': 1.5, '3-4 hours': 3.5, '5-6 hours': 5.5, 'More than 6 hours': 7}
df['Time Spent'] = df['Time Spent'].map(time_spent_map)

# Convert 'Duration of Breaks' to numeric
duration_map = {'Less than 5 minutes': 2.5, '5-10 minutes': 7.5, '10-20 minutes': 15, '20-30 minutes': 25, 'More than 30 minutes': 35}
df['Duration of Breaks'] = df['Duration of Breaks'].map(duration_map)

# Convert 'Affects academic performance' to numeric
performance_map = {'Not at all': 0, 'Slightly': 1, 'Moderately': 2, 'Significantly': 3, 'Extremely': 4}
df['Affects academic performance'] = df['Affects academic performance'].map(performance_map)

# Select features for correlation analysis
features = ['Time Spent', 'Duration of Breaks', 'Frequency of Procrastination', 'Affects academic performance']
correlation_matrix = df[features].corr()

# Visualization: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Key Factors')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Prepare data for predictive modeling
X = df[['Time Spent', 'Duration of Breaks', 'Frequency of Procrastination']]
y = df['Affects academic performance']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Visualization: Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance for Predicting Academic Performance Impact')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Visualization: Scatter plot of Time Spent vs Academic Performance Impact
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Time Spent', y='Affects academic performance', data=df)
plt.title('Time Spent on Social Media vs Academic Performance Impact')
plt.tight_layout()
plt.savefig('time_spent_vs_performance.png')
plt.close()

# Print top strategies based on academic performance
top_strategies = df.groupby('Strategies used')['Affects academic performance'].mean().sort_values(ascending=True)
print("\nTop Strategies for Minimizing Negative Impact on Academic Performance:")
for strategy, impact in top_strategies.items():
    strategy_name = le.inverse_transform([strategy])[0]
    print(f"{strategy_name}: {impact:.2f}")

# Suggestions based on findings
print("\nSuggestions for Students and Educational Institutions:")
print("1. Limit daily social media usage to less than 2 hours.")
print("2. Take shorter breaks (5-10 minutes) when using social media.")
print("3. Use productivity apps and set time limits to manage social media use.")
print("4. Implement strategies to reduce procrastination, such as setting specific study goals.")
print("5. Educational institutions should provide workshops on effective time management and responsible social media use.")
print("6. Encourage students to turn off notifications during study hours.")
print("7. Promote a balanced approach to social media use and academic responsibilities.")
