import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("data/sales_data.csv")

if df.shape[1] == 5:
    df.columns = ['Index','TV','Radio','Newspaper','Sales']
    df = df.drop('Index', axis=1)

print(df.head())

# -----------------------------
# Visualization 🔥
# -----------------------------

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("outputs/heatmap.png")
plt.show()

# Pairplot (very attractive)
sns.pairplot(df)
plt.savefig("outputs/pairplot.png")
plt.show()

# -----------------------------
# Model
# -----------------------------
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
print("\nModel Performance:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -----------------------------
# Final Graph 🔥
# -----------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.savefig("outputs/prediction_graph.png")
plt.show()

# -----------------------------
# Business Insight 💡
# -----------------------------
print("\nBusiness Insight:")
print("TV advertising has the strongest impact on sales.")
print("Radio contributes moderately.")
print("Newspaper has the least effect.")

