import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
print("Loading data...")
df = pd.read_csv("Crop_recommendation.csv")
print("First 5 records:")
print(df.head())

# 2. Separate features (soil & weather) from target (crop label)
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["label"]

# 3. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Standardize the data (essential for distance-based KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train initial KNN model with K=5
print("\nTraining KNN (K=5)...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 6. Evaluate baseline performance
y_pred = knn.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy with K=5: {acc * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Test different values of K (satisfies Task 3 requirements)
print("Evaluating K from 1 to 20...")
k_values = range(1, 21)
accuracies = []

for k in k_values:
  clf = KNeighborsClassifier(n_neighbors=k)
  clf.fit(X_train_scaled, y_train)
  accuracies.append(clf.score(X_test_scaled, y_test))

# 8. Generate and save K comparison plot
plt.figure(figsize=(9, 5))
plt.plot(k_values, accuracies, marker="o", color="forestgreen", linewidth=2)
plt.title("KNN Accuracy vs. Value of K")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("k_tuning_plot.png")
print("\nSaved plot as 'k_tuning_plot.png'")

# 9. Test a manual prediction
sample = np.array([[90, 42, 43, 20.8, 82.0, 6.5, 202.9]])
sample_scaled = scaler.transform(sample)
pred = knn.predict(sample_scaled)
print(f"Sample prediction result: Recommended crop is -> {pred[0]}")


import seaborn as sns

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=knn.classes_,
    yticklabels=knn.classes_,
)
plt.title('Crop Classification Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved confusion matrix as 'confusion_matrix.png'")