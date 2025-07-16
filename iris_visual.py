import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset (from seaborn's built-in datasets)
iris = sns.load_dataset("iris")

# Show the first 5 rows
print("🔍 First 5 rows of the dataset:")
print(iris.head())

# Basic dataset info
print("\nℹ️ Dataset info:")
print(iris.info())

# Summary statistics
print("\n📊 Summary statistics:")
print(iris.describe())

# Count per species
print("\n🌸 Count of each species:")
print(iris['species'].value_counts())

# Pair plot to visualize relationships
sns.set(style="ticks")
sns.pairplot(iris, hue="species", height=2.5)
plt.suptitle("🌼 Iris Dataset Feature Relationships", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(iris.drop("species", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("📈 Feature Correlation Heatmap")
plt.show()
