import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")

print("🔍 First 5 rows of the dataset:")
print(iris.head())

print("\nℹ️ Dataset info:")
print(iris.info())

print("\n📊 Summary statistics:")
print(iris.describe())

print("\n🌸 Count of each species:")
print(iris['species'].value_counts())

sns.set(style="ticks")
sns.pairplot(iris, hue="species", height=2.5)
plt.suptitle("🌼 Iris Dataset Feature Relationships", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(iris.drop("species", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("📈 Feature Correlation Heatmap")
plt.show()
