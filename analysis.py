import numpy as np
import matplotlib.pyplot as plt

diagnosis = np.genfromtxt('wdbc.data', delimiter=',', usecols=1, dtype=str)
data = np.genfromtxt('wdbc.data', delimiter=',', usecols=range(2, 32))

feature_names = []
for version in ['mean', 'se', 'worst']:
    for m in ['radius', 'texture', 'perimeter', 'area', 'smoothness',
              'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']:
        feature_names.append(version + '_' + m)

y = np.where(diagnosis == 'M', 1, 0)

print("Number of rows:", data.shape[0])
print("Number of features:", data.shape[1])
print("\nAny missing values?", np.isnan(data).sum())
print("\nBenign:", np.sum(diagnosis == 'B'))
print("Malignant:", np.sum(diagnosis == 'M'))

print("\nBasic statistics for each feature:")
print(f"{'Feature':<25}{'Mean':>10}{'Min':>10}{'Max':>10}{'Std':>10}")
for i in range(data.shape[1]):
    col = data[:, i]
    print(f"{feature_names[i]:<25}{col.mean():>10.2f}{col.min():>10.2f}{col.max():>10.2f}{col.std():>10.2f}")

counts = [np.sum(diagnosis == 'B'), np.sum(diagnosis == 'M')]
plt.bar(['Benign', 'Malignant'], counts, color=['green', 'red'])
plt.title('Number of Benign vs Malignant Cases')
plt.ylabel('Count')
plt.savefig('class_counts.png')
plt.close()

plt.hist(data[:, 0], bins=20, color='blue')
plt.title('Distribution of Mean Radius')
plt.xlabel('Mean Radius')
plt.ylabel('Frequency')
plt.savefig('mean_radius_histogram.png')
plt.close()

print("\nSaved class_counts.png and mean_radius_histogram.png")

centroid_0 = data[y == 0].mean(axis=0)
centroid_1 = data[y == 1].mean(axis=0)

def predict(X):
    dist_0 = np.sqrt(((X - centroid_0) ** 2).sum(axis=1))
    dist_1 = np.sqrt(((X - centroid_1) ** 2).sum(axis=1))
    return np.where(dist_1 < dist_0, 1, 0)

baseline_acc = np.mean(predict(data) == y)
print("\nBaseline accuracy:", round(baseline_acc, 4))

importances = np.zeros(data.shape[1])
np.random.seed(42)

for i in range(data.shape[1]):
    shuffled = data.copy()
    np.random.shuffle(shuffled[:, i])
    new_acc = np.mean(predict(shuffled) == y)
    importances[i] = baseline_acc - new_acc

order = np.argsort(importances)[::-1]

print("\nTop 10 most important features:")
for i in order[:10]:
    print(feature_names[i], round(importances[i], 4))

top10_names = [feature_names[i] for i in order[:10]]
top10_scores = [importances[i] for i in order[:10]]

plt.barh(top10_names[::-1], top10_scores[::-1], color='purple')
plt.xlabel('Drop in Accuracy When Shuffled')
plt.title('Permutation Importance (Top 10 Features)')
plt.tight_layout()
plt.savefig('permutation_importance.png')
plt.close()

print("\nSaved permutation_importance.png")
print("\nDone!")
