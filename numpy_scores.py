import numpy as np
# Task 1 — Generate and Inspect the Data
np.random.seed(42)
scores = np.random.randint(50, 101, size=(5, 4))
print("Scores:\n", scores)
print()
print("3rd student, 2nd subject:", scores[2, 1])
print("\nLast 2 students:\n", scores[-2:, :])
print("\nFirst 3 students, subjects 2 & 3:\n", scores[:3, 1:3])
print("\n---------------------------------\n")


# Task 2 — Analyze with Broadcasting
column_means = np.round(np.mean(scores, axis=0), 2)
print("Column-wise mean:", column_means)
curve = np.array([5, 3, 7, 2])
curved_scores = scores + curve
curved_scores = np.clip(curved_scores, 0, 100)
print("\nCurved Scores:\n", curved_scores)
row_max = np.max(curved_scores, axis=1)
print("\nBest score per student:", row_max)
print("\n---------------------------------\n")

# Task 3 — Normalize and Identify
row_min = np.min(curved_scores, axis=1, keepdims=True)
row_max = np.max(curved_scores, axis=1, keepdims=True)
normalized = (curved_scores - row_min) / (row_max - row_min)
print("Normalized Scores:\n", normalized)
max_index = np.unravel_index(np.argmax(normalized), normalized.shape)
print("\nHighest normalized value at (student, subject):", max_index)
above_90 = curved_scores[curved_scores > 90]
print("\nScores above 90:", above_90)
