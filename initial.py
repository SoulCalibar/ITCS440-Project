import pandas as pd

# Define column names as per UCI description
columns = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
    "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat"
]

# Load the data file (assumes it's named 'mushrooms.data' and is in the same folder)
df = pd.read_csv("mushrooms.data", header=None, names=columns)

# Save to CSV
df.to_csv("mushrooms.csv", index=False)

print("Converted to mushrooms.csv successfully.")
