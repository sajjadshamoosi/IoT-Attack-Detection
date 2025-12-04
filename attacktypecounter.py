import csv
from collections import Counter

csv_file = "RT_IOT2022.csv"   # ← change to your CSV filename

attack_types = Counter()

with open(csv_file, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        attack = row.get("Attack_type")
        if attack is not None:
            attack_types[attack] += 1

print(f"Number of unique Attack_type values: {len(attack_types)}\n")

print("Unique Attack Types:")
for a in sorted(attack_types):
    print("-", a)

print("\nCounts per Attack Type:")
for a, count in attack_types.items():
    print(f"{a}: {count}")
