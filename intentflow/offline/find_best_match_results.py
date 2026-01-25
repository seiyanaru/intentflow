
import os
import re

search_dir = "/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results"
targets = {
    "2a_base": 84.67,
    "2a_otta": 87.36,
    "2b_base": 84.85,
    "2b_otta": 84.99,
    "hgd_base": 96.43,
    "hgd_otta": 97.18
}

def get_acc_from_file(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Look for "Average Accuracy: 84.67%" or similar patterns
            match = re.search(r'Average Accuracy[:\s]+([\d\.]+)', content)
            if match:
                return float(match.group(1))
            
            # Look for "test_acc": 0.8467 or similar
            match = re.search(r'"test_acc"[:\s]+([\d\.]+)', content)
            if match:
                val = float(match.group(1))
                if val < 1.0: val *= 100
                return val
            
            # Look for "accuracy": 0.8467
            match = re.search(r'accuracy[:\s]+([\d\.]+)', content)
            if match:
                val = float(match.group(1))
                if val <= 1.0: val *= 100
                return val
            
            # For summary_results.txt which might have "Baseline Accuracy: 84.67%"
            match = re.search(r'Baseline Accuracy[:\s]+([\d\.]+)', content)
            if match:
                 return float(match.group(1))
            
             # For summary_results.txt which might have "OTTA Accuracy: 87.36%"
            match = re.search(r'OTTA Accuracy[:\s]+([\d\.]+)', content)
            if match:
                 return float(match.group(1))

    except Exception as e:
        return None
    return None

print("Searching for matching result files...")

for root, dirs, files in os.walk(search_dir):
    for file in files:
        if file in ["results.txt", "summary_results.txt"]:
            path = os.path.join(root, file)
            acc = get_acc_from_file(path)
            if acc:
                for key, target in targets.items():
                    if abs(acc - target) < 0.05: # Strict tolerance
                        print(f"MATCH {key}: {acc:.2f}% -> {path}")
                    elif abs(acc - target) < 0.2: # Loose tolerance
                         print(f"CLOSE {key}: {acc:.2f}% -> {path}")

