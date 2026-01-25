
import os
import re

search_dir = "/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results"

def get_acc_from_file(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            match = re.search(r'Average Accuracy[:\s]+([\d\.]+)', content)
            if match:
                return float(match.group(1))
            
            match = re.search(r'"test_acc"[:\s]+([\d\.]+)', content)
            if match:
                val = float(match.group(1))
                if val < 1.0: val *= 100
                return val
            
            match = re.search(r'Baseline Accuracy[:\s]+([\d\.]+)', content)
            if match:
                 return float(match.group(1))
            
            match = re.search(r'OTTA Accuracy[:\s]+([\d\.]+)', content)
            if match:
                 return float(match.group(1))
                 
    except:
        return None
    return None

print("Scanning all result files...")

for root, dirs, files in os.walk(search_dir):
    for file in files:
        if file in ["results.txt", "summary_results.txt"]:
            path = os.path.join(root, file)
            acc = get_acc_from_file(path)
            if acc:
                print(f"File: {path}\n  -> Accuracy: {acc:.2f}%")
