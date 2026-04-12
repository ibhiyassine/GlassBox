import json
from glassbox.frame.io import read_csv
from glassbox.inspector.auditor import DataAuditor

def main():
    filepath = "datasets/Penguins/penguins.csv"
    print(f"Loading dataset from {filepath}...")
    dataset = read_csv(filepath)
    print(f"Loaded {dataset.shape[0]} rows and {dataset.shape[1]} columns.")
    
    print("\n--- Dataset Head (First 5 rows) ---")
    print(f"{' | '.join(dataset.columns)}")
    print("-" * 50)
    # Preview top 5 rows
    for row in dataset.data[:5]:
        print(" | ".join(str(cell) for cell in row))
    
    auditor = DataAuditor()
    print("Running audit...")
    report = auditor.run_audit(dataset)
    
    print("\n--- Audit Report ---")
    json_output = report.to_json()
    
    # Pretty print the JSON output
    parsed = json.loads(json_output)
    print(json.dumps(parsed, indent=4))

if __name__ == "__main__":
    main()
