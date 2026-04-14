import json

import numpy as np

from glassbox.cleaner import (
    ImputationStrategy,
    OutlierCapper,
    SimpleImputer,
    StandardScaler,
)
from glassbox.frame.dataset import Dataset
from glassbox.frame.io import read_csv
from glassbox.inspector.auditor import DataAuditor


def main():
    filepath = "datasets/Penguins/penguins.csv"
    print(f"Loading dataset from {filepath}...")
    dataset = read_csv(filepath)
    print(f"Loaded {dataset.shape[0]} rows and {dataset.shape[1]} columns.")

    auditor = DataAuditor()

    print("\n" + "="*50)
    print("EDA REPORT: BEFORE CLEANING")
    print("="*50)
    report_before = auditor.run_audit(dataset)
    print(json.dumps(json.loads(report_before.to_json()), indent=4))

    print("\n" + "-"*50)
    print("Running Cleaner Modules Pipeline...")
    print("-"*50)

    # penguins dataset indices:
    # 0: id, 1: species, 2: island, 3: bill_length_mm, 4: bill_depth_mm
    # 5: flipper_length_mm, 6: body_mass_g, 7: sex, 8: year
    num_idx = [0, 3, 4, 5, 6, 8]
    cat_idx = [1, 2, 7]

    cleaned_data_matrix = dataset.data.copy()
    num_data = np.array(cleaned_data_matrix[:, num_idx], dtype=float)

    print("-> Applying Imputation (MEAN)...")
    imputer = SimpleImputer(strategy=ImputationStrategy.MEAN)
    imputed_num = imputer.fit_transform(num_data)

    print("-> Applying Scaler (StandardScaler)...")
    scaler = StandardScaler()
    scaled_num = scaler.fit_transform(imputed_num)

    print("-> Applying Outlier Capper...")
    capper = OutlierCapper()
    capped_num = capper.fit_transform(scaled_num)

    # Impute categorical features
    cat_data = cleaned_data_matrix[:, cat_idx]
    print("-> Applying Imputation (MODE) for categoricals...")
    cat_imputer = SimpleImputer(strategy=ImputationStrategy.MODE)
    imputed_cat = cat_imputer.fit_transform(cat_data)

    cleaned_data_matrix[:, num_idx] = capped_num
    cleaned_data_matrix[:, cat_idx] = imputed_cat

    clean_dataset = Dataset(cleaned_data_matrix, dataset.columns)

    print("\n" + "="*50)
    print("EDA REPORT: AFTER CLEANING")
    print("="*50)
    report_after = auditor.run_audit(clean_dataset)
    print(json.dumps(json.loads(report_after.to_json()), indent=4))

if __name__ == "__main__":
    main()
