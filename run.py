import os
import subprocess
import sys

def run(script):
    print(f"\n>>> Running {script}...")
    result = subprocess.run(
        [sys.executable, script], check=True
    )

print("HR Attrition Dashboard Launcher")
print("="*40)

# Check if model files exist
required_files = [
    'models/xgb_attrition.pkl',
    'models/predictions.csv',
    'models/shap_values.csv',
    'data/master_hr_dataset.csv',
    'data/processed_hr_dataset.csv'
]

missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    print(f"Missing files detected: {missing}")
    print("Running pipeline to regenerate...")
    run('src/build_dataset.py')
    run('src/preprocess.py')
    run('src/train_model.py')
    run('src/tune_threshold.py')
    run('src/explain.py')
    print("\nPipeline complete.")
else:
    print("All model files found — skipping training.")

print("\nLaunching dashboard...")
os.system('streamlit run app/dashboard.py')