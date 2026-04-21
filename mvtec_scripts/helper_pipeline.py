import os
import subprocess

ROOT = "MVTEC-AD"
LOG_FILE = "mvtec_pipeline_output.txt"

# Get all categories
categories = [
    c for c in os.listdir(ROOT)
    if os.path.isdir(os.path.join(ROOT, c))
]

print("Categories found:", categories)

# Open log file
with open(LOG_FILE, "w") as f:

    for category in categories:

        print(f"\n===== RUNNING PIPELINE FOR: {category} =====")

        f.write(f"\n===== CATEGORY: {category} =====\n")

        # Run pipeline and capture output
        result = subprocess.run(
            ["python", "mvtec_pipeline_final.py", category],
            capture_output=True,
            text=True
        )

        # Write output to file
        f.write(result.stdout)

        # If errors occur
        if result.stderr:
            f.write("\n[ERROR]\n")
            f.write(result.stderr)

print(f"\nAll pipeline outputs saved to {LOG_FILE}")
