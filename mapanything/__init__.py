import os

# Ensure headless backend if invoked from Jupyter / Kaggle / Colab environments
if "matplotlib_inline" in os.environ.get("MPLBACKEND", ""):
    os.environ["MPLBACKEND"] = "Agg"
