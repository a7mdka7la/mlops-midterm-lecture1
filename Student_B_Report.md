# Student B Report: GAN Technical Debt Analysis

## Overview
Student A provided a Jupyter notebook (`student_a_gan.ipynb`) and a data file (`data.csv`), simulating a simple GAN training pipeline. The goal was to run the script and achieve the exact same accuracy score, answering standard MLOps debugging questions.

### 1. How many commands did I have to run before it worked?
It took **4 changes/commands** to get the script running:
1. `pip install pandas numpy torch` – The script was missing a `requirements.txt` file, requiring manual detection of dependencies.
2. `DATA_PATH = 'data.csv'` – The original path was hardcoded to `/Users/studentA/my_secret_folder/data.csv`, forcing a manual change to an accessible local path.
3. Upgrading or changing the numpy type – The notebook used `np.float`, which is deprecated and removed in `numpy 1.24+`. I had to change it to `float` or downgrade numpy to an older version.
4. Actually running the notebook cells.

### 2. What libraries were missing? Did version mismatches cause errors?
- **Missing Libraries:** `pandas`, `numpy`, and `torch` were required, but no documentation or environment spec was provided.
- **Version Mismatches:** Yes! `np.float` has been deprecated. Running the script with a modern version of numpy threw an `AttributeError`. This perfectly illustrates how unchecked dependencies can break seemingly working code over time.

### 3. Did the model produce the same result? If not, why?
**No, the model did not produce the exact same final loss.**
- **Why? (Missing Seed):** Student A did not set a random seed (e.g., `torch.manual_seed(42)`). Because PyTorch randomly initializes the network’s weights and the Adam optimizer handles random step updates, every execution produces a completely different `Final loss` value. Reproducibility was mathematically impossible.

### 4. If this had to run on a server at 3:00 AM, would it survive?
**Absolutely not.** 
- It has strict unportable paths that would crash immediately in a production environment (like a Linux server or Docker container).
- There is no automated saving of the model weights (`torch.save(model.state_dict())`), meaning the trained data is lost immediately after computation.
- Due to the versioning issue with Numpy, any modern automated CI/CD pipeline installing the `latest` packages would fail at runtime. Let alone the lack of a headless/script form (it's currently trapped in a notebook).
