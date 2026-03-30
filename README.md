# MLOps Midterm Study - Lecture 1

This repository contains the codebase and project documentation for the MLOps Midterm Study project.

## Overview

The project focuses on demonstrating and resolving technical debt in machine learning workflows, specifically working with Generative Adversarial Networks (GANs). It includes:
- **`gan_script.py` & `student_a_gan.ipynb`**: Initial, unstructured code intentionally containing technical debt (Student A).
- **`Student_B_Report.md`**: An analysis report on the reproducibility issues, missing dependencies, and necessary remediation steps (Student B).
- **Dockerization**: A `DockerFile` for containerizing the application.
- **CI/CD Pipeline**: A GitHub Actions workflow (`.github/workflows/ml-pipeline.yml`) to validate the model, run linter checks, and manage artifacts.

## Environment Setup
You can set up the environment using either Conda or pip:

**Using Conda:**
```sh
conda env create -f environment.yml
conda activate <your-env-name>
```

**Using Pip:**
```sh
pip install -r requirements.txt
```

### GitHub Actions (CI/CD)
The project uses GitHub Actions for automated testing. On every push or pull request to the `main` branch, the pipeline will:
1. Setup Python 3.10
2. Install dependencies via `requirements.txt`
3. Run `flake8` for code linting
4. Run a dry test for the model
5. Upload this `README.md` and `environment.yml` as build artifacts.
