#!/usr/bin/env bash

# ==============================================================================
# init_setup.sh
# Project: fraud-detection
# Description: Bootstrap full project structure (package-based src/fraud_detection)
# Safe to re-run (conditional creation)
# ==============================================================================

set -e
set -o pipefail

echo "=============================================="
echo "RAF-complaint-Chatbot - Project Structure Setup"
echo "=============================================="

# ----------------------------
# Helper functions (conditional)
# ----------------------------
create_dir () {
    [ -d "$1" ] || mkdir -p "$1"
}

create_file () {
    [ -f "$1" ] || touch "$1"
}

# ----------------------------
# 1. Project root directories
# ----------------------------
dirs=(
    "config"
    "data/raw"
    "data/interim"
    "data/processed"
    "data/external"
    "notebooks"
    "src/fraud_detection"
    "src/fraud_detection/core"
    "src/fraud_detection/data"
    "src/fraud_detection/features"
    "src/fraud_detection/models"
    "src/fraud_detection/explainability"
    "src/fraud_detection/api"
    "src/fraud_detection/pipeline"
    "src/fraud_detection/utils"
    "tests"
    "docker"
    "scripts"
    "mlruns"
    ".github/workflows"
)

for d in "${dirs[@]}"; do
    create_dir "$d"
done

echo "✓ Directories ready"

# ----------------------------
# 2. Python package files
# ----------------------------
py_files=(
    # Root package
    "src/fraud_detection/__init__.py"

    # Core
    "src/fraud_detection/core/__init__.py"
    "src/fraud_detection/core/config.py"
    "src/fraud_detection/core/settings.py"

    # Data
    "src/fraud_detection/data/__init__.py"
    "src/fraud_detection/data/load_data.py"
    "src/fraud_detection/data/clean.py"
    "src/fraud_detection/data/ip_geolocation.py"
    "src/fraud_detection/data/preprocess.py"
    "src/fraud_detection/data/imbalance.py"
    "src/fraud_detection/data/splitter.py"

    # Features
    "src/fraud_detection/features/__init__.py"
    "src/fraud_detection/features/time_features.py"
    "src/fraud_detection/features/behavioral.py"
    "src/fraud_detection/features/transaction.py"
    "src/fraud_detection/features/geo_features.py"
    "src/fraud_detection/features/feature_builder.py"

    # Models
    "src/fraud_detection/models/__init__.py"
    "src/fraud_detection/models/baseline.py"
    "src/fraud_detection/models/ensemble.py"
    "src/fraud_detection/models/train.py"
    "src/fraud_detection/models/tuning.py"
    "src/fraud_detection/models/evaluate.py"
    "src/fraud_detection/models/predict.py"

    # Explainability
    "src/fraud_detection/explainability/__init__.py"
    "src/fraud_detection/explainability/shap_analysis.py"
    "src/fraud_detection/explainability/feature_importance.py"
    "src/fraud_detection/explainability/report.py"

    # API
    "src/fraud_detection/api/__init__.py"
    "src/fraud_detection/api/main.py"
    "src/fraud_detection/api/schemas.py"
    "src/fraud_detection/api/utils.py"

    # Pipeline
    "src/fraud_detection/pipeline/__init__.py"
    "src/fraud_detection/pipeline/dvc_stage_data.py"
    "src/fraud_detection/pipeline/dvc_stage_features.py"
    "src/fraud_detection/pipeline/dvc_stage_train.py"
    "src/fraud_detection/pipeline/dvc_stage_evaluate.py"

    # Utils
    "src/fraud_detection/utils/__init__.py"
    "src/fraud_detection/utils/project_root.py"
    "src/fraud_detection/utils/logger.py"
    "src/fraud_detection/utils/metrics.py"
    "src/fraud_detection/utils/constants.py"
    "src/fraud_detection/utils/helpers.py"

    # Tests
    "tests/__init__.py"
    "tests/test_data_cleaning.py"
    "tests/test_feature_engineering.py"
    "tests/test_imbalance_handling.py"
    "tests/test_model_training.py"
    "tests/test_api.py"
)

for f in "${py_files[@]}"; do
    create_file "$f"
done

echo "✓ Python files ready"

# ----------------------------
# 3. YAML configuration files
# ----------------------------
yaml_files=(
    "config/data.yaml"
    "config/features.yaml"
    "config/model.yaml"
    "config/imbalance.yaml"
    "config/train.yaml"
    "config/explainability.yaml"
    "config/api.yaml"
)

for y in "${yaml_files[@]}"; do
    create_file "$y"
done

echo "✓ Config files ready"

# ----------------------------
# 4. Core project files
# ----------------------------
core_files=(
    "README.md"
    ".gitignore"
    ".dockerignore"
    ".env.example"
    "pyproject.toml"
    "requirements.txt"
    "dvc.yaml"
    "params.yaml"
    "docker/Dockerfile"
    "docker/docker-compose.yml"
    "docker/start.sh"
    "notebooks/eda_ecommerce.ipynb"
    "notebooks/eda_creditcard.ipynb"
    "notebooks/feature_engineering.ipynb"
    "notebooks/modeling.ipynb"
    "notebooks/explainability.ipynb"
)

for f in "${core_files[@]}"; do
    create_file "$f"
done

chmod +x docker/start.sh 2>/dev/null || true

echo "✓ Core files ready"

# ----------------------------
# 5. Scripts
# ----------------------------
scripts=(
    "scripts/run_api.sh"
    "scripts/run_training.sh"
    "scripts/run_pipeline.sh"
)

for s in "${scripts[@]}"; do
    create_file "$s"
    chmod +x "$s"
done

# Script contents (only if empty)
if [ ! -s scripts/run_api.sh ]; then
cat <<EOF > scripts/run_api.sh
#!/usr/bin/env bash
echo "Starting Fraud Detection API..."
uvicorn fraud_detection.api.main:app --host 0.0.0.0 --port 8000 --reload
EOF
fi

if [ ! -s scripts/run_training.sh ]; then
cat <<EOF > scripts/run_training.sh
#!/usr/bin/env bash
echo "Running training pipeline..."
python -m fraud_detection.models.train
EOF
fi

if [ ! -s scripts/run_pipeline.sh ]; then
cat <<EOF > scripts/run_pipeline.sh
#!/usr/bin/env bash
echo "Running full DVC pipeline..."
dvc repro
EOF
fi

echo "✓ Scripts ready"

# ----------------------------
# 6. Virtual environment
# ----------------------------
if [ ! -d ".venv" ]; then
    python -m venv .venv
    echo "✓ Virtual environment created (.venv)"
fi

# ----------------------------
# 7. Final message
# ----------------------------
echo "=============================================="
echo "Fraud Detection project structure is ready"
echo "Activate env: source .venv/bin/activate"
echo "Run API: scripts/run_api.sh"
echo "Run training: scripts/run_training.sh"
echo "Run pipeline: scripts/run_pipeline.sh"
echo "Edit configs in ./config"
echo "=============================================="
