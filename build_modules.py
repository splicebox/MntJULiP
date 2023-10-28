import os
from pathlib import Path
from models import init_null_NB_cov_model, init_alt_NB_cov_model, init_DM_cov_model


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = Path(base_dir) / 'lib'
    model_dir.mkdir(parents=True, exist_ok=True)
    init_null_NB_cov_model(model_dir)
    init_alt_NB_cov_model(model_dir)
    init_null_DM_cov_model(model_dir)
    init_alt_DM_cov_model(model_dir)

if __name__ == "__main__":
  main()
