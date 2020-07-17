import os
from pathlib import Path
from models import init_null_BN_model, init_alt_BN_model, init_null_DM_model, init_alt_DM_model


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = Path(base_dir) / 'lib'
    model_dir.mkdir(parents=True, exist_ok=True)
    init_null_BN_model(model_dir)
    init_alt_BN_model(model_dir)
    init_null_DM_model(model_dir)
    init_alt_DM_model(model_dir)

if __name__ == "__main__":
  main()
