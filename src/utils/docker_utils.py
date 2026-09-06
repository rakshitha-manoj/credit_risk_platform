import os

def resolve_data_path(filename: str, data_dir: str) -> str:
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected dataset file at '{path}'. Confirm the Home Credit CSVs are placed in '{data_dir}' (mounted via docker-compose volumes, not committed to git).")
    return path