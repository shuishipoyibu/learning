import importlib.util
import shutil
import subprocess
import sys

from project_env import ensure_cuda_env, sanitize_thread_env

sanitize_thread_env()
ensure_cuda_env()

import torch


def run(command):
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    print("Python:", sys.version.replace("\n", " "))
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("nvcc:", shutil.which("nvcc") or "not found")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run the project tests.")
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc was not found. Install CUDA Toolkit or add nvcc to PATH.")
    if importlib.util.find_spec("pytest") is None:
        raise SystemExit("pytest was not found. Install it with `pip install pytest`.")

    run([sys.executable, "setup.py", "develop"])
    run([sys.executable, "-m", "pytest", "-q"])
    run([sys.executable, "demo_mini_model.py"])


if __name__ == "__main__":
    main()
