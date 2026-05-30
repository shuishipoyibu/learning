import os
import platform
import shutil
from pathlib import Path

_INVALID_THREAD_ENV_VALUES = {"", "0"}


def sanitize_thread_env() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = os.environ.get(key)
        if value is not None and value.strip() in _INVALID_THREAD_ENV_VALUES:
            os.environ.pop(key, None)


def ensure_cuda_env() -> str:
    cuda_home = os.environ.get("CUDA_HOME")
    candidate_homes = []

    if cuda_home:
        candidate_homes.append(Path(cuda_home))
    candidate_homes.extend(
        [
            Path("/usr/local/cuda"),
            Path("/usr/local/cuda-12.8"),
            Path("/opt/cuda"),
        ]
    )

    nvcc_name = "nvcc.exe" if platform.system() == "Windows" else "nvcc"

    for candidate in candidate_homes:
        nvcc_path = candidate / "bin" / nvcc_name
        if nvcc_path.exists():
            resolved_home = str(candidate)
            bin_dir = str(candidate / "bin")

            os.environ["CUDA_HOME"] = resolved_home
            current_path = os.environ.get("PATH", "")
            if current_path:
                path_parts = current_path.split(os.pathsep)
                if bin_dir not in path_parts:
                    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{current_path}"
            else:
                os.environ["PATH"] = bin_dir

            if platform.system() != "Windows":
                lib64 = str(candidate / "lib64")
                current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
                if current_ld_library_path:
                    ld_parts = current_ld_library_path.split(os.pathsep)
                    if lib64 not in ld_parts:
                        os.environ["LD_LIBRARY_PATH"] = f"{lib64}{os.pathsep}{current_ld_library_path}"
                else:
                    os.environ["LD_LIBRARY_PATH"] = lib64

            return resolved_home

    nvcc_on_path = shutil.which("nvcc")
    if nvcc_on_path:
        resolved_home = str(Path(nvcc_on_path).resolve().parent.parent)
        os.environ.setdefault("CUDA_HOME", resolved_home)
        return resolved_home

    raise RuntimeError(
        "CUDA toolkit not found. Please install CUDA Toolkit or set CUDA_HOME so that "
        "`$CUDA_HOME/bin/nvcc` exists."
    )
