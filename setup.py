import os
import platform
import shutil
from pathlib import Path

from project_env import sanitize_thread_env

sanitize_thread_env()

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def ensure_cuda_env():
    cuda_home = os.environ.get("CUDA_HOME")
    candidate_homes = []

    if cuda_home:
        candidate_homes.append(Path(cuda_home))
    candidate_homes.extend([
        Path("/usr/local/cuda"),
        Path("/usr/local/cuda-12.8"),
        Path("/opt/cuda"),
    ])

    for candidate in candidate_homes:
        nvcc_name = "nvcc.exe" if platform.system() == "Windows" else "nvcc"
        nvcc_path = candidate / "bin" / nvcc_name
        if nvcc_path.exists():
            resolved_home = str(candidate)
            os.environ["CUDA_HOME"] = resolved_home
            current_path = os.environ.get("PATH", "")
            bin_dir = str(candidate / "bin")
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{current_path}" if current_path else bin_dir
            if platform.system() != "Windows":
                lib64 = str(candidate / "lib64")
                current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = f"{lib64}{os.pathsep}{current_ld_library_path}" if current_ld_library_path else lib64
            return resolved_home

    nvcc_on_path = shutil.which("nvcc")
    if nvcc_on_path:
        return str(Path(nvcc_on_path).resolve().parent.parent)

    raise RuntimeError(
        "CUDA toolkit not found. Please install CUDA Toolkit or set CUDA_HOME so that "
        "`$CUDA_HOME/bin/nvcc` exists."
    )


def cxx_flags():
    if platform.system() == "Windows":
        return ["/O2"]
    return ["-O2"]


def nvcc_flags():
    flags = ["-O2", "--use_fast_math"]
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch_list:
        return flags
    return flags


ensure_cuda_env()


setup(
    name="beginner_flashatt",
    version="0.1.0",
    description="A tiny readable FlashAttention CUDA extension for beginners",
    packages=["beginner_flashatt"],
    ext_modules=[
        CUDAExtension(
            name="beginner_flashatt._C",
            sources=["beginner_flashatt/flashattention.cu"],
            extra_compile_args={
                "cxx": cxx_flags(),
                "nvcc": nvcc_flags(),
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
