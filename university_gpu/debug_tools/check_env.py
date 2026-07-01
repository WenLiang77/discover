import os
import sys
import platform
import shutil
import subprocess


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_import(package_name):
    try:
        module = __import__(package_name)
        version = getattr(module, "__version__", "version not found")
        print(f"[OK] {package_name}: {version}")
    except Exception as e:
        print(f"[MISSING] {package_name}: {repr(e)}")


def run_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=30,
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
    except Exception as e:
        print(f"Command failed: {repr(e)}")


def main():
    print_section("Basic Python Environment")
    print("Python version:", sys.version)
    print("Python executable:", sys.executable)
    print("Current working directory:", os.getcwd())
    print("Platform:", platform.platform())
    print("Machine:", platform.machine())

    print_section("Important Environment Variables")
    for key in [
        "HOME",
        "USER",
        "SHELL",
        "PATH",
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_GPUS",
        "SLURM_CPUS_PER_TASK",
    ]:
        print(f"{key} =", os.environ.get(key, "<not set>"))

    print_section("Check nvidia-smi")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        print("nvidia-smi not found. This may be normal on a login node.")
    else:
        print("nvidia-smi found at:", nvidia_smi)
        run_command("nvidia-smi")

    print_section("Python Package Checks")
    for package in [
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "numpy",
        "pandas",
        "scipy",
    ]:
        check_import(package)

    print_section("PyTorch CUDA Check")
    try:
        import torch

        print("torch.cuda.is_available():", torch.cuda.is_available())
        print("torch.cuda.device_count():", torch.cuda.device_count())

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}:", torch.cuda.get_device_name(i))
    except Exception as e:
        print("PyTorch CUDA check failed:", repr(e))

    print_section("Done")
    print("Environment check finished.")


if __name__ == "__main__":
    main()