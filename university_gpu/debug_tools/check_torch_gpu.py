import torch
import time


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    print_section("PyTorch Version")
    print("torch version:", torch.__version__)
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())

    print_section("CUDA Availability")
    cuda_available = torch.cuda.is_available()
    print("torch.cuda.is_available():", cuda_available)
    print("torch.cuda.device_count():", torch.cuda.device_count())

    if not cuda_available:
        print("\nNo CUDA GPU is available to PyTorch.")
        print("This may be normal if you are running on a login node.")
        print("On the server, you may need to request a GPU job with srun or sbatch.")
        return

    print_section("GPU Information")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")

    print_section("Small GPU Tensor Test")
    device = torch.device("cuda:0")

    print("Creating tensors on GPU...")
    a = torch.randn((2048, 2048), device=device, dtype=torch.float16)
    b = torch.randn((2048, 2048), device=device, dtype=torch.float16)

    torch.cuda.synchronize()
    start = time.time()

    c = a @ b

    torch.cuda.synchronize()
    end = time.time()

    print("Matrix multiplication finished.")
    print("Result tensor shape:", c.shape)
    print("Result tensor device:", c.device)
    print(f"Time used: {end - start:.4f} seconds")

    print_section("GPU Memory Usage")
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3

    print(f"Memory allocated: {allocated:.4f} GB")
    print(f"Memory reserved: {reserved:.4f} GB")

    print_section("Done")
    print("PyTorch GPU test finished successfully.")


if __name__ == "__main__":
    main()