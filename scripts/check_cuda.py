import torch, os, subprocess, sys

print("==== PyTorch CUDA Diagnose ====")
print("Python:", sys.version)
print("Torch version:", torch.__version__)
print("Torch built with CUDA:", torch.version.cuda)
print("CUDA available?:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}:", torch.cuda.get_device_name(i))
        print("    Capability:", torch.cuda.get_device_capability(i))

print("\n==== Environment Vars ====")
for var in ["CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"]:
    print(f"{var} =", os.environ.get(var, "<not set>"))

print("\n==== nvcc version ====")
try:
    print(subprocess.check_output(["nvcc", "--version"], text=True))
except Exception as e:
    print("nvcc not found:", e)

print("\n==== nvidia-smi ====")
try:
    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception as e:
    print("nvidia-smi error:", e)
