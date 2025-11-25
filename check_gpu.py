"""
Script to check GPU/CUDA availability for training
"""
import torch

print("="*60)
print("GPU/CUDA Check for Training")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory Total: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Check memory usage
        if torch.cuda.is_available():
            torch.cuda.set_device(i)
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  Memory Allocated: {memory_allocated:.2f} GB")
            print(f"  Memory Reserved: {memory_reserved:.2f} GB")
    
    print(f"\n[SUCCESS] GPU is available and ready for training!")
    print(f"  Training will use: cuda")
else:
    print("\n[WARNING] CUDA is NOT available. Training will use CPU.")
    print("\nTo use GPU, make sure:")
    print("  1. You have NVIDIA GPU installed")
    print("  2. CUDA toolkit is installed")
    print("  3. PyTorch is installed with CUDA support")
    print("  4. GPU drivers are up to date")

print("="*60)

