# Run this test on the command line to verify CUDA installation:
# python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')"    
# python testCuda.py
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    print(f'Number of devices: {torch.cuda.device_count()}')
    print(f'Device capability: {torch.cuda.get_device_capability(torch.cuda.current_device())}')
    # Allocate a tensor on the GPU and perform a simple operation
    tensor = torch.cuda.FloatTensor([1.0, 2.0, 3.0])
    print(f'Tensor on GPU: {tensor}')
    print(f'Tensor multiplied by 2: {tensor * 2}')  
else:
    print("CUDA is not available.") 
# Check if CUDA is available and print device information

print(f"Torch version:", torch.__version__)
print(f"CUDA version:", torch.version.cuda)
print(f"CUDA available:", torch.cuda.is_available())
print(f"GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

import llama_cpp
print(llama_cpp.llama_cpp.llama_print_system_info().decode())
