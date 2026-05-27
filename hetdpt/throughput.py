import torch
from torch.utils.data import DataLoader, TensorDataset
import time

def measure_model_throughput(model, 
                             batch_size=64, 
                             image_size=(3, 224, 224),
                             num_iterations=100,
                             warmup_iterations=10,
                             verbose=True):
    """
    Measure the throughput of a deep learning model.
    
    Args:
        model (torch.nn.Module): PyTorch model to benchmark.
        batch_size (int): Batch size, default 256.
        image_size (tuple): Input image size as (channels, height, width), default (3, 224, 224).
        num_iterations (int): Number of benchmark iterations, default 100.
        warmup_iterations (int): Number of warmup iterations, default 10.
        verbose (bool): Whether to print detailed output, default True.
        
    Returns:
        float: Throughput in images per second.
    """
    # Set the device.
    device = next(model.parameters()).device
    model.eval()
    
    if verbose:
        print(f"Starting model throughput measurement...")
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"Input size: {image_size}")
        print(f"Warmup iterations: {warmup_iterations}, benchmark iterations: {num_iterations}")
    
    # Check whether GPU memory is sufficient.
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory
        # Conservative estimate: batch size * 4 bytes per pixel * channels * width * height * extra factor (8).
        required_memory = batch_size * 4 * image_size[0] * image_size[1] * image_size[2] * 8
        if required_memory > total_memory * 0.8:
            raise RuntimeError(f"Insufficient GPU memory! Available memory: {total_memory/1e9:.1f}GB, "
                            f"required memory: {required_memory/1e9:.1f}GB。Consider reducing the batch size.")
    elif batch_size > 16 and verbose:
        print(f"Warning: a large CPU batch size ({batch_size}) may reduce performance.")
    
    # Create synthetic input.
    batch_image_size = tuple([batch_size]) + image_size
    # images = torch.randn(256, 3, 224, 224)
    images = torch.randn(batch_image_size)
    print(images.shape)
    
    if verbose:
        print(f"Generated synthetic input.")
        print("Starting warmup...")
    images = images.to(device)
    # Warmup phase.
    with torch.no_grad():
        for i in range(warmup_iterations):
            model(images)
    
    # Benchmark phase.
    if verbose:
        print("Starting benchmark...")
        
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_iterations):
            model(images)
    
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    end_time = time.time()
    
    # Compute results.
    elapsed_time = end_time - start_time
    total_samples = num_iterations * batch_size
    throughput = total_samples / elapsed_time
    
    if verbose:
        print("\n====== Benchmark Results ======")
        print(f"Completed iterations: {num_iterations}")
        print(f"Total samples: {total_samples}")
        print(f"Total elapsed time: {elapsed_time:.4f}s")
        print(f"Throughput: {throughput:.2f} images/s")
        print(f"Throughput: {throughput/1000:.2f}k images/s")
        print("="*30)
    
    return throughput

# Usage examples.
if __name__ == "__main__":
    import torchvision
    
    # Example 1: benchmark a standard ViT model.
    vit_model = torchvision.models.vit_b_16(weights="DEFAULT")
    vit_model = vit_model.cuda().eval()  # Ensure the model is on GPU.
    
    print("Benchmarking a standard ViT model:")
    throughput = measure_model_throughput(vit_model)
    
    # Example 2: benchmark a ResNet model.
    resnet_model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    resnet_model = resnet_model.cuda().eval()
    
    print("\nBenchmarking a ResNet-50 model:")
    throughput = measure_model_throughput(
        model=resnet_model,
        batch_size=128,
        image_size=(3, 224, 224),
        num_iterations=50
    )
    
    # Example 3: benchmark a custom model.
    class CustomModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
            self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.fc = torch.nn.Linear(128*56*56, 1000)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    custom_model = CustomModel().cuda().eval()
    
    print("\nBenchmarking a custom model:")
    throughput = measure_model_throughput(
        model=custom_model,
        batch_size=64,
        image_size=(3, 224, 224),
        num_iterations=100,
        verbose=True
    )