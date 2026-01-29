import torch
import time
import cv2
import fastcv
import numpy as np
import torch.cuda.nvtx as nvtx

def benchmark_laplacian(sizes=[1024, 2048, 4096], runs=50):
    results = []
    
    for size in sizes:
        print(f"\n=== Benchmarking {size}x{size} image ===")
        
        img_np = np.random.randint(0, 256, (size, size), dtype=np.uint8)
        img_torch = torch.from_numpy(img_np)
        nvtx.range_push("CPU_Loop")
        start = time.perf_counter()
        for _ in range(runs):
            laplacian = cv2.Laplacian(img_np, cv2.CV_16S, ksize=1)
            _ = cv2.convertScaleAbs(laplacian)
        end = time.perf_counter()
        nvtx.range_pop()
        cv_time = (end - start) / runs * 1000  # ms per run

        torch.cuda.synchronize()
        nvtx.range_push("FastCV_GPU_Loop")
        start = time.perf_counter()
        for _ in range(runs):
            _ = fastcv.laplacian(img_torch)
        torch.cuda.synchronize()
        end = time.perf_counter()
        nvtx.range_pop()
        fc_time = (end - start) / runs * 1000  # ms per run

        results.append((size, cv_time, fc_time))
        print(f"OpenCV (CPU): {cv_time:.4f} ms | fastcv (CUDA): {fc_time:.4f} ms")
    
    return results


if __name__ == "__main__":
    results = benchmark_laplacian()
    print("\n=== Final Results ===")
    print("Size\t\tOpenCV (CPU)\tfastcv (CUDA)")
    for size, cv_time, fc_time in results:
        print(f"{size}x{size}\t{cv_time:.4f} ms\t{fc_time:.4f} ms")
