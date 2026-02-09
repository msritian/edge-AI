import subprocess
import json
import os

def run_bench(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {cmd}: {result.stderr}")
        return None
    # Expecting the last line to be a JSON string like {"peak_mb": 123.4}
    try:
        lines = result.stdout.strip().split('\n')
        return json.loads(lines[-1])
    except:
        print(f"Failed to parse output from {cmd}")
        return None

def main():
    contenders = {
        "PyTorch (FP32 Sim)": "python3 benchmark_memory_isolated.py sim",
        "ONNX Runtime": "python3 benchmark_memory_isolated.py onnx",
        "Bitwise Kernel (Manual)": "python3 benchmark_memory_isolated.py bitwise"
    }
    
    results = {}
    for name, cmd in contenders.items():
        print(f"Benchmarking {name}...")
        results[name] = run_bench(cmd)
        
    print("\n--- FINAL MEMORY RESULTS (Batch 128) ---")
    print(f"{'Contender':<25} | {'Peak RAM (MB)':<15}")
    print("-" * 45)
    for name, res in results.items():
        if res:
            print(f"{name:<25} | {res['peak_mb']:<15.2f}")

if __name__ == "__main__":
    main()
