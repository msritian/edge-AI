import torch
print(f"Parallel Info:\n{torch.__config__.parallel_info()}")
print(f"Num Threads: {torch.get_num_threads()}")
print(f"OMP Num Threads: {torch.get_num_interop_threads()}")
