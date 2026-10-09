import time

import cupy as cp
from numba import cuda
import pynvml

class CGTracker:
    """
    This class aims to retrieve the number of iterations of the CG for debugging purposes
    """
    def __init__(self):
        self.iterations = 0
        self.vector_to_dbf = 0.0
        self.matvec = 0.0
        self.ravel = 0.0
        self.total_time = 0.0
        self.nMatvecOps = 0
    
    def __call__(self, xk):
        # xk is the current solution vector at each iteration
        self.iterations += 1

    def calculate_total_time(self):
        self.total_time = (
                self.vector_to_dbf +
                self.matvec +
                self.ravel
                )

    def relative_times(self):
        self.calculate_total_time()
        total_time = self.total_time
        percent = lambda t: (t / total_time * 100) if total_time > 0 else 0.0

        print(f"\nEvaluating the Tracker Timmings")
        print(f"\t \t The total number of A*x operations was: {self.nMatvecOps}\n")
        print(f"\t \t Part 1: Transform vector to DBF")
        print(f"\t \t Taken up {percent(self.vector_to_dbf):.1f}% ({self.vector_to_dbf:.3e}s)")
        print(f"\t \t Part 2: Matrix-Vector multiplication")
        print(f"\t \t Taken up {percent(self.matvec):.1f}% ({self.matvec:.3e}s)")
        print(f"\t \t Part 3: Raveling of Y vector")
        print(f"\t \t Taken up {percent(self.ravel):.1f}% ({self.ravel:.3e}s)")
        print(f"\n \t \t Function took {total_time:.4f}s\n \n")



class NewCGTracker:
    """
    This class aims to time all the CG operations for performance analysis
    """
    def __init__(self):
        self.iterations      = 0
        self.preprocess      = 0.0
        self.preconditioning = 0.0
        self.obtain_rho      = 0.0
        self.obtain_p        = 0.0
        self.SpMV_K          = 0.0
        self.obtain_alpha    = 0.0
        self.update_vectors  = 0.0
        self.callback        = 0.0
        self.residual        = 0.0
    
    def __call__(self, xk):
        # xk is the current solution vector at each iteration
        self.iterations += 1

    def calculate_total_time(self):
        self.total_time = (
            self.preprocess      +
            self.preconditioning +
            self.obtain_rho      +
            self.obtain_p        +
            self.SpMV_K          +
            self.obtain_alpha    +
            self.update_vectors  +
            self.callback        +
            self.residual        
                )

    def relative_times(self):
        self.calculate_total_time()
        total_time = self.total_time
        percent = lambda t: (t / total_time * 100) if total_time > 0 else 0.0

        print(f"\nEvaluating the Conjugate Gradient in detail")
        print(f"\t \t Part 1: Preprocessing")
        print(f"\t \t Taken up {percent(self.preprocess):.1f}% ({self.preprocess:.3e}s)")
        print(f"\t \t Part 2: Preconditioning")
        print(f"\t \t Taken up {percent(self.preconditioning):.1f}% ({self.preconditioning:.3e}s)")
        print(f"\t \t Part 3: Obtain rho")
        print(f"\t \t Taken up {percent(self.obtain_rho):.1f}% ({self.obtain_rho:.3e}s)")
        print(f"\t \t Part 4: Obtain p")
        print(f"\t \t Taken up {percent(self.obtain_p):.1f}% ({self.obtain_p:.3e}s)")
        print(f"\t \t Part 5: Sparse Matrix-Vector Product")
        print(f"\t \t Taken up {percent(self.SpMV_K):.1f}% ({self.SpMV_K:.3e}s)")
        print(f"\t \t Part 6: Obtain alpha")
        print(f"\t \t Taken up {percent(self.obtain_alpha):.1f}% ({self.obtain_alpha:.3e}s)")
        print(f"\t \t Part 7: Update Vectors")
        print(f"\t \t Taken up {percent(self.update_vectors):.1f}% ({self.update_vectors:.3e}s)")
        print(f"\t \t Part 8: Callback function")
        print(f"\t \t Taken up {percent(self.callback):.1f}% ({self.callback:.3e}s)")
        print(f"\t \t Part 9: Calculate Residual")
        print(f"\t \t Taken up {percent(self.residual):.1f}% ({self.residual:.3e}s)")
        print(f"\n \t \t Function took {total_time:.4f}s\n \n")





class CudaTracker:
    def __init__(self):
        self.start_event = cp.cuda.Event()
        self.end_event = cp.cuda.Event()

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()
        self.end_event.synchronize()
        # get_elapsed_time returns milliseconds, divide by 1000 for seconds
        return cp.cuda.get_elapsed_time(self.start_event, self.end_event) / 1000.0

        
        


class GPUMemoryTracker:
    def __init__(self, device_index=0):
        self.device_index = device_index

    def display_cupy(self):
        print(f"--- CuPy Memory Pool (Device {self.device_index}) ---")
        mempool = cp.get_default_memory_pool()
        active = mempool.used_bytes() / (1024**3)
        reserved = mempool.total_bytes() / (1024**3)
        print(f"Active Arrays: {active:>7.3f} GB")
        print(f"Reserved Pool: {reserved:>7.3f} GB\n")

    def display_numba(self):
        print(f"--- Numba / CUDA Context (Device {self.device_index}) ---")
        try:
            free_bytes, total_bytes = cuda.current_context().get_memory_info()
            used = (total_bytes - free_bytes) / (1024**3)
            free = free_bytes / (1024**3)
            print(f"Context Used:  {used:>7.3f} GB")
            print(f"Context Free:  {free:>7.3f} GB\n")
        except Exception as e:
            print(f"Unavailable: {e}\n")

    def display_hardware(self):
        print(f"--- PyNVML / Hardware (Device {self.device_index}) ---")
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used = info.used / (1024**3)
            free = info.free / (1024**3)
            total = info.total / (1024**3)
            print(f"Global Used:   {used:>7.3f} GB")
            print(f"Global Free:   {free:>7.3f} GB")
            print(f"Total VRAM:    {total:>7.3f} GB\n")
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            print(f"Unavailable: {e}\n")
        except NameError:
            print("pynvml module not found. Install with: pip install nvidia-ml-py\n")

    def display_all(self):
        print("=" * 45)
        print(f"      GPU MEMORY REPORT (Device {self.device_index})")
        print("=" * 45)
        self.display_cupy()
        self.display_numba()
        self.display_hardware()


