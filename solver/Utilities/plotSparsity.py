import matplotlib
# Force the 'Agg' backend BEFORE importing pyplot to prevent X11/display errors on SSH
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import scipy.sparse as sps

def plot_sparsity_pattern(K_uu_gpu, filename="sparsity_pattern.png", zoom_size=None):
    """
    Plots the sparsity pattern of a CuPy sparse matrix and saves it to disk.
    
    Parameters:
        K_uu_gpu (cupyx.scipy.sparse matrix): The GPU matrix.
        filename (str): The output file name/path (e.g., 'matrix_spy.png').
        zoom_size (int, optional): If provided, zooms in on the top-left N x N block.
    """
    # 1. Convert to COO format on the GPU
    K_coo_gpu = K_uu_gpu.tocoo()
    
    # 2. Transfer ONLY the 1D coordinate vectors and data to the CPU
    rows = K_coo_gpu.row.get()
    cols = K_coo_gpu.col.get()
    vals = K_coo_gpu.data.get()
    
    # 3. Reconstruct as a SciPy sparse matrix on the CPU
    K_cpu = sps.coo_matrix((vals, (rows, cols)), shape=K_uu_gpu.shape)
    
    # 4. Plot using Matplotlib's native sparse spy function
    # A higher DPI ensures tiny dots aren't lost in pixelation on large matrices
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    
    if zoom_size:
        # Slice the top-left corner to see the block structure clearly
        K_zoom = K_cpu.tocsr()[:zoom_size, :zoom_size]
        ax.spy(K_zoom, markersize=2, color='blue')
        ax.set_title(f"Sparsity Pattern (Zoomed {zoom_size}x{zoom_size})")
    else:
        # For full matrices, use a tiny markersize to prevent the plot from becoming a solid blue block
        ax.spy(K_cpu, markersize=0.1, color='blue')
        ax.set_title(f"Global Sparsity Pattern (Shape: {K_uu_gpu.shape[0]}x{K_uu_gpu.shape[1]})")
        
    plt.tight_layout()
    
    # 5. Save and explicitly close the figure to free up system RAM
    fileLocation = "plots/" + filename
    fig.savefig(fileLocation, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved sparsity pattern to {fileLocation}")

# Example usage:
# save_sparsity_pattern(K_uu_gpu, filename="full_matrix.png")
# save_sparsity_pattern(K_uu_gpu, filename="zoomed_matrix.png", zoom_size=100)
