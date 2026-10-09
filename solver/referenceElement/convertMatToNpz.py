import os
import numpy as np
import scipy.io

def mat_to_npz(mat_path, npz_path):
    """
    Converts a .mat file to .npz.
    If the .npz already exists, the function won't run the conversion again.
    """
    if not os.path.exists(npz_path):
        # Load the .mat file (returns a dictionary)
        data = scipy.io.loadmat(mat_path)
        
        # Remove internal fields that are not needed
        data.pop('__header__', None)
        data.pop('__version__', None)
        data.pop('__globals__', None)

        # Save to .npz
        np.savez(npz_path, **data)
        print(f"{npz_path} was saved successfully.")
    else:
        print(f"{npz_path} already exists. Conversion skipped.")
"""
# Example usage:
if __name__ == "__main__":
    mat_file = "C:/Users/Usuario/Desktop/TFM/TASK 1/HDG_Poisson_Python/T1_HDG_Poisson/referenceElement/positionFeketeNodesTri2D_EZ4U.mat" # Replace with the path to your .mat file
    npz_file = "C:/Users/Usuario/Desktop/TFM/TASK 1/HDG_Poisson_Python/T1_HDG_Poisson/referenceElement/positionFeketeNodesTri2D_EZ4U.npz"   # Replace with the desired .npz output path
    
    # Call the function
    #mat_to_npz(mat_file, npz_file)"""

def print_npz(npz_path):
    """
    Loads and prints the contents of a .npz file.
    """
    data = np.load(npz_path, allow_pickle=True) 
    print("Keys in npz file:", data.files)
    for key in data.files:
        print(f"Key: {key}\nData:\n{data[key]}\n")


