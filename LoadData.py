import os
import numpy as np
import pandas as pd


def load_data(pathDir):
    """
    Carga datos de archivos .parquet y .npz en un directorio especificado.

    Args:
        pathDir (str): Ruta al directorio que contiene los archivos.

    Returns:
        tuple: Una lista de listas con ventanas EEG (windows) y una lista de listas con metadatos.
    """
    
    files = os.listdir(pathDir)

    # Filtrar archivos .parquet y .npz
    parquets = [file for file in files if file.endswith('.parquet')]
    npzs = [file for file in files if file.endswith('.npz')]

    metadata = []
    windows = []

    for parquet, npz in zip(parquets, npzs):
        # Leer archivo parquet
        parquet_path = os.path.join(pathDir, parquet)
        meta = pd.read_parquet(parquet_path, engine='fastparquet')
        print(f"Archivo parquet cargado: {parquet}")
        meta_list = meta.values.tolist()
        
        # Cargar archivo npz
        npz_path = os.path.join(pathDir, npz)
        data = np.load(npz_path, allow_pickle=True)
        print(f"Archivo parquet cargado: {npz}")
        EEG_win = data["EEG_win"]
        EEG_segments = [EEG_win[i, :, :] for i in range(EEG_win.shape[0])]

        # Almacenar resultados
        metadata.append(meta_list)
        windows.append(EEG_segments)

    return windows, metadata


# Ruta al directorio
pathDir = 'input'

# Llamada a la función
windows, metadata = load_data(pathDir)
