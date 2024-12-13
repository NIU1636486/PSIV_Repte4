import os
import numpy as np
import pandas as pd
import gc

def loadData(pathDir):
    """
    Carga datos de archivos .parquet y .npz en un directorio especificado.

    Args:
        pathDir (str): Ruta al directorio que contiene los archivos.

    Returns:
        tuple: Una lista de ventanas EEG (windows) y una lista de metadatos.
    """
    files = os.listdir(pathDir)

    # Filtrar archivos .parquet y .npz
    parquets = sorted([file for file in files if file.endswith('.parquet')])
    npzs = sorted([file for file in files if file.endswith('.npz')])

    labels = []
    windows = []

    for parquet, npz in zip(parquets, npzs):
        # Verificar que los archivos coinciden
        if parquet.split('_')[0] != npz.split('_')[0]:
            print(f"Error: Archivos no coinciden -> {parquet}, {npz}")
            continue

        # Leer archivo parquet
        parquet_path = os.path.join(pathDir, parquet)
        meta = pd.read_parquet(parquet_path, engine='fastparquet')
        print(f"Archivo parquet cargado: {parquet}")
        label_list = meta.iloc[:, 0].to_numpy()  # Use NumPy array for labels

        # Cargar archivo npz
        npz_path = os.path.join(pathDir, npz)
        data = np.load(npz_path, allow_pickle=True, mmap_mode='r')  # Use mmap_mode
        print(f"Archivo npz cargado: {npz}")
        EEG_win = data["EEG_win"]

        # Almacenar directamente los resultados sin listas intermedias
        labels.extend(label_list.tolist())
        windows.append(EEG_win)  # Keep the NumPy array as-is

        # Liberar memoria intermedia
        del label_list, data, EEG_win
        gc.collect()

    print(f"Metadatos almacenados: {len(labels)}")
    print(f"Ventanas EEG almacenadas: {len(windows)}")
    return windows, labels
