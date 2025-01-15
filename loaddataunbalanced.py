# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:40:23 2025

@author: marcp
"""

import os
import numpy as np
import pandas as pd


def loadData(pathDir):
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
    parquets.sort()
    npzs.sort()

    labels = []
    windows = []

    for parquet, npz in zip(parquets, npzs):
        # Leer archivo parquet
        if parquet.split('_')[0] != npz.split('_')[0]:
            print("Error: Archivos no coinciden")
            continue
        parquet_path = os.path.join(pathDir, parquet)
        meta = pd.read_parquet(parquet_path, engine='fastparquet')
        print(f"Archivo parquet cargado: {parquet}")
        meta_list = meta.values.tolist()
        label_list = [label[0] for label in meta_list]
        # Cargar archivo npz
        npz_path = os.path.join(pathDir, npz)
        data = np.load(npz_path, allow_pickle=True)
        print(f"Archivo parquet cargado: {npz}")
        EEG_win = data["EEG_win"]
        EEG_segments = [EEG_win[i, :, :] for i in range(EEG_win.shape[0])]
        
        # Determinar el 15% de índices a eliminar
        num_windows = len(label_list)
        num_to_remove = int(0.45 * num_windows)
        indices_to_remove = set(np.random.choice(num_windows,num_to_remove, replace=False))

        # Filtrar datos
        filtered_labels = [label for i, label in enumerate(label_list) if i not in indices_to_remove]
        filtered_windows = [window for i, window in enumerate(EEG_segments) if i not in indices_to_remove]
        
        
        
        # Almacenar resultados
        labels.extend(filtered_labels)
        windows.extend(filtered_windows)
       
    print(f"Metadatos almacenados: {len(labels)}")
    print(f"Ventanas EEG almacenadas: {len(windows)}")
    return windows, labels

if __name__ == "__main__":
    # Ruta al directorio
    pathDir = './input_reduit'

    # Llamada a la función
    windows, metadata = loadData(pathDir)
    print(metadata[0:10000])
