import os
import numpy as np
import pandas as pd
import random
import gc

def loadData(pathDir, N):
    """
    Carga datos de archivos .parquet y .npz en un directorio especificado, seleccionando
    aleatoriamente hasta N agrupaciones por paciente basadas en global_interval y label.

    Args:
        pathDir (str): Ruta al directorio que contiene los archivos.
        N (int): Número máximo de agrupaciones a cargar por paciente.

    Returns:
        tuple: Una lista de listas con ventanas EEG (windows), una lista de etiquetas (labels)
        y una lista de grupos (group).
    """
    files = os.listdir(pathDir)

    # Filtrar archivos .parquet y .npz
    parquets = [file for file in files if file.endswith('.parquet')]
    npzs = [file for file in files if file.endswith('.npz')]
    parquets.sort()
    npzs.sort()

    labels = []
    windows = []
    groups = []

    for i, (parquet, npz) in enumerate(zip(parquets, npzs)):
        # Verificar que los archivos correspondan al mismo paciente
        if parquet.split('_')[0] != npz.split('_')[0]:
            print("Error: Archivos no coinciden")
            continue

        # Leer archivo parquet
        parquet_path = os.path.join(pathDir, parquet)
        meta = pd.read_parquet(parquet_path, engine='fastparquet')
        print(f"Archivo parquet cargado: {parquet}")

        # Agrupar por global_interval y label
        grouped = meta.groupby([meta.columns[2], meta.columns[0]])
        selected_groups = random.sample(list(grouped), min(N, len(grouped)))

        # Leer archivo npz
        npz_path = os.path.join(pathDir, npz)
        data = np.load(npz_path, allow_pickle=True)
        print(f"Archivo npz cargado: {npz}")

        EEG_win = data["EEG_win"]

        # Procesar las agrupaciones seleccionadas
        for (global_interval, label), group in selected_groups:
            indices = group.index
            EEG_segments = [EEG_win[idx, :, :] for idx in indices]

            # Almacenar resultados
            labels.extend([label] * len(EEG_segments))
            windows.extend(EEG_segments)
            groups.extend([i + 1] * len(EEG_segments))

        # Liberar memoria de los datos procesados
        del meta, grouped, selected_groups, data, EEG_win
        gc.collect()

        if parquet.split("_")[0] == "chb01":
            break

    print(f"Metadatos almacenados: {len(labels)}")
    print(f"Ventanas EEG almacenadas: {len(windows)}")
    return windows, labels, groups

if __name__ == "__main__":
    # Ruta al directorio
    pathDir = './input'
    
    # Número de agrupaciones a seleccionar por paciente
    N = 10

    # Llamada a la función
    windows, label, group = loadData(pathDir, N)
