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
    groups = []

    for i, (parquet, npz) in enumerate(zip(parquets, npzs)):
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
        # Almacenar resultados
        labels.extend(label_list)
        windows.extend(EEG_segments)
        
        #if parquet.split("_")[0] == "chb01":
            #break

    # agrupar a nivell de label, interval i pacient
    grouped = meta.groupby([meta.columns[0], meta.columns[2], meta.columns[3]])

    # Crear un diccionario para mapear los índices de los grupos
    group_idx_map = {}
    for group_idx, (group_keys, group_data) in enumerate(grouped):
        # Asignamos a cada grupo su índice
        for idx in group_data.index:
            group_idx_map[idx] = group_idx
    
    # Asignar el índice del grupo a cada fila de metadata, manteniendo el orden
    for idx in meta.index:
        group_idx = group_idx_map.get(idx, -1)  # Si no está en un grupo, asignar -1
        groups.append(group_idx)

    print(f"Metadatos almacenados: {len(labels)}")
    print(f"Ventanas EEG almacenadas: {len(windows)}")
    return windows, labels, groups

if __name__ == "__main__":
    # Ruta al directorio
    pathDir = './input'

    # Llamada a la función
    windows, metadata, groups = loadData(pathDir)
    #print(groups)
