import os
import numpy as np
import pandas as pd
import random

def loadData(pathDir):
    """
    Carga datos de archivos .parquet y .npz en un directorio especificado.

    Args:
        pathDir (str): Ruta al directorio que contiene los archivos.

    Returns:
        tuple: Una lista de listas con ventanas EEG (windows), una lista de etiquetas (labels),
               y una lista de índices de grupo (groups).
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
        # Verificar correspondencia entre archivos
        if parquet.split('_')[0] != npz.split('_')[0]:
            print("Error: Archivos no coinciden")
            continue

        # Leer archivo parquet
        parquet_path = os.path.join(pathDir, parquet)
        meta = pd.read_parquet(parquet_path, engine='fastparquet')
        meta["Index"] = meta.index
        print(f"Archivo parquet cargado: {parquet}")

        # Leer archivo npz
        npz_path = os.path.join(pathDir, npz)
        data = np.load(npz_path, allow_pickle=True)
        print(f"Archivo npz cargado: {npz}")

        # Obtener ventanas EEG y etiquetas
        EEG_win = data["EEG_win"]
        EEG_segments = [EEG_win[i, :, :] for i in range(EEG_win.shape[0])]

        # Asegurarse de que las dimensiones coincidan
        if len(EEG_segments) != len(meta):
            print("Error: El número de ventanas no coincide con los metadatos")
            continue

        # Agrupar los datos y eliminar un porcentaje de grupos
        grouped = meta.groupby([meta.columns[0], meta.columns[2], meta.columns[3]])
        group_keys = list(grouped.groups.keys())  # Obtener las claves de los grupos
        num_groups = len(group_keys)
        num_to_remove = int(0.20 * num_groups)  # Calcular el 15%
        groups_to_remove = random.sample(group_keys, num_to_remove)  # Seleccionar grupos al azar

        # Filtrar los datos para eliminar los grupos seleccionados
        meta_filtered = meta[~meta[[meta.columns[0], meta.columns[2], meta.columns[3]]]
                             .apply(tuple, axis=1).isin(groups_to_remove)].reset_index(drop=True)
        EEG_filtered = [EEG_segments[i] for i in range(len(EEG_segments)) if i in meta_filtered.index]

        # Actualizar etiquetas y grupos
        label_list = meta_filtered[meta.columns[0]].tolist()
        grouped_filtered = meta_filtered.groupby([meta.columns[0], meta.columns[2], meta.columns[3]])
        group_indices = [group_id for group_id, (_, group_data) in enumerate(grouped_filtered)
                         for _ in range(len(group_data))]

        # Almacenar resultados
        labels.extend(label_list)
        windows.extend(EEG_filtered)
        groups.extend(group_indices)

        del label_list, EEG_filtered, group_indices

        # Salir después de procesar "chb02" (opcional)
        #if parquet.split("_")[0] == "chb10":
            #break

    print(f"Metadatos almacenados: {len(labels)}")
    print(f"Ventanas EEG almacenadas: {len(windows)}")
    print(f"Grupos almacenados: {len(groups)}")

    return windows, labels, groups

if __name__ == "__main__":
    # Ruta al directorio
    pathDir = './input'

    # Llamada a la función
    windows, metadata, groups = loadData(pathDir)
