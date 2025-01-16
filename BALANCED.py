import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import resample

def loadData(pathDir):
    """
    Carga datos de archivos .parquet y .npz en un directorio especificado y balancea las etiquetas.

    Args:
        pathDir (str): Ruta al directorio que contiene los archivos.

    Returns:
        tuple: Una lista de listas con ventanas EEG (windows), una lista de etiquetas (labels),
               y una lista de índices de grupo (groups), balanceados según las etiquetas.
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
        print(f"Archivo parquet cargado: {parquet}")

        # Leer archivo npz
        npz_path = os.path.join(pathDir, npz)
        data = np.load(npz_path, allow_pickle=True)
        print(f"Archivo npz cargado: {npz}")

        # Obtener ventanas EEG y etiquetas
        EEG_win = data["EEG_win"]
        EEG_segments = [EEG_win[i, :, :] for i in range(EEG_win.shape[0])]
        
        meta_list = meta.values.tolist()
        label_list = [label[0] for label in meta_list]

        # Asegurarse de que las dimensiones coincidan
        if len(EEG_segments) != len(meta_list):
            print("Error: El número de ventanas no coincide con los metadatos")
            continue

        # Agrupar los datos
        grouped = meta.groupby([meta.columns[0], meta.columns[2], meta.columns[3]])

        # Crear un diccionario para mapear índices de grupo
        group_idx_map = {}
        for group_idx, (group_keys, group_data) in enumerate(grouped):
            for idx in group_data.index:
                group_idx_map[idx] = group_idx

        # Asignar índices de grupo a las ventanas EEG
        group_indices = [group_idx_map[idx] for idx in meta.index]

        # Filtrar datos
        labels.extend(label_list)
        windows.extend(EEG_segments)
        groups.extend(group_indices)

        # Salir después de procesar "chb10" (opcional)
        if parquet.split("_")[0] == "chb10":
            break

    # Balancear los datos
    windows, labels, groups = balanceData(windows, labels, groups)

    print(f"Metadatos almacenados: {len(labels)}")
    print(f"Ventanas EEG almacenadas: {len(windows)}")
    print(f"Grupos almacenados: {len(groups)}")

    return windows, labels, groups

def balanceData(windows, labels, groups):
    """
    Balancea los datos según las etiquetas.

    Args:
        windows (list): Lista de ventanas EEG.
        labels (list): Lista de etiquetas.
        groups (list): Lista de índices de grupo.

    Returns:
        tuple: Datos balanceados (windows, labels, groups).
    """
    # Combinar datos para procesamiento conjunto
    data = list(zip(windows, labels, groups))
    
    # Separar por clases
    class_data = {label: [] for label in set(labels)}
    for window, label, group in data:
        class_data[label].append((window, group))
    
    # Encontrar la frecuencia mínima
    min_samples = min(len(samples) for samples in class_data.values())
    
    # Balancear las clases
    balanced_windows = []
    balanced_labels = []
    balanced_groups = []
    
    for label, samples in class_data.items():
        if len(samples) > min_samples:
            # Submuestreo para clases mayoritarias
            sampled = resample(samples, replace=False, n_samples=min_samples, random_state=42)
        else:
            # Sobremuestreo para clases minoritarias
            sampled = resample(samples, replace=True, n_samples=min_samples, random_state=42)
        
        # Separar datos balanceados
        for window, group in sampled:
            balanced_windows.append(window)
            balanced_labels.append(label)
            balanced_groups.append(group)
    
    print(f"Clases balanceadas: {Counter(balanced_labels)}")
    return balanced_windows, balanced_labels, balanced_groups

if __name__ == "__main__":
    # Ruta al directorio
    pathDir = './input'

    # Llamada a la función
    windows, metadata, groups = loadData(pathDir)
