import os
import numpy as np
import pandas as pd


import os
import numpy as np
import pandas as pd

def loadData(pathDir):
    """
    Carga datos de archivos .parquet y .npz en un directorio especificado, y balancea las etiquetas.

    Args:
        pathDir (str): Ruta al directorio que contiene los archivos.

    Returns:
        tuple: Una lista de listas con ventanas EEG (windows), una lista con etiquetas balanceadas y una lista de grupos.
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

    for i, (parquet, npz) in enumerate(zip(parquets, npzs)):  # for patient in patients
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
        print(f"Archivo npz cargado: {npz}")
        EEG_win = data["EEG_win"]
        EEG_segments = [EEG_win[i, :, :] for i in range(EEG_win.shape[0])]

        # Balancear etiquetas y asociar ventanas y grupos
        label_0_indices = [idx for idx, lbl in enumerate(label_list) if lbl == 0]
        label_1_indices = [idx for idx, lbl in enumerate(label_list) if lbl == 1]

        min_count = min(len(label_0_indices), len(label_1_indices))
        balanced_indices = label_0_indices[:min_count] + label_1_indices[:min_count]

        balanced_labels = [label_list[idx] for idx in balanced_indices]
        balanced_windows = [EEG_segments[idx] for idx in balanced_indices]
        balanced_groups = [i + 1 for _ in balanced_indices]

        # Almacenar resultados balanceados
        labels.extend(balanced_labels)
        windows.extend(balanced_windows)
        groups.extend(balanced_groups)

    print(f"Metadatos almacenados: {len(labels)}")
    print(f"Ventanas EEG almacenadas: {len(windows)}")
    return windows, labels, groups

if __name__ == "__main__":
    # Ruta al directorio
    pathDir = './input_reduit'

    # Llamada a la función
    windows, metadata, groups = loadData(pathDir)
    print(metadata[:100])


if __name__ == "__main__":
    # Ruta al directorio
    pathDir = './input_reduit'

    # Llamada a la función
    windows, metadata, groups = loadData(pathDir)
    print(metadata[0:10000])
    zeroes = [idx for idx, lbl in enumerate(metadata) if lbl == 0]
    print(len(zeroes))
    ones = [idx for idx, lbl in enumerate(metadata) if lbl == 1]
    print(len(ones))
    print(windows[0].shape)
    print(groups)
