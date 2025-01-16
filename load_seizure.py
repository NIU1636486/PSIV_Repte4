import os
import numpy as np
import pandas as pd

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

        # Determinar el 15% de índices a eliminar
        num_windows = len(label_list)
        num_to_remove = int(0.45 * num_windows)
        indices_to_remove = set(np.random.choice(num_windows,num_to_remove, replace=False))

        # Filtrar datos
        filtered_labels = [label for i, label in enumerate(label_list) if i not in indices_to_remove]
        filtered_windows = [window for i, window in enumerate(EEG_segments) if i not in indices_to_remove]
        filtered_groups = [group for i, group in enumerate(group_indices) if i not in indices_to_remove]

        # Almacenar resultados
        labels.extend(filtered_labels)
        windows.extend(filtered_windows)
        groups.extend(filtered_groups)

        # Salir después de procesar "chb02" (opcional)
        if parquet.split("_")[0] == "chb20":
            break

    print(f"Metadatos almacenados: {len(labels)}")
    print(f"Ventanas EEG almacenadas: {len(windows)}")
    print(f"Grupos almacenados: {len(groups)}")

    return windows, labels, groups

if __name__ == "__main__":
    # Ruta al directorio
    pathDir = './input'

    # Llamada a la función
    windows, metadata, groups = loadData(pathDir)
    #print(len(windows), len(metadata), len(groups))
