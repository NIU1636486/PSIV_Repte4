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

        # Cargar archivo npz
        npz_path = os.path.join(pathDir, npz)
        data = np.load(npz_path, allow_pickle=True)
        print(f"Archivo npz cargado: {npz}")
        EEG_win = data["EEG_win"]
        
        # Cargar archivo parquet
        parquet_path = os.path.join(pathDir, parquet)
        meta = pd.read_parquet(parquet_path, engine='fastparquet')
        print(f"Archivo parquet cargado: {parquet}")
        meta['filename'] = meta['filename'].str[:5]

        # Agrupar los datos y eliminar un porcentaje de grupos
        grouped = meta.groupby([meta.columns[0], meta.columns[2], meta.columns[3]])
        keys = [ key for key,_ in grouped] 
        label_0 = [ i for i, (key,_) in enumerate(grouped) if key[0] == 0]
        num_to_remove = int(0.7 * len(label_0))  # Calcular el 15%
        groups_to_remove = random.sample(label_0, num_to_remove)  # Seleccionar grupos al azar
        keys = [key for i, key in enumerate(keys) if i not in groups_to_remove]



        meta_list = meta.values.tolist()
        label_list = []
        key_list = []
        windows_list = []
        for i in range(len(meta_list)):
            meta = meta_list[i]
            key_meta = (meta[0], meta[2], meta[3])
            window = EEG_win[i,:,:]
            if key_meta in keys:
                label_list.append(meta[0])
                key_list.append(str(key_meta))
                windows_list.append(window)
        

        # Almacenar resultados
        labels.extend(label_list)
        windows.extend(windows_list)
        groups.extend(key_list)
        
        del label_list, windows_list, key_list

        if parquet.split("_")[0] == "chb20":
            break
    
    #print(groups)
    print(f"Metadatos almacenados: {len(labels)}")
    print(f"Ventanas EEG almacenadas: {len(windows)}")
    print(f"Grupos almacenados: {len(groups)}")

    return windows, labels, groups

if __name__ == "__main__":
    # Ruta al directorio
    pathDir = './input'

    # Llamada a la función
    windows, metadata, groups = loadData(pathDir)
    #print(windows[0:10])
    #print(metadata[0:10])
    #print(groups[0:10])