import os
import numpy as np
import pyarrow.parquet as pq
import gc

def loadData(pathDir):
    """
    Carga datos de archivos .parquet y .npz en un directorio especificado usando solo numpy.

    Args:
        pathDir (str): Ruta al directorio que contiene los archivos.

    Returns:
        tuple: Un arreglo de numpy con ventanas EEG (windows) y un arreglo de numpy con metadatos.
    """
    files = os.listdir(pathDir)

    # Filtrar archivos .parquet y .npz
    parquets = [file for file in files if file.endswith('.parquet')]
    npzs = [file for file in files if file.endswith('.npz')]
    parquets.sort()
    npzs.sort()

    # Inicializar arrays de numpy vacíos con tipo adecuado
    labels = np.empty(0, dtype=np.int32)  # Asumiendo que las etiquetas son enteros
    windows = np.empty((0, 0, 0), dtype=np.float32)  # Este será ajustado en función de las dimensiones de EEG_win

    for parquet, npz in zip(parquets, npzs):
        # Verificar coincidencia entre nombres de archivos
        if parquet.split('_')[0] != npz.split('_')[0]:
            print("Error: Archivos no coinciden")
            continue

        # Leer archivo parquet
        parquet_path = os.path.join(pathDir, parquet)
        table = pq.read_table(parquet_path)
        meta = table.to_pandas().to_numpy()
        print(f"Archivo parquet cargado: {parquet}")
        label_list = meta[:, 0]  # Asumimos que la primera columna contiene las etiquetas

        # Cargar archivo npz
        npz_path = os.path.join(pathDir, npz)
        data = np.load(npz_path, allow_pickle=True)
        EEG_win = data["EEG_win"]
        print(EEG_win.shape)

        # Concatenar las etiquetas y ventanas EEG con numpy
        labels = np.concatenate((labels, label_list))
        EEG_segments = np.array([EEG_win[i, :, :] for i in range(EEG_win.shape[0])])
        windows = np.concatenate((windows, EEG_segments), axis=0)

        # Liberar memoria
        del EEG_win, EEG_segments, label_list, data
        gc.collect()

        # if parquet.split("_")[0] == "chb12":
        #     break  # Puedes descomentar esta línea si necesitas parar después de un archivo específico

    print(f"Metadatos almacenados: {len(labels)}")
    print(f"Ventanas EEG almacenadas: {len(windows)}")
    return windows, labels

if __name__ == "__main__":
    # Ruta al directorio
    pathDir = './input_reduit'

    # Llamada a la función
    windows, metadata = loadData(pathDir)
    print(metadata[0:10000])
