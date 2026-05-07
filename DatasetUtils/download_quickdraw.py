#!/usr/bin/env python3
"""
Skrypt do pobierania i rozpakowania Quick Draw dataset.
Pobiera wszystkie klasy .npy i rozpakuje obrazki do osobnych plików 32x32.
"""

import os
import urllib.request
import numpy as np
from pathlib import Path
from PIL import Image
import time

# Konfiguracja
DEST_FOLDER = "/tmp/quick_draw_dataset"
UNPACKED_FOLDER = os.path.join(DEST_FOLDER, "unpacked")
NPY_FOLDER = os.path.join(DEST_FOLDER, "npy_files")

import os
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Pobiera pojedynczy plik z Google Cloud Storage."""
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Tworzymy folder, jeśli nie istnieje
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)

    print(f"Pobieranie: {source_blob_name}...")
    blob.download_to_filename(destination_file_name)
    return f"Zakończono: {source_blob_name}"


def download_all_simplified(target_dir=DEST_FOLDER + "/npy_files"):
    bucket_name = "quickdraw_dataset"
    prefix = "full/numpy_bitmap/"

    # Klient anonimowy wystarczy do publicznych danych
    client = storage.Client.create_anonymous_client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)

    # Filtrujemy tylko pliki .npy
    npy_files = [blob.name for blob in blobs if blob.name.endswith(".npy")]

    print(f"Znaleziono {len(npy_files)} plików do pobrania.")

    # Używamy ThreadPoolExecutor dla przyspieszenia (odpowiednik gsutil -m)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for blob_name in npy_files:
            # Tworzymy lokalną ścieżkę (usuwamy prefix z nazwy pliku)
            local_name = os.path.join(target_dir, os.path.basename(blob_name))
            futures.append(executor.submit(download_blob, bucket_name, blob_name, local_name))

        for future in futures:
            future.result()


def create_folders():
    os.makedirs(NPY_FOLDER, exist_ok=True)
    os.makedirs(UNPACKED_FOLDER, exist_ok=True)
    print(f"✓ Foldery utworzone: {NPY_FOLDER}, {UNPACKED_FOLDER}")


def download_npy_files():
    """Pobieranie wszystkich plików .npy z Quick Draw dataset."""
    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

    urllib.request.urlretrieve(base_url, NPY_FOLDER)  # Pobranie listy plików

    print("Pobieranie zakończone.")


def unpack_npy_files():
    """Rozpakowanie plików .npy na pojedyncze obrazki 32x32."""
    npy_files = [f for f in os.listdir(NPY_FOLDER) if f.endswith(".npy")]

    print(f"\n📦 Rozpakowanie {len(npy_files)} plików .npy...")

    total_images = 0

    for file_idx, npy_file in enumerate(npy_files, 1):
        class_name = npy_file[:-4]  # Usunięcie .npy
        file_path = os.path.join(NPY_FOLDER, npy_file)

        try:
            print(f"[{file_idx}/{len(npy_files)}] Rozpakowanie: {class_name}...", end=" ", flush=True)

            # Wczytanie tablicy numpy
            data = np.load(file_path)
            print(f"({data.shape[0]} obrazków)", end=" ", flush=True)

            # Tworzenie folderu klasy
            class_folder = os.path.join(UNPACKED_FOLDER, class_name)
            os.makedirs(class_folder, exist_ok=True)

            # Rozpakowanie każdego obrazka
            for img_idx, image_data in enumerate(data):
                # Normalizacja do zakresu 0-255
                image_array = image_data.reshape(28, 28).astype(np.uint8)

                # Tworzenie obrazka PIL i zapisanie
                image = Image.fromarray(image_array, mode="L")

                # Zapisanie jako PNG
                img_path = os.path.join(class_folder, f"{img_idx:06d}.png")
                image.save(img_path)

            total_images += data.shape[0]
            print("✓")

        except Exception as e:
            print(f"✗ (Błąd: {e})")


def main():
    """Główna funkcja."""
    print("=" * 60)
    print("Quick Draw Dataset - Pobieranie i rozpakowanie")
    print("=" * 60)

    create_folders()
    download_all_simplified()
    unpack_npy_files()

    print("\n" + "=" * 60)
    print("✓ Pobieranie i rozpakowanie zakończone!")
    print(f"📁 Folder: {UNPACKED_FOLDER}")
    print("=" * 60)


if __name__ == "__main__":
    main()
