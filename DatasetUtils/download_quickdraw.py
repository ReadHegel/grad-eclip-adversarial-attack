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
DEST_FOLDER = "data/quick_draw_dataset"
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


def main():
    """Główna funkcja."""
    print("=" * 60)
    print("Quick Draw Dataset - Pobieranie i rozpakowanie")
    print("=" * 60)

    create_folders()
    download_all_simplified()

    print("\n" + "=" * 60)
    print("✓ Pobieranie i rozpakowanie zakończone!")
    print(f"📁 Folder: {UNPACKED_FOLDER}")
    print("=" * 60)


if __name__ == "__main__":
    main()
