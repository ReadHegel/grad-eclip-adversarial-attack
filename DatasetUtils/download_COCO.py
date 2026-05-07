import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def prepare_coco_dataset(output_dir="coco_test_set", num_samples=1000):
    # 1. Stworzenie struktury folderów
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print("Wczytywanie datasetu z Hugging Face (może to chwilę potrwać)...")
    # Używamy splitu 'validation', bo jest lepiej opisany i mniejszy
    dataset = load_dataset("ydshieh/coco_dataset_script", "2017", split="validation", streaming=True)

    metadata = []

    print(f"Pobieranie i zapisywanie {num_samples} obrazów...")

    for i, entry in enumerate(tqdm(dataset.take(num_samples))):
        image = entry["image"]
        # COCO ma zazwyczaj 5 podpisów (captions) na obrazek
        captions = entry["captions"]

        image_filename = f"coco_{i:06d}.jpg"
        image_path = os.path.join(images_dir, image_filename)

        # Zapis obrazka (konwersja do RGB, by uniknąć problemów z obrazami czarno-białymi)
        image.convert("RGB").save(image_path)

        # Dodajemy do metadanych - bierzemy pierwszy podpis jako główny,
        # ale przechowujemy wszystkie
        metadata.append(
            {
                "image_path": image_filename,
                "main_caption": captions[0],
                "all_captions": "|".join(captions),  # rozdzielone kreską pionową
            }
        )

    # 2. Zapis metadanych do CSV
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    print(f"\nGotowe! Dane znajdują się w folderze: {output_dir}")
    print(f"Liczba pobranych próbek: {len(df)}")


if __name__ == "__main__":
    # Możesz zmienić num_samples na większą liczbę, jeśli masz miejsce na dysku
    prepare_coco_dataset(num_samples=500)
