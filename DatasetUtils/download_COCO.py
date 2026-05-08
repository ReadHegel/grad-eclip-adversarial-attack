import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def download_jxie_coco(target_base_path="data/coco_test_set", num_samples=5000):
    # 1. Przygotowanie struktury folderów na HDD
    images_dir = os.path.join(target_base_path, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"Pobieranie datasetu jxie/coco_captions do: {target_base_path}")
    
    try:
        # Ładujemy split 'train' lub 'validation' - jxie zazwyczaj ma oba. 
        # Używamy streaming=True dla wydajności.
        dataset = load_dataset("jxie/coco_captions", split="test", streaming=True)
    except Exception as e:
        print(f"Błąd ładowania: {e}")
        return

    metadata = []
    
    print(f"Rozpoczynam zapisywanie obrazów...")
    
    for i, entry in enumerate(tqdm(dataset.take(num_samples))):
        if i % 5 != 0: 
            continue
        # W jxie/coco_captions klucze to zazwyczaj 'image' i 'caption' (lub 'captions')
        image = entry['image']
        
        # Obsługa podpisów (wyciągamy tekst)
        raw_captions = entry.get('caption', entry.get('captions', ["Brak opisu"]))
        
        # Jeśli captions to lista, bierzemy pierwszy, jeśli string - bierzemy całość
        if isinstance(raw_captions, list):
            main_caption = raw_captions[0]
            all_captions_str = "|".join([str(c) for c in raw_captions])
        else:
            main_caption = raw_captions
            all_captions_str = raw_captions

        image_filename = f"coco_{i:06d}.jpg"
        image_path = os.path.join(images_dir, image_filename)
        
        # Zapis obrazu bezpośrednio na HDD
        try:
            image.convert("RGB").save(image_path, quality=95)
        except Exception as e:
            print(f"Pominięto obraz {i} z powodu błędu: {e}")
            continue
        
        metadata.append({
            "image_path": image_filename,
            "main_caption": main_caption,
        })

    # 2. Zapis pliku CSV z metadanymi na HDD
    df = pd.DataFrame(metadata)
    csv_path = os.path.join(target_base_path, "metadata.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nZakończono!")
    print(f"Lokalizacja obrazów: {images_dir}")
    print(f"Plik metadanych: {csv_path}")
    print(f"Łącznie pobrano: {len(df)} próbek.")

if __name__ == "__main__":
    # Możesz zwiększyć num_samples do np. 30000, jeśli HDD ma wystarczająco miejsca
    download_jxie_coco(num_samples=25000)