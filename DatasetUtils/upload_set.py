import os
from datasets import load_dataset
from huggingface_hub import HfApi

def upload_adversarial_dataset(local_path, hf_username):
    # 1. Automatyczne wyciąganie nazwy modelu z folderu
    # Przykład: 'openai-vit-b16-adv-recognition-ds' -> 'adv-recognition-openai-vit-b16'
    folder_name = os.path.basename(local_path.strip('/'))
    clean_model_name = folder_name.replace('-ds', '')
    repo_id = f"{hf_username}/{clean_model_name}"

    print(f"🚀 Przygotowanie do uploadu: {local_path} -> {repo_id}")

    try:
        dataset = load_dataset("imagefolder", data_dir=os.path.join(local_path, "images"))
        dataset = dataset.class_encode_column("is_attacked")

        # --- DODAJ TO PONIŻEJ ---
        print("\n" + "="*30)
        print("📊 PODSTAWOWE INFORMACJE")
        print("="*30)

        # 1. Jakie mamy splity (np. train, test) i ile mają rekordów?
        for split_name, split_data in dataset.items():
            print(f"📁 Split: '{split_name}' | Liczba przykładów: {len(split_data)}")

        # 2. Jakie są kolumny (metadata)?
        # Pobieramy nazwy kolumn z pierwszego dostępnego splitu
        first_split = list(dataset.keys())[0]
        columns = dataset[first_split].column_names
        print(f"\n📝 Dostępne kolumny (metadata): {columns}")

        # 3. Podgląd pierwszego elementu (żeby zobaczyć wartości)
        print("\n🔍 Podgląd pierwszego wiersza:")
        print(dataset[first_split][0])
        print("="*30 + "\n")

        dataset.push_to_hub(repo_id, private=False)
        
        print(f"✅ Sukces! Dataset dostępny pod adresem: https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        print(f"❌ Błąd podczas uploadu: {e}")

if __name__ == "__main__":
    USERNAME = "ReadHegel"
    
    folders_to_upload = [
        "DatasetUtils/data/generated_datasets/openai-vit-b16-adv-recognition-ds",
        # "DatasetUtils/data/generated_datasets/openai-vit-b18-adv-recognition-ds", 
    ]

    for path in folders_to_upload:
        if os.path.exists(path):
            upload_adversarial_dataset(path, USERNAME)
        else:
            print(f"⚠️ Folder {path} nie istnieje, pomijam.")