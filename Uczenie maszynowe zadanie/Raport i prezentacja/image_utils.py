from IPython.display import display
import json


def show_image(dataset, split_name, target_image_id, use_system_viewer=False):
    """
    Wyświetla zdjęcie z danego zbioru o określonym image_id.

    Argumenty:
    dataset -- słownik datasetów (Hugging Face DatasetDict)
    split_name -- nazwa zbioru ('train', 'test', 'validation')
    target_image_id -- szukane ID zdjęcia
    """
    if split_name not in dataset:
        print(f"Błąd: Nieznany split '{split_name}'. Dostępne: {list(dataset.keys())}")
        return

    print(f"Szukam zdjęcia o ID {target_image_id} w zbiorze '{split_name}'...")

    found = False
    for record in dataset[split_name]:
        try:
            # Parsowanie pola ground_truth
            if isinstance(record['ground_truth'], str):
                gt_data = json.loads(record['ground_truth'])
            else:
                gt_data = record['ground_truth']

            meta = gt_data.get('meta', {})

            # Sprawdzamy ID (int lub str)
            current_id = meta.get('image_id')

            # Porównanie bezpieczne dla typów (int vs str)
            if str(current_id) == str(target_image_id):
                print(f"Znaleziono zdjęcie! ID: {current_id}")
                if 'image' in record:
                    if use_system_viewer:
                        record['image'].show()
                    else:
                        display(record['image'])
                else:
                    print("Uwaga: Rekord nie zawiera pola 'image'.")
                found = True
                break
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            print(f"Błąd przetwarzania rekordu: {e}")  # Opcjonalne debugowanie
            continue

    if not found:
        print(f"Nie znaleziono zdjęcia o ID {target_image_id} w zbiorze '{split_name}'.")


if __name__ == "__main__":
    import datasets
    from pathlib import Path

    # Wczytanie danych z dysku jeśli istnieją, jeśli nie to pobierz ze strony huggingface
    if Path("cord-v2-dataset").exists():
        dataset = datasets.load_from_disk("cord-v2-dataset")
    else:
        dataset = datasets.load_dataset("naver-clova-ix/cord-v2")
        dataset.save_to_disk("cord-v2-dataset")  # Zapisz dane na dysku na przyszłość

    # Przykład użycia funkcji show_image
    show_image(dataset, split_name="train", target_image_id=6, use_system_viewer=True)
