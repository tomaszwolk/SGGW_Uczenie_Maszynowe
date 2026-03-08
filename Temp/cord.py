import csv
import json
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load the dataset
# Note: other available arguments include 'max_samples', etc
dataset = load_from_hub("Voxel51/consolidated_receipt_dataset")
print(type(dataset))
# # Launch the App
# session = fo.launch_app(dataset)

# session.wait()

# Ścieżka do pliku wyjściowego
csv_file = "dane_faktury.csv"

# Otwieramy plik CSV do zapisu
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Nagłówki kolumn
    writer.writerow(["image_path", "text", "label", "bbox"])

    # Iterujemy po próbkach z FiftyOne
    for sample in dataset:
        # Ścieżka do obrazka (przydatne, jeśli będziesz chciał go wczytać)
        img_path = sample.filepath
        
        # Sprawdzamy, czy próbka ma adnotacje (ground_truth)
        if sample.ground_truth:
            for detection in sample.ground_truth.detections:
                # detection.label to np. "menu.nm"
                # detection.bounding_box to [top-left-x, top-left-y, width, height]
                
                # CORD w FiftyOne może przechowywać tekst w atrybutach,
                # sprawdźmy czy istnieje pole, które może zawierać OCR
                # Często w CORD tekst jest ukryty, ale dla uproszczenia załóżmy, 
                # że wyciągamy label.
                
                # UWAGA: W CORD z FiftyOne, tekst z OCR może nie być bezpośrednio dostępny 
                # w obiekcie detection (zależy od wersji datasetu). 
                # Jeśli go nie ma, będziesz musiał sam zrobić OCR na wycinku obrazka.
                
                label = detection.label
                bbox = json.dumps(detection.bounding_box) # zapiszmy bbox jako string
                
                # Tutaj symulacja - normalnie w 'detection' szukalibyśmy pola z tekstem
                # Jeśli dataset go nie ma, musisz wyciąć fragment obrazka i puścić OCR.
                extracted_text = "BRAK_TEKSTU_W_METADANYCH" 
                
                # Zapisujemy wiersz
                writer.writerow([img_path, extracted_text, label, bbox])

print(f"Zapisano dane do {csv_file}")
