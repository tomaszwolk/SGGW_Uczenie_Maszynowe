# Dokumentacja Projektu: Semantyczna Klasyfikacja Pozycji Paragonowych (Food vs. Drinks)

## 1. Definicja Tematu
Celem projektu jest stworzenie modelu uczenia maszynowego (Machine Learning), który na podstawie tekstu pozycji widniejącej na paragonie fiskalnym (np. "Caffe Latte", "Spaghetti Bolognese") automatycznie przypisze jej odpowiednią kategorię semantyczną: **Jedzenie (Food)**, **Napój (Drink)** lub **Inne (Other)**.

Projekt skupia się na analizie tekstu (NLP) pochodzącego z rzeczywistych paragonów gastronomicznych, zmagając się z problemami takimi jak skróty, literówki oraz specyficzne nazewnictwo produktów.

## 2. Pytania Badawcze i Hipoteza

### Hipoteza
Wykorzystanie modeli językowych opartych na architekturze Transformer (np. BERT/RoBERTa) pozwoli na skuteczną klasyfikację kategorii produktów na podstawie samej nazwy, osiągając skuteczność (Accuracy) powyżej 90%, nawet w przypadku występowania skrótów i nazw w języku mieszanym (angielski/lokalny).

### Pytania badawcze
1.  Czy model nauczy się rozróżniać napoje od jedzenia na podstawie kontekstu słownego (np. słowa kluczowe "Tea", "Ice", "Plate")?
2.  Jak model poradzi sobie z niejednoznacznymi nazwami (np. zestawy "Combo", które zawierają zarówno jedzenie, jak i napój)?
3.  Czy zastosowanie modelu wielojęzycznego (Multilingual) jest konieczne, biorąc pod uwagę specyfikę zbioru danych CORD (Indonezja/Angielski)?

## 3. Główne Problemy i Metryki Oceny

### Wyzwania (Challenges)
*   **Jakość danych (OCR Errors):** Tekst pochodzi z automatycznego odczytu (OCR), może zawierać błędy (np. "Burgr" zamiast "Burger").
*   **Skróty i kodowanie:** Paragony często używają agresywnych skrótów (np. "Chckn Wings", "Sm. Coke").
*   **Kontekst** Czy modelowi wystarczy zobaczyć tylko nazwę ("Pepsi"), bez informacji o cenie czy sekcji menu.
*   **Niezbalansowane klasy:** W zbiorze danych zazwyczaj jest znacznie więcej pozycji typu "Jedzenie" niż "Napoje".

### Metryki sukcesu
Model będzie oceniany przy użyciu następujących parametrów:
1.  **F1-Score (Macro):** Kluczowa metryka ze względu na potencjalną nierównowagę klas. Pozwoli ocenić, czy model tak samo dobrze radzi sobie z mniejszą klasą "Napoje".
2.  **Accuracy (Dokładność):** Ogólny procent poprawnych klasyfikacji.
3.  **Confusion Matrix (Macierz pomyłek):** Do wizualizacji, jakie produkty są mylone najczęściej (np. czy zupy są klasyfikowane jako napoje).

## 4. Przewidywane Zastosowanie
Tego typu model może znaleźć zastosowanie w systemach FinTech oraz Business Intelligence:
*   **Aplikacje do finansów osobistych:** Automatyczna analiza, ile użytkownik wydaje na alkohol/kawę w stosunku do jedzenia.
*   **Analityka dla restauracji:** Analiza trendów sprzedaży (food-to-beverage ratio) na podstawie surowych danych z kas fiskalnych.
*   **Systemy lojalnościowe:** Przyznawanie punktów tylko za określone kategorie produktów (np. "Kup 5 kaw, 6. gratis").

## 5. Funkcjonalności Aplikacji (Use Case)
W przypadku wdrożenia modelu jako mikroserwisu lub aplikacji demo, główne funkcjonalności to:
1.  **Input:** Użytkownik przesyła zdjęcie paragonu.
2.  **Processing:** Model przetwarza każdą pozycję i przypisuje etykietę oraz pewność predykcji (confidence score).
3.  **Output:** Wykres kołowy prezentujący podział wydatków (np. 70% Jedzenie, 30% Napoje) oraz wyróżnienie pozycji niejednoznacznych.

## 6. Opis Danych (Dataset)

Do trenowania wykorzystany zostanie zbiór **CORD (Consolidated Receipt Dataset)**.

*   **Źródło:** Publicznie dostępny zbiór (np. Hugging Face / FiftyOne).
*   **Charakterystyka:** Tysiące zeskanowanych paragonów z restauracji i sklepów w Indonezji.
*   **Preprocessing i Etykietowanie:**
    *   Z danych surowych zostaną wyekstrahowane pola oznaczone jako `menu.nm` (nazwa pozycji).
    *   **Proces etykietowania (Data Augmentation):** Ponieważ CORD nie posiada etykiet "Food/Drink", zostanie zastosowane podejście *Weak Supervision*. Unikalne nazwy produktów zostaną przepuszczone przez LLM (np. GPT-4o-mini/Gemini) w celu nadania wstępnych etykiet (0: Food, 1: Drink, 2: Other), które posłużą jako Ground Truth do treningu właściwego modelu.

## 7. Model i Architektura

W projekcie zostanie wykorzystane podejście **Transfer Learning** w dziedzinie NLP (Natural Language Processing).

*   **Architektura:** **DistilBERT** lub **XLM-RoBERTa** (wersja base).
    *   *Dlaczego?* Modele oparte na Transformerach świetnie radzą sobie z rozumieniem kontekstu w krótkich tekstach. XLM-RoBERTa jest preferowany ze względu na wielojęzyczny charakter danych CORD (Indonezyjski + Angielski). DistilBERT jest lżejszą alternatywą, szybszą w treningu.
*   **Framework:** Python + biblioteka `transformers` (Hugging Face) + `scikit-learn`.
*   **Metoda treningu:** Fine-tuning. Pobrany zostanie wytrenowany model językowy, a następnie "dotrenowana" zostanie warstwa klasyfikująca (Classification Head) na przygotowanym zbiorze danych z paragonów.

---

### Dodatkowa uwaga do projektu:
*Dane CORD są specyficzne językowo. Model trenowany na danych z Indonezji może gorzej działać na paragonach z polskich sklepów, dlatego rozważam zastosowanie modelu multilingualnego i dodanie polskich przykładów do zbioru treningowego.*