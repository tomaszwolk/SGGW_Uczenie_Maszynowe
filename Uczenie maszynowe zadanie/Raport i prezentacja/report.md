# Raport Projektowy: Semantyczna Klasyfikacja Pozycji Paragonowych (Food vs. Drinks)

**Autor:** Tomasz Wołk  
**Przedmiot:** Uczenie Maszynowe (SGGW)  
**Data:** 22 Grudnia 2025

---

## 1. Wstęp i Cel Projektu

Celem projektu jest stworzenie modelu uczenia maszynowego zdolnego do automatycznej klasyfikacji pozycji paragonowych na trzy kategorie semantyczne:
1.  **FOOD** (Jedzenie)
2.  **DRINK** (Napoje)
3.  **OTHER** (Inne - np. opłaty, opakowania)

Problem klasyfikacji paragonów jest istotny z punktu widzenia automatyzacji analizy wydatków, systemów lojalnościowych oraz aplikacji dietetycznych. Głównym wyzwaniem jest krótka, często niejednoznaczna forma tekstu na paragonach (np. skróty, brak kontekstu) oraz wielojęzyczność danych (angielski, indonezyjski i inne).

W projekcie wykorzystano nowoczesne modele językowe oparte na architekturze Transformer (BERT, RoBERTa), badając wpływ augmentacji danych na jakość predykcji, ze szczególnym uwzględnieniem klasy mniejszościowej (DRINK).

---

## 2. Dane i Przetwarzanie (EDA)

### Źródło Danych
Wykorzystano zbiór danych **CORD-v2** (Consolidated Receipt Dataset) dostępny w repozytorium Hugging Face (`naver-clova-ix/cord-v2`), zawierający obrazy paragonów oraz ich anotacje w formacie JSON.

### Przygotowanie Danych
1.  **Ekstrakcja Tekstu**: Z surowych plików JSON wyodrębniono nazwy produktów (pole `nm` w strukturze `menu`).
2.  **Czyszczenie**:
    - Usunięto zduplikowane nazwy produktów.
    - Odrzucono wpisy składające się wyłącznie z cyfr lub krótsze niż 2 znaki.
    - Znormalizowano tekst (usunięcie zbędnych spacji).
3.  **Etykietowanie**: Ze względu na brak gotowych etykiet klasyfikacyjnych w źródle CORD, przeprowadzono etykietowanie zewnętrzne (przy pomocy LLM), przypisując każdej pozycji jedną z trzech klas: FOOD, DRINK, OTHER.

### Analiza Eksploracyjna (EDA)
Po oczyszczeniu otrzymano **1506 unikalnych produktów**. Rozkład klas w zbiorze jest **niezbalansowany**:
- **FOOD**: 1050 (ok. 70%)
- **DRINK**: 322 (ok. 21%)
- **OTHER**: 132 (ok. 9%)

Niezbalansowanie to stanowi główne wyzwanie, sugerując ryzyko, że model będzie faworyzował klasę większościową (FOOD) kosztem mniejszościowych.

---

## 3. Metodologia i Modele

W projekcie przetestowano trzy podejścia modelowania, aby zweryfikować hipotezę o wpływie augmentacji i doboru architektury na wyniki.

W każdym przypadku użyto biblioteki `transformers` (Hugging Face) i podziału danych: Train (80%) / Test (20%) z zachowaniem proporcji klas (`stratify`).

### Model 1: DistilBERT Multilingual (Baseline)
- **Model**: `distilbert-base-multilingual-cased`
- **Dane**: Oryginalny zbiór treningowy.
- **Cel**: Ustalenie punktu odniesienia (baseline). DistilBERT wybrano ze względu na szybkość trenowania i wsparcie dla wielu języków.

### Model 2: DistilBERT + Augmentacja
- **Model**: `distilbert-base-multilingual-cased`
- **Metoda Augmentacji (Syntaktyczna/Regułowa)**:
    - Dla klasy **DRINK**: Dodano losowe prefiksy (np. "Ice ", "Hot ") i sufiksy (np. " (L)", " Bottle").
    - Dla klasy **OTHER**: Dodano określenia typu "Charge", "Tax", "Extra".
    - Losowa zamiana na wielkie litery (uppercase).
- **Cel**: Zwiększenie liczebności klas mniejszościowych i sprawdzenie, czy model lepiej nauczy się cech dystynktywnych napojów.

### Model 3: XLM-RoBERTa + Augmentacja
- **Model**: `xlm-roberta-base`
- **Dane**: Zbiór po augmentacji.
- **Cel**: Sprawdzenie, czy większy, bardziej zaawansowany model wielojęzyczny (XLM-R) lepiej poradzi sobie z generalizacją niż mniejszy DistilBERT, zwłaszcza w obliczu sztucznie generowanych danych.

---

## 4. Wyniki i Porównanie

Modele oceniano przy użyciu metryk: **Accuracy** (Dokładność), **F1-Macro** (średnia harmoniczna precyzji i czułości, ważna przy nierównych klasach) oraz **Recall** dla poszczególnych klas.

### Tabela Wyników

| Model | Accuracy | F1-Score (Macro) | Recall (DRINK) | Recall (FOOD) | Recall (OTHER) | Validation Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. DistilBERT Baseline** | 86% | 0.77 | 71% | **94%** | 58% | 0.60 |
| **2. DistilBERT Aug.** | 87% | 0.79 | 77% | 93% | 62% | **0.84** (Overfitting) |
| **3. XLM-RoBERTa Aug.** | **91%** | **0.84** | **85%** | **96%** | **62%** | 0.64 |

### Analiza

1.  **Baseline (Model 1)**:
    - Osiągnął przyzwoitą dokładność globalną (86%), ale słabo radził sobie z klasami mniejszościowymi.
    - Recall dla DRINK wynosił tylko 71%, co oznacza, że model 30% napojów błędnie klasyfikował jako jedzenie.

2.  **Efekt Augmentacji (Model 2)**:
    - Zastosowanie prostych reguł augmentacji podniosło Recall dla DRINK do 77%.
    - Drastycznie wzrósł jednak **Validation Loss (0.84)**, co wskazuje na mocne **przeuczenie (overfitting)**. Model nauczył się na pamięć reguł augmentacji (np. słowa "Ice" zawsze oznaczają napój), co może nie działać w rzeczywistych, bardziej zróżnicowanych danych.

3.  **Zwycięzca: XLM-RoBERTa (Model 3)**:
    - Zmiana architektury na XLM-R przy zachowaniu augmentacji dała najlepsze rezultaty.
    - **Accuracy 91%**, a co najważniejsze, **Recall dla DRINK wzrósł aż do 85%**.
    - Model ten lepiej generalizuje wiedzę i jest mniej podatny na overfitting niż DistilBERT (Validation Loss rósł wolniej i łagodniej).

---

## 5. Wnioski

1.  **Architektura ma znaczenie**: XLM-RoBERTa, będąca modelem większym i trenowanym na większym korpusie wielojęzycznym, poradziła sobie znacznie lepiej ze specyficznymi, krótkimi tekstami paragonowymi niż lżejszy DistilBERT.
2.  **Augmentacja pomogła, ale z ryzykiem**: Prosta augmentacja tekstowa skutecznie zbalansowała klasy, poprawiając wykrywalność napojów. Niesie jednak ryzyko przeuczenia na specyficznych słowach kluczowych (np. "Ice").
3.  **Rekomendacja**: Do środowiska produkcyjnego rekomendowany jest model **XLM-RoBERTa**. Spełnia on wymagania zadania z wysoką skutecznością (>90%).
4.  **Dalsze kroki**: Aby jeszcze bardziej poprawić wyniki dla klasy OTHER (nadal tylko 62% recall), należałoby zebrać więcej rzeczywistych przykładów tej klasy lub zastosować bardziej zaawansowaną, semantyczną augmentację (np. generowanie przykładów przez GPT-4), zamiast prostych reguł leksykalnych.
