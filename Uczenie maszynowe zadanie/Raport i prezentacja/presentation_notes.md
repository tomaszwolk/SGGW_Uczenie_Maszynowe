# Prezentacja Projektu: Klasyfikacja Paragonów - Notatki

**Czas:** 5-10 minut

## 1. Slajd Tytułowy
- **Temat:** Semantyczna Klasyfikacja Pozycji Paragonowych (Food vs. Drinks).
- **Autor:** Tomasz Wołk.
- **Cel:** Automatyczne przypisywanie kategorii (Jedzenie, Napój, Inne) do pozycji na paragonie.

Dzień dobry, nazywam się Tomasz Wołk. Chciałbym przedstawić mój projekt zaliczeniowy na przedmiot Uczenie Maszynowe
Semantyczna klasyfikacja pozycji paragonowych.

## 2. Problem i Dane
- **Wyzwanie:** Tekst na paragonach jest trudny (skróty, błędy OCR, mix języków).
- **Zbiór Danych:** CORD-v2 (1506 unikalnych produktów po oczyszczeniu).
- **Problem Niezbalansowania:** 
    - 70% to Jedzenie (FOOD).
    - Tylko 21% to Napoje (DRINK).
    - Ryzyko: Model będzie ignorował napoje i wszystko uznawał za jedzenie.

Projekt opiera się o ogólnodostępną bazę danych CORD, która zawiera skany paragonów z krajów azjatyckich.
Same paragony zostały zeskanowane przy użyciu OCR. 
Wyzwaniem dla takiej bazy danych są skróty, błędy OCR oraz mix języków.
Dodatkowo kategoria jedzenie występuje częściej na paragonach niż napoje, co powoduje niezbalansownaie klas.


## 3. Podejście (Eksperymenty)
10 epok, główny parametr do oceny modelu to F1 score. Dodatkowo sprawdzam dokładność i Validation Loss.
Przetestowano 3 scenariusze:
1.  **Baseline:** Prosty model *DistilBERT* na surowych danych.
2.  **Augmentacja:** Ten sam model, ale dane "sztucznie" powiększone (dodawanie słów np. "Ice", "Bottle" do napojów).
3.  **Lepszy Model:** Zmiana na potężniejszy model *XLM-RoBERTa* + Augmentacja.

Projekt zaczynam od wczytania bibliotek. Skorzystałem z bibliotek transformers do treningu.
Dane wczytuję ze strony huggingface.
Po podejrzeniu danych widać, że zostały one podzielone na 3 zestawy (treningowe, walidacyjne i testowe).
Łącznie zawierają one 1000 zeskanowanych paragonów. Do każdego zdjęcia paragonu dodane są również informacje w sekcji ground truth. To co nas interesuje to są pozycje "nm" (nazwa).
Wyciągam listę interesujących mnie danych z każdego datasetu. To pozwoliło mi pobrać 2142 produkty.
Sprawdzam czy wszystkie pozycje zostały poprawnie pobrane. Widać, że jedna pozycja została pobrana jako lista a nie jako string. Przez co muszę spłaszczyć tą listę. Dzięki czemu mogę usunąć duplikaty - co daje mi 1506 unikalnych produktów.

EDA
Następnie muszę przypisać kategorię do każdej pozycji. Wykorzystuję do tego Gemini 3 Pro.
Usuwam zbędne spacje, usuwam również pozycje, które mają mniej niż 2 znaki oraz wiersze które składają się tylko z cyfr.
Podliczenie ilości pozycji dla każdej kategorii pokazuje duże niezbalansowanie klas. Przewiduję, że będę potrzebował wykonać augmentację danych treningowych. Ale żeby sprawdzić czy daje to pozytywne wyniki, najpierw trenuję model na zbiorze bez augmentacji.

Modelowanie rozpoczynam od mapowania etykiet tekstowych na liczby. Wykorzystuję do tego LabelEncoder, który przypisuje numery zgodnie z kolejnością alfabetyczną.
Dzielę dane na treningowe i testowe w proporcji 80/20. Dodatkowo używam stratify, które gwarantuje, że zostaną zachowane proporcje klas w zbiorach treningowym i testowym.
Podgląd ilości pozycji w zbiorze treningowym i testowym pokazuje, że rzeczywiście proporcje zostały zachowane.
Przewiduję, kilkukrotne trenowanie modelu, dlatego stworzyłem własną klasę ModelTrainer. 
Przyjmuje ona nazwę modelu, ilość epok do treningu, wielkość batcha, jak i ścieżkę do danych treningowych i testowych.
Do tokenizacji używam AutoTokenizer, który sam dobiera sposób tokenizacji na podstawie nazwy modelu.
Model będzie przechodził ewaluację co 1 epokę, a kryterium wyboru najlepszego modelu będzie F1 score.

Pierwszym modelem jest Distilbert, trenowany na 10 epokach.
Patrząc na wyniki widać, że validation loss zaczyna rosnąć po 3 epoce. Natomiast najlepszy wynik F1 jest w 7 epoce. Przy niezłym wyniku dokładności 86%.
Spoglądając na szczegółowe wyniki widać problem niezbalansowania klas. Model prawie nie myli się na kategorii Food, natomiast dla kategorii Drink przeoczył niemal 30% wyników. Kategoria Other jest niewiele lepsza od rzutu monetą.
To wszystko pokazuje problem niezbalansowania klas.

By temu przeciwdziałać dokonuję augmentacji danych treningowych. Do kategorii Drink i Other losowo dodaję prefixy i sufixy z wcześniej stworzonej listy. Dane testowe zostawiam nie zmienione.

Drugim modelem również jest Distilbert, tutaj jedynie zmieniam dane treningowe na dane augmentowane.
Poprawiło to wyniki ale również znacząco podniosło Validation Loss do 0.88 przy 8 epoce - co jest objawem overfittingu.
Recall dla kategorii Drink wzrósł o 6% do 77% co było największym problemem poprzedniego modelu, kosztem popełnienia większej ilości błędów, ale jest to ruch w dobrą stronę z naszego punktu widzenia. Nieznacznie spadł recall dla Food, a Other wzrósł do 62%.
Augmentacja poprawiła ogólne wyniki, ale sprawiła że model jest przeuczony, co może być spowodowane sposobem dodania nowych danych treningowych.

Trzecim modelem jest pełny model wielojęzyczny XLM ROBERTA, trenowany na danych augmentowanych.
Tutaj widać znaczącą poprawę, Dokładność wzrosła do 90%, przy F1 83%.
Recall dla Drink wzrósł o 8% do 85%, Food do 96%, jedyni kategoria Other się nie zmieniła i została na poziomie 62%.
Validation Loss rośnie od 3 epoki ale znacznie wolniej niż poprzednio.

## 4. Wyniki
- **Baseline (DistilBERT):** Dokładność 86%, ale słabo wykrywał napoje (tylko 71% skuteczności).
- **DistilBERT + Augmentacja:** Poprawa wykrywania napojów (77%), ale model zaczął "wkuwać na pamięć" (Overfitting).
- **XLM-RoBERTa + Augmentacja (Zwycięzca):**
    - Najlepsza dokładność: **91%**.
    - Skuteczność dla napojów wzrosła do **85%**.
    - Najbardziej stabilny model.

## 5. Wnioski
- Sama augmentacja danych pomaga, ale trzeba uważać na przeuczenie.
- Wybór odpowiedniej architektury (XLM-R vs DistilBERT) miał kluczowe znaczenie dla jakości.
- Widać, że należy zwiększyć ilość danych w kategorii Other.
- Prawdopodobnie można by uzyskać lepsze wyniki stosując augmentację semantyczną z użyciem LLMów
- Można spróbować dostroić parametry lub dodać sztywne reguły kategoryzujące, na dla Tax -> Other
