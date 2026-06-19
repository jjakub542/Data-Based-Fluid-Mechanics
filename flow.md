# Klasyfikacja reżimu przepływu (laminarny / przejściowy / turbulentny)

Skrypt klasyfikuje reżim przepływu cieczy na podstawie pomiarów ciśnienia, wykorzystując liczbę Reynoldsa jako etykietę referencyjną oraz las losowy (`RandomForestClassifier`) jako model predykcyjny. Jest to replikacja podejścia z MATLAB-a (`TreeBagger`) w scikit-learn.

## 1. Dane pomiarowe

Zbiór wejściowy to 16 pomiarów, każdy opisany czterema wielkościami:

| Kolumna | Opis | Jednostka |
|---|---|---|
| `P_przed` | ciśnienie przed przewężeniem | cmH₂O |
| `P_za` | ciśnienie za przewężeniem | cmH₂O |
| `delta_p` | różnica ciśnień (`P_przed - P_za`) | cmH₂O |
| `Q` | zmierzone natężenie przepływu objętościowego | m³/s |

Dane są wpisane wprost w skrypcie (nie wczytywane z pliku zewnętrznego).

## 2. Przeliczenia (transformacje)

### 2.1 Liczba Reynoldsa

Na podstawie `Q` obliczana jest liczba Reynoldsa dla przepływu w przewodzie kołowym:

```
Re = 4 · Q / (π · D · ν)
```

gdzie:
- `D = 0.003 m` — średnica przewodu (stała),
- `ν = 1×10⁻⁶ m²/s` — lepkość kinematyczna wody (stała).

### 2.2 Klasyfikacja reżimu (etykieta `Regime`)

Liczba Reynoldsa jest dzielona na trzy klasy progowe:

| Zakres Re | Etykieta | Kod |
|---|---|---|
| `Re < 2000` | laminarny | `laminar` |
| `2000 ≤ Re < 3000` | przejściowy | `transitional` |
| `Re ≥ 3000` | turbulentny | `turbulent` |

Etykieta `Regime` wyznaczona z `Re` służy jako **prawda referencyjna (ground truth)** — to właśnie ją model uczy się przewidywać na podstawie samych ciśnień, bez znajomości `Q` ani `Re`.

W bieżącym zbiorze danych rozkład klas wygląda następująco:

```
laminar         8
transitional    8
turbulent       0
```

> Żaden z 16 pomiarów nie przekracza `Re = 3000` — klasa turbulentna jest w tym zbiorze pusta. Model trenuje się i ocenia tylko na dwóch klasach (`laminar`, `transitional`), mimo że kod jest w pełni przygotowany na trzy.

## 3. Dane wejściowe do modelu

| Obiekt | Zawartość | Shape |
|---|---|---|
| `X` | `P_przed`, `P_za` (2 cechy) | `(16, 2)` |
| `y` | etykieta `Regime` | `(16,)` |

Podział na zbiór treningowy/testowy: `train_test_split(test_size=0.20, random_state=42)`, **bez stratyfikacji** (zgodnie z oryginalnym podejściem MATLAB, `cvpartition(..., "HoldOut", 0.2)`).

| Zbiór | Liczba próbek |
|---|---|
| treningowy | 12 |
| testowy | 4 |

## 4. Model

`RandomForestClassifier` jako odpowiednik MATLAB-owego `TreeBagger`:

| Parametr sklearn | Wartość | Odpowiednik MATLAB |
|---|---|---|
| `n_estimators` | 300 | `TreeBagger(300, ...)` |
| `bootstrap` | `True` | `SampleWithReplacement` |
| `min_samples_leaf` | 1 | `MinLeafSize` |
| `max_leaf_nodes` | 10 | `MaxNumSplits = 9` (9 podziałów → 10 listków) |
| `max_samples` | 0.8 | `InBagFraction` |
| `oob_score` | `True` | `OOBPrediction` |
| `random_state` | 42 | — (powtarzalność wyników) |

Dodatkowo liczona jest **ważność cech** metodą `permutation_importance` (odpowiednik `OOBPermutedVarDeltaError` z MATLAB-a) — mierzy spadek dokładności modelu po losowej permutacji wartości danej cechy.

## 5. Wyniki

- **Wynik OOB (Out-Of-Bag)** na zbiorze treningowym: `0.8333`
- **Klasy rozpoznane przez model:** `laminar`, `transitional` (klasa `turbulent` nieobecna w danych)
- **Dokładność (accuracy) na zbiorze testowym:** `100.00%` (4/4 trafień — przy tak małym zbiorze testowym wynik należy traktować jako orientacyjny, nie statystycznie istotny)

### Ważność cech (permutation importance)

| Cecha | Ważność |
|---|---|
| `P_przed` | 0.4528 |
| `P_za` | 0.0889 |

`P_przed` ma istotnie większy wpływ na predykcję reżimu przepływu niż `P_za` — usunięcie informacji o `P_przed` (permutacja) powoduje znacznie większy spadek dokładności modelu.

### Rozkład klas: rzeczywisty vs przewidziany (zbiór testowy)

| Klasa | Rzeczywisty | Przewidziany |
|---|---|---|
| Laminarny | 3 | 3 |
| Przejściowy | 1 | 1 |
| Turbulentny | 0 | 0 |

Model idealnie odtworzył rozkład klas na zbiorze testowym (co przy 4 próbkach i braku stratyfikacji jest w dużej mierze przypadkowe).

## 6. Wygenerowane wykresy

| Plik | Zawartość |
|---|---|
| `confusion_matrix.png` | macierz błędów 3×3 (laminarny / przejściowy / turbulentny) |
| `first_tree.png` | wizualizacja pierwszego drzewa z lasu losowego |
| `feature_importance.png` | wykres ważności cech (`P_przed`, `P_za`) |
| `rozklad_klas_rzeczywisty_vs_przewidziany.png` | porównanie liczby próbek w każdej klasie — rzeczywiste vs przewidziane |

## 7. Uwaga dotycząca skalowania na większy zbiór danych

Przy zwiększeniu liczby pomiarów (np. do 75 próbek z szerszym zakresem `Q`) należy oczekiwać wypełnienia klasy `turbulent`, o ile zakres pomiarowy obejmuje `Re ≥ 3000`. Warto też wtedy rozważyć stratyfikację podziału (`stratify=y`), aby każda z trzech klas była reprezentowana proporcjonalnie w zbiorze treningowym i testowym.