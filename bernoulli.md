# Bernoulli — predykcja prędkości przepływu (MLPRegressor)

Program analizuje dane z laboratoryjnego eksperymentu przepływu cieczy (rurka Venturiego / pomiar ciśnień) i trenuje prostą sieć neuronową (MLP), która na podstawie prędkości wyznaczonej z ciśnienia dynamicznego przewiduje prędkość wyznaczoną z równania ciągłości.

## 1. Dane eksperymentalne (wejście)

Dane wejściowe to słownik `series` zawierający **15 przebiegów pomiarowych** (`run_1`–`run_15`). Dla każdego przebiegu zapisano:

- `static_pressure` — 6 wartości ciśnienia statycznego [mm H₂O] w 6 punktach pomiarowych rurki,
- `total_pressure` — 6 wartości ciśnienia całkowitego [mm H₂O] w tych samych punktach,
- `Q` — objętość cieczy przepuszczonej przez układ podczas pomiaru [l] (zawsze 10 l),
- `t` — czas, w którym ta objętość przepłynęła [s] (różny dla każdego przebiegu, ok. 42–88 s).

Dane są następnie przekształcane do tabeli `df_wide` (DataFrame z wielopoziomowymi kolumnami) o wymiarach **15 wierszy × 14 kolumn** (6 ciśnień statycznych + 6 całkowitych + objętość + czas), indeksowanej nazwą przebiegu.

## 2. Transformacje danych (obliczenia)

Z surowych pomiarów wyliczane są wielkości fizyczne potrzebne do dalszej analizy (funkcja w komórce „Model” danych, sekcja obliczeniowa):

- **Natężenie przepływu** — z objętości i czasu:
  - `q_l_s = V/t`, następnie `q_l_h = q_l_s * 3600` [l/h] oraz `q_m3_s = q_l_s / 1000` [m³/s].
- **Prędkość z ciśnienia dynamicznego** (dla każdego z 6 punktów pomiarowych), na podstawie wzoru Bernoulliego:
  - Δh = (ciśnienie całkowite − ciśnienie statyczne) / 1000 [m],
  - v = √(2·g·Δh), gdzie g = 9,80665 m/s² (wynik obcinany do ≥ 0).
- **Prędkość z równania ciągłości** (dla każdego z 6 punktów, mających różny przekrój rurki):
  - v = Q[m³/s] / A[m²], gdzie powierzchnie przekroju `area_m2` to stałe, znane z geometrii rurki wartości (od 84,6·10⁻⁶ do 338,6·10⁻⁶ m²).

Wynikiem jest tabela `df_flow` (15 wierszy) zawierająca: natężenie przepływu (l/h i m³/s) oraz po 6 wartości prędkości z obu metod dla każdego przebiegu. Dodatkowo program generuje 4 wykresy porównujące ciśnienie statyczne, ciśnienie dynamiczne oraz obie prędkości w funkcji numeru punktu pomiarowego, dla wszystkich 15 przebiegów.

## 3. Trenowanie modelu

**Algorytm:** `MLPRegressor` ze scikit-learn — perceptron wielowarstwowy (sieć neuronowa) z:
- 1 warstwą skrytą o **10 neuronach**,
- funkcją aktywacji **tanh** (odwzorowanie klasycznego `fitnet` z MATLABa),
- solverem **lbfgs** (optymalizacja quasi-Newtonowska, odpowiednia dla małych zbiorów danych — tu tylko 15 próbek),
- `max_iter=1000`, `tol=1e-6`.

Dodatkowo, wyłącznie w celu narysowania krzywej uczenia (loss curve), trenowany jest pomocniczy model z solverem **adam** (gradient stochastyczny), bo `lbfgs` nie udostępnia historii strat epoka po epoce.

**Dane wejściowe do modelu:**
- **X** = `df_flow["Prędkość z ciśnienia dynamicznego, m/s"]` — tablica o wymiarach **(15, 6)** (15 przebiegów × 6 punktów pomiarowych),
- **y** = `df_flow["Prędkość z równania ciągłości, m/s"]` — analogicznie **(15, 6)**.

Model uczy się więc mapowania „prędkość z ciśnienia dynamicznego → prędkość z równania ciągłości” jednocześnie dla wszystkich 6 punktów pomiarowych (multi-output regression).

**Przygotowanie danych:**
- podział na zbiór treningowy/walidacyjny: `train_test_split` (80%/20%, `random_state=42`),
- standaryzacja (`StandardScaler`) cech X i celów y — osobne skalery dopasowane tylko na zbiorze treningowym, następnie zastosowane do walidacyjnego i całego zbioru,
- po predykcji wyniki są odwracane (`inverse_transform`) do oryginalnej skali [m/s].

## 4. Wyniki

| Metryka | Wartość |
|---|---|
| MSE (trening) | ≈ 0.000000 (model dopasował się idealnie do 15 próbek) |
| MSE (walidacja) | ≈ 0.0801 |
| MSE (cały zbiór) | ≈ 0.0160 |
| R² (cały zbiór) | ≈ 0.32 |

Model praktycznie zapamiętał (wyuczył się na pamięć) bardzo mały zbiór treningowy — błąd treningowy spada do zera, podczas gdy błąd walidacyjny jest znacznie wyższy. R² na poziomie ~0,32 oznacza umiarkowanie słabe dopasowanie predykcji do rzeczywistych prędkości z równania ciągłości na pełnym zbiorze. Jest to typowy efekt **przeuczenia (overfitting)** wynikający z bardzo małej liczby próbek (tylko 15 przebiegów) względem złożoności sieci.

Program wizualizuje wyniki trzema wykresami (analogicznie do MATLAB `fitnet`):
1. krzywa uczenia (MSE w skali logarytmicznej na pomocniczym modelu Adam),
2. histogram błędów predykcji,
3. wykres regresji (predykcja vs wartość rzeczywista) z dopasowaną linią i linią idealną Y=T.

## Wymagane biblioteki

`numpy`, `pandas`, `matplotlib`, `scikit-learn` (moduły: `neural_network`, `model_selection`, `preprocessing`, `metrics`).