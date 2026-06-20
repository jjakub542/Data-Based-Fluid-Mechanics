# Bernoulli — predykcja prędkości przepływu (MLPRegressor)

Program analizuje dane z laboratoryjnego eksperymentu przepływu cieczy (rurka Venturiego / pomiar ciśnień) i trenuje prostą sieć neuronową (MLP), która na podstawie pełnego zestawu zmiennych pomiarowych (ciśnienia, objętość, czas) przewiduje pełny zestaw zmiennych wyliczonych (natężenie przepływu i prędkości).

## 1. Dane eksperymentalne (wejście)

Dane wejściowe to słownik `series` zawierający **87 przebiegów pomiarowych** (po usunięciu przebiegów z brakującymi wartościami — `None`/NaN — w ciśnieniach). Dla każdego przebiegu zapisano:

- `static_pressure` — 6 wartości ciśnienia statycznego [mm H₂O] w 6 punktach pomiarowych rurki,
- `total_pressure` — 6 wartości ciśnienia całkowitego [mm H₂O] w tych samych punktach,
- `Q` — objętość cieczy przepuszczonej przez układ podczas pomiaru [l],
- `t` — czas, w którym ta objętość przepłynęła [s].

Dane są następnie przekształcane do tabeli `df_wide` (DataFrame z wielopoziomowymi kolumnami) o wymiarach **87 wierszy × 14 kolumn** (6 ciśnień statycznych + 6 całkowitych + objętość + czas), indeksowanej nazwą przebiegu.

## 2. Transformacje danych (obliczenia)

Z surowych pomiarów wyliczane są wielkości fizyczne potrzebne do dalszej analizy (funkcja w komórce „Model" danych, sekcja obliczeniowa):

- **Natężenie przepływu** — z objętości i czasu:
  - `q_l_s = V/t`, następnie `q_l_h = q_l_s * 3600` [l/h] oraz `q_m3_s = q_l_s / 1000` [m³/s].
- **Prędkość z ciśnienia dynamicznego** (dla każdego z 6 punktów pomiarowych), na podstawie wzoru Bernoulliego:
  - Δh = (ciśnienie całkowite − ciśnienie statyczne) / 1000 [m],
  - v = √(2·g·Δh), gdzie g = 9,80665 m/s² (wynik obcinany do ≥ 0).
- **Prędkość z równania ciągłości** (dla każdego z 6 punktów, mających różny przekrój rurki):
  - v = Q[m³/s] / A[m²], gdzie powierzchnie przekroju `area_m2` to stałe, znane z geometrii rurki wartości (od 84,6·10⁻⁶ do 338,6·10⁻⁶ m²).

Wynikiem jest tabela `df_flow` (87 wierszy × 14 kolumn) zawierająca: natężenie przepływu (l/h i m³/s) oraz po 6 wartości prędkości z obu metod dla każdego przebiegu. Dodatkowo program generuje 4 wykresy porównujące ciśnienie statyczne, ciśnienie dynamiczne oraz obie prędkości w funkcji numeru punktu pomiarowego, dla wszystkich 87 przebiegów.

## 3. Trenowanie modelu

**Algorytm:** `MLPRegressor` ze scikit-learn — perceptron wielowarstwowy (sieć neuronowa) z:
- 1 warstwą skrytą, funkcją aktywacji **tanh** (odwzorowanie klasycznego `fitnet` z MATLABa),
- solverem **lbfgs** (optymalizacja quasi-Newtonowska, odpowiednia dla małych zbiorów danych),
- `tol=1e-6`.

Liczba neuronów w warstwie skrytej (`hidden_layer_sizes=(n,)`) oraz `max_iter` zostały dobrane empirycznie: przetestowano kilka różnych wartości `n` i `max_iter`, porównując MSE/R² na zbiorze walidacyjnym i testowym, aż znaleziono ustawienia dające najlepszy kompromis między dopasowaniem a przeuczeniem. Finalnie przyjęto `hidden_layer_sizes=(10,)` i `max_iter=1000`.

Dodatkowo, wyłącznie w celu narysowania krzywej uczenia (loss curve), trenowany jest pomocniczy model z solverem **adam** (gradient stochastyczny), bo `lbfgs` nie udostępnia historii strat epoka po epoce.

**Dane wejściowe do modelu:**
- **X** = `df_wide.values` — tablica o wymiarach **(87, 14)**: wszystkie surowe zmienne pomiarowe (6 ciśnień statycznych, 6 ciśnień całkowitych, objętość, czas) dla 87 przebiegów,
- **y** = `df_flow.values` — tablica o wymiarach **(87, 14)**: wszystkie zmienne wyliczone (natężenie przepływu w l/h i m³/s, 6 prędkości z ciśnienia dynamicznego, 6 prędkości z równania ciągłości) dla 87 przebiegów.

Model uczy się więc mapowania „pełny zestaw pomiarów surowych → pełny zestaw wielkości wyliczonych" (multi-output regression z 14 wejściami i 14 wyjściami), a nie tylko zależności między dwoma metodami wyznaczania prędkości jak w poprzedniej wersji.

**Przygotowanie danych:**
- podział na trzy zbiory: treningowy/walidacyjny/testowy w proporcji **70%/15%/15%** (analogicznie do domyślnego podziału w MATLAB `fitnet`), realizowany dwoma wywołaniami `train_test_split` (`random_state=42`) — co dało **60 / 13 / 14** wierszy,
- standaryzacja (`StandardScaler`) cech X i celów y — osobne skalery dopasowane tylko na zbiorze treningowym, następnie zastosowane do walidacyjnego, testowego i całego zbioru,
- po predykcji wyniki są odwracane (`inverse_transform`) do oryginalnej skali.

## 4. Wyniki

Wydajność modelu (MSE oraz R²) jest raportowana **osobno dla każdego z trzech zbiorów**, a nie tylko łącznie:

| Zbiór | Liczba próbek | MSE | R² |
|---|---|---|---|
| Treningowy | 60 | 0.003151 | 1.0000 |
| Walidacyjny | 13 | 344.809686 | 0.8218 |
| Testowy | 14 | 352.269964 | 0.6634 |
| Cały zbiór | 87 | 108.212580 | 0.8918 |

MSE treningowe bliskie zeru przy R² = 1.0000 wskazuje, że model praktycznie zapamiętał zbiór treningowy. R² na zbiorze walidacyjnym (0.82) i testowym (0.66) jest niższe, co jest typowym sygnałem **przeuczenia (overfitting)** — mimo to model generalizuje wystarczająco dobrze, by R² na całym zbiorze wyniosło ok. 0.89. Różnica między MSE walidacyjnym/testowym a treningowym jest oczekiwana przy tej wielkości zbioru (87 próbek podzielonych na trzy podzbiory) i złożoności sieci.

Ponieważ X i y zawierają wielkości o bardzo różnych jednostkach i skalach (mm H₂O, litry, sekundy, l/h, m³/s, m/s), `StandardScaler` normalizuje każdą kolumnę niezależnie, ale metryki zagregowane (histogram błędów, wspólny wykres regresji, MSE z tabeli powyżej) łączą błędy z różnych wielkości fizycznych w jedną zbiorczą statystykę — interpretacja pojedynczej liczby MSE/R² jest więc orientacyjna, a wysoka wartość MSE (np. 108 czy 352) wynika głównie z wielkości w innych jednostkach (np. l/h), nie z błędu rzędu setek m/s.

Program wizualizuje wyniki trzema wykresami (analogicznie do MATLAB `fitnet`), liczonymi na pełnym zbiorze (`y` vs `y_pred_all`):
1. krzywa uczenia (MSE w skali logarytmicznej na pomocniczym modelu Adam),
2. histogram błędów predykcji,
3. wykres regresji (predykcja vs wartość rzeczywista) z dopasowaną linią i linią idealną Y=T.

## Wymagane biblioteki

`numpy`, `pandas`, `matplotlib`, `scikit-learn` (moduły: `neural_network`, `model_selection`, `preprocessing`, `metrics`).