# Klasyfikacja reżimu przepływu — Las Losowy

Projekt realizuje klasyfikację rodzaju przepływu cieczy (laminarny / przejściowy / turbulentny) przy użyciu modelu lasu losowego.

---

## Dane pomiarowe

Dane pochodzą z dwóch źródeł:

**Zestaw 1 — (`lam_turb_data`):
zawiera pomiary dla dwóch typów przepływu (laminarny i turbulentny). Dla przepływu laminarnego mierzone były ciśnienia statyczne `p_1` i `p_2` (mmH₂O) po obu stronach odcinka pomiarowego oraz czas napełnienia zbiornika (s) przy objętości 0,5 L. Dla przepływu turbulentnego mierzono różnicę ciśnień `p_diff` (bar) i czas napełnienia zbiornika (s) przy objętości 2 L. Łącznie zebrano 141 próbek (72 laminarne + 69 turbulentnych).

**Zestaw 2 — własne dane pomiarowe** (`data`):
16 próbek z bezpośrednimi odczytami ciśnień `p_1`, `p_2` (mmH₂O), różnicy ciśnień `p_diff` (mmH₂O) oraz natężenia przepływu `Q` (m³/s).

---

## Transformacje i obliczenia

Z danych surowych wyznaczano następujące wielkości:

- **Ciśnienia w Pa**: przeliczenie z mmH₂O na Pascale wg wzoru `P [Pa] = p [mmH₂O] × 9,81`, a z barów na Pascale wg wzoru `P [Pa] = p [bar] × 10⁵`.
- **Różnica ciśnień `P_diff_Pa`**: dla przepływu laminarnego jako `P_1_Pa − P_2_Pa`, dla turbulentnego z odczytu różnicowego.
- **Objętościowe natężenie przepływu `Q_m³/s`**: jako iloraz objętości przez czas napełnienia zbiornika.
- **Liczba Reynoldsa `Re`**: wyznaczana ze wzoru `Re = 4Q / (π · D · ν)`, gdzie ν = 10⁻⁶ m²/s (lepkość kinematyczna wody).

Na podstawie liczby Reynoldsa przypisywano etykietę przepływu:

| Kryterium | Przepływ |
|---|---|
| Re < 2000 | laminarny |
| 2000 ≤ Re < 3000 | przejściowy |
| Re ≥ 3000 | turbulentny |

---

## Dane wejściowe do modelu

Po połączeniu obu zbiorów danych model otrzymuje dwie cechy:

| Cecha | Opis |
|---|---|
| `P_diff_Pa` | Różnica ciśnień [Pa] |
| `Q_m3s` | Objętościowe natężenie przepływu [m³/s] |

Zmienna docelowa `y` to etykieta przepływu: `laminar`, `transitional` lub `turbulent`.

Dane podzielono na zbiór treningowy (80%) i testowy (20%) przy ustalonym ziarnie losowości (`random_state=42`).

---

## Model — Las Losowy (Random Forest)

Las losowy to metoda ensemble łącząca wiele niezależnych drzew decyzyjnych. Każde drzewo trenowane jest na losowej podpróbce danych i losowym podzbiorze cech, a predykcja końcowa wyznaczana jest przez głosowanie większościowe. Dzięki temu model jest odporny na przeuczenie i dobrze radzi sobie z nieliniowymi granicami decyzyjnymi.

Zastosowana implementacja (`RandomForestClassifier` z biblioteki scikit-learn) jest odpowiednikiem `TreeBagger` z MATLAB-a.

### Parametry modelu (dobrane optymalnie)

| Parametr | Wartość | Opis |
|---|---|---|
| `n_estimators` | 300 | Liczba drzew w lesie |
| `bootstrap` | True | Losowanie ze zwracaniem (próbkowanie bootstrapowe) |
| `min_samples_leaf` | 1 | Minimalna liczba próbek w liściu |
| `max_leaf_nodes` | 10 | Maksymalna liczba liści w drzewie |
| `max_samples` | 0.8 | Frakcja danych użyta do treningu każdego drzewa |
| `oob_score` | True | Włączone szacowanie błędu Out-Of-Bag |

Powyższe parametry zostały wybrane jako optymalne dla tego zbioru danych, zapewniając równowagę między złożonością modelu a jego generalizacją.

---

## Wyniki

Model osiągnął wysoką skuteczność klasyfikacji na zbiorze testowym. Do oceny użyto:

- **Dokładność (Accuracy)** — wyznaczona na zbiorze testowym.
- **Wynik OOB (Out-Of-Bag)** — estymacja błędu na próbkach nieużytych w treningu poszczególnych drzew, bez konieczności osobnej walidacji.
- **Macierz błędów (Confusion Matrix)** — wizualizacja poprawnych i błędnych klasyfikacji dla każdej klasy.
- **Ważność cech (Feature Importance)** — wyznaczona metodą permutacyjną (odpowiednik `OOBPermutedVarDeltaError` z MATLAB-a): każda cecha była kolejno losowo permutowana, a miarą ważności był wynikowy spadek dokładności. Cecha `P_diff_Pa` okazała się dominującym predyktorem rodzaju przepływu.

Wizualizacje obejmują macierz błędów, strukturę przykładowego drzewa z lasu, wykres ważności cech oraz porównanie rozkładów klas rzeczywistych i przewidzianych.

## Wymagane biblioteki

`numpy`, `pandas`, `matplotlib`, `scikit-learn` (moduły: `neural_network`, `model_selection`, `preprocessing`, `metrics`).