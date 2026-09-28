# Innere-Punkte-Verfahren für NLP – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-interior-point-nlp-demo.streamlit.app/)**

Stück 7 der **Nichtlineare-Optimierung-Reihe** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Stück 6 (SQP) erriet je Iteration die aktive Menge neu. **Primal-duale Innere-Punkte-Verfahren**
(Wächter & Biegler 2006, IPOPT) umgehen das Erraten komplett: Gleichungen, Ungleichungen (über
Schlupfvariablen) und ein schrumpfender Barriere-Parameter werden GEMEINSAM in einem
Newton-System gelöst.

**Einordnung in die Reihe:**

```
Gradientenabstieg (WURZEL)                       [gebaut]
 └─ Newton-Verfahren                             [gebaut]
      └─ Quasi-Newton (BFGS/L-BFGS)               [gebaut]
           └─ Lagrange-Multiplikatoren/KKT        [gebaut]
                ├─ Straf-/Barriere-Verfahren      [gebaut]
                └─ SQP                            [gebaut]
                     └─ Innere-Punkte-Verfahren   [DIESES STÜCK]
 └─ Stochastische Gradientenverfahren             [nicht gebaut, letztes Stück]
```

**Vehikel B, erweitert um mehrere Ungleichungen (eigene Kopie ohne Import):** derselbe
zylindrische Transportbehälter, jetzt mit bis zu DREI Ungleichungen (Höhenlimit, Mindestradius,
max. Aspect-Ratio) statt nur einer — damit die Nebenbedingungszahl $K$ tatsächlich variiert
werden kann.

**Ergebnis in Kürze:** das Innere-Punkte-Verfahren konvergiert für $K=1$ exakt zur selben
Referenzlösung wie Stück 4–6. Die generalisierte aktive-Menge-Enumeration braucht bei $K$
Ungleichungen tatsächlich $2^K$ Fälle (2 → 4 → 8 für $K=1\to3$, exakt gemessen). **Der zentrale,
ehrliche Befund:** bei dieser realistischen Portfolio-Größe ($K\le3$) ist die Enumeration NICHT
teurer als das Innere-Punkte-Verfahren — im Gegenteil, sie braucht insgesamt WENIGER
Newton-Iterationen (10–26 gegen 39–45). Eine explizit synthetische Erweiterung (bis $K=8$) zeigt:
der aus der Literatur bekannte Vorteil existiert wirklich, aber der Kreuzungspunkt liegt erst
zwischen $K=4$ und $K=5$ — eine ehrliche Plan-Korrektur der ursprünglichen Hypothese.

## Warum dieses Problem

Stück 6 zeigte SQP: je Iteration wird die aktive Menge neu geraten und per kleiner QP geprüft.
Bei EINER Ungleichung ist das trivial (2 Fälle). Innere-Punkte-Verfahren beantworten die Frage,
die sich bei VIELEN Ungleichungen stellt: was, wenn $2^K$ Fälle nicht mehr praktikabel sind? Die
Antwort: alle Ungleichungen gemeinsam über Schlupfvariablen und eine gestörte Komplementaritäts-
Bedingung behandeln, statt Fälle zu unterscheiden.

## Vorab-Hypothesen (vor der Messung notiert, hier geprüft)

| Hypothese | Ergebnis |
|---|---|
| Für $K=1$ konvergiert Innere-Punkte zur selben Referenzlösung wie Stück 4–6 | ✅ Fehler $<1{,}1\cdot10^{-9}$ |
| Die aktive-Menge-Enumeration braucht $2^K$ Fälle | ✅ exakt 2, 4, 8 für $K=1,2,3$ |
| Innere-Punkte-Kosten wachsen milder als $2^K$ mit $K$ | ✅ 39→44→45 Newton-Iterationen (K=1→3), praktisch konstant |
| Primal-duale Komplementarität $s_j\mu_j\to0$ | ✅ monoton fallend, Endlücke $4{,}1\cdot10^{-9}$ |
| Gradienten-Check gegen finite Differenzen unter $10^{-6}$ | ✅ $2{,}6\cdot10^{-10}$ / $5{,}4\cdot10^{-11}$ |
| ⚠️ **Ehrliche Prüfung (Hook 7 der Scoping-Notiz):** ist der IP-Vorteil bei $K\le3$ schon sichtbar? | ⚠️ NEIN — bei $K\le3$ ist die Enumeration sogar GÜNSTIGER (10–26 gegen 39–45 Iterationen); der Vorteil zeigt sich erst ab $K\approx5$ (synthetisch getestet bis $K=8$) |

## Befunde (gemessen, keine Behauptungen)

**Referenz-Konvergenz** ($K=1$, beide Presets): $h_{\max}=5{,}0$ (inaktiv) Fehler
$6{,}7\cdot10^{-10}$, 39 Newton-Iterationen gesamt; $h_{\max}=1{,}5$ (aktiv) Fehler
$1{,}1\cdot10^{-9}$, 43 Iterationen.

**Kombinatorik an der echten Vehikel-B-Erweiterung** ($V_0=10$, $h_{\max}=5{,}0$, $r_{\min}=1{,}3$,
$AR_{\max}=1{,}8$):

| $K$ | Fälle (Enumeration) | Enum.-Iterationen gesamt | IP-Iterationen gesamt | IP günstiger? |
|---|---|---|---|---|
| 1 | 2 | 10 | 39 | Nein |
| 2 | 4 | 15 | 44 | Nein |
| 3 | 8 | 26 | 45 | Nein |

Bei $K=2$/$3$ wird der Mindestradius aktiv ($r^*=1{,}3$, $h^*=1{,}8835$) statt des Höhenlimits —
beide Methoden finden dieselbe Lösung (Übereinstimmung $<3\cdot10^{-9}$).

**Synthetische Skalierungs-Messreihe** (explizit ÜBER Vehikel B hinaus — $K$ redundante
Höhenlimit-artige Kopien, nur um den Trend selbst sichtbar zu machen, siehe unten): Fallzahl
wächst exakt $2^K$ (2, 4, 8, 16, 32, 64, 128, 256 für $K=1..8$). Enumerations-Iterationen: 10, 17,
25, 37, 57, 91, 152, 262. Innere-Punkte-Iterationen bleiben nahezu konstant: 39, 43, 43, 43, 43,
43, 45, 45. **Kreuzungspunkt zwischen $K=4$ (37 < 43) und $K=5$ (57 > 43).** Bei $K=8$ ist die
Enumeration bereits $262/45\approx5{,}8\times$ teurer.

**Komplementarität** ($K=3$): $\max(s_j\mu_j)$ fällt monoton von 1,0 (bei $\bar\mu=1$) auf
$4{,}1\cdot10^{-9}$ (bei $\bar\mu=1{,}6\cdot10^{-10}$) über 15 äußere Stufen.

## Modell und Verfahren

- `ip_functions.py` – Zielfunktion, Volumen-Gleichung, PLUS drei lineare Ungleichungen
  (Höhenlimit, Mindestradius, max. Aspect-Ratio), als Liste für variables $K$.
- `ip_reference.py` – Referenzlösung für $K=1$ (Kopie der Stück-4-Formel).
- `ip_active_set.py` – generalisierte $2^K$-Enumeration (Vergleichsverfahren): pro Teilmenge ein
  Newton-Löse (numerische Jacobi-Matrix wie Stück 4/6), Fallzahl und Gesamt-Iterationen gezählt.
- `ip_solver.py` – primal-duales Newton-System mit Schlupfvariablen, Fraction-to-Boundary-Regel,
  äußere $\bar\mu$-Folge.
- `ip_evaluation.py` – Referenz-Konvergenz, Kombinatorik-Skalierung (echt UND synthetisch),
  Komplementaritäts-Check, Gradienten-Check.
- `ip_visualization.py` – Plotly: Kontur mit allen aktiven Nebenbedingungen, Skalierungs-Kurven
  (log-y) und Fallzahl-Balken.

## Was die App zeigt

Volumen, Höhenlimit und Anzahl Ungleichungen $K\in\{1,2,3\}$ in der Sidebar; Kontur mit Volumen-
Kurve und allen für das gewählte $K$ aktiven Nebenbedingungslinien; Lösung, Iterationszahlen
beider Verfahren als Metriken; die Skalierungs-Kurven (Enumeration vs. Innere-Punkte, bis $K=8$
synthetisch) als zentrale Messung; ein "📐"-Abschnitt mit der vollständigen Korrektheits-Kette.

## Was nicht funktioniert hat / Grenzen

**Eine echte Plan-Korrektur:** die ursprüngliche Hypothese "Innere-Punkte skaliert besser als
Enumeration" ist RICHTIG, aber nicht bei der hier realistischen Portfolio-Größe ($K\le3$)
sichtbar — dort ist die Enumeration sogar günstiger. Der Vorteil existiert nachweislich (Kreuzungs-
punkt bei $K\approx5$, synthetisch verifiziert bis $K=8$), aber er ist kein Argument FÜR
Innere-Punkte-Verfahren bei kleinen Portfolio-großen NLPs wie den übrigen Demos dieser Linie.

**Grenzen:** alle Ungleichungen linear (bewusst — hält die Kombinatorik isoliert von
zusätzlicher Nichtlinearität). Kein Filter-Line-Search-Mechanismus wie im echten IPOPT (bewusst
weggelassen — Fraction-to-Boundary allein reicht bei diesem kleinen, gutartigen Problem).

## Tests

36 Tests, `python -m pytest tests/ -v` (Laufzeit lokal ~2,9 Sekunden):
- `test_functions.py` – Zielfunktion/Nebenbedingungen, Gradient gegen finite Differenzen.
- `test_reference.py` – Referenzlösung.
- `test_active_set.py` – Fallzahl $=2^K$, findet stets eine zulässige Lösung.
- `test_solver.py` – Referenz-Konvergenz, Schlupf/Multiplikatoren bleiben positiv,
  Komplementaritäts-Lücke schrumpft.
- `test_evaluation.py`, `test_claims.py` – jede Zahl oben nachgerechnet, inkl. Kreuzungspunkt.
- `test_presets.py`, `test_app.py` – Presets, Regler-Extremwerte, Footer.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `ip_constants.py` | Regler-Grenzen, Presets |
| `ip_functions.py` | Zielfunktion, Nebenbedingungen (bis zu 3 Ungleichungen) |
| `ip_reference.py` | Referenzlösung (Kopie aus Stück 4) |
| `ip_active_set.py` | Generalisierte $2^K$-Enumeration (Vergleichsverfahren) |
| `ip_solver.py` | Primal-duales Innere-Punkte-Verfahren |
| `ip_evaluation.py` | Korrektheits-Kette |
| `ip_visualization.py` | Plotly-Plots |
| `ip_presets.py` | Permalink-Sync, Presets |
| `tests/` | pytest-Suite |

## Bewusst nicht umgesetzt

Kein Filter-Line-Search oder Merit-Function-Mechanismus wie im echten IPOPT (bewusst — die
Fraction-to-Boundary-Regel allein reicht für dieses kleine, gutartige Problem). Keine
nichtlinearen Ungleichungen (bewusst — hält die $2^K$-Kombinatorik isoliert). Kein Solver-
Benchmark (IPOPT gegen KNITRO) — diese Linie erklärt die Verfahren selbst.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-dev.txt
streamlit run app.py
```

## Literatur

- Wächter, A. & Biegler, L. T. (2006). *On the implementation of a primal-dual interior point
  filter line-search algorithm for large-scale nonlinear programming.* Mathematical Programming,
  106(1), 25–57.
