"""Innere-Punkte-Verfahren für NLP

Sebastian Hanisch - Operations Research und Machine Learning

Stück 7 der "Nichtlineare Optimierung"-Reihe der "Konzepte"-Reihe:
Gradientenabstieg -> Newton-Verfahren -> Quasi-Newton -> Lagrange/KKT ->
{Straf-/Barriere-Verfahren, SQP -> Innere-Punkte-Verfahren} + Stochastische Gradientenverfahren.
Stück 6 (SQP) erriet je Iteration die aktive Menge neu. Primal-duale Innere-Punkte-Verfahren
(Wächter & Biegler 2006, IPOPT) umgehen das Erraten komplett: Gleichungen, Ungleichungen (über
Schlupfvariablen) und ein schrumpfender Barriere-Parameter werden GEMEINSAM in einem Newton-
System gelöst - das System wächst nur LINEAR mit der Anzahl Ungleichungen K, nicht 2^K wie die
aktive-Menge-Enumeration.

Lauffähig mit: streamlit run app.py
"""
import streamlit as st

import ip_constants as C
import ip_evaluation as ev
import ip_presets as pr
import ip_visualization as viz

st.set_page_config(page_title="Innere-Punkte-Verfahren", layout="wide")


@st.cache_data(show_spinner=False)
def _run(V0, h_max, K):
    out = ev.analyse(ev.Settings(V0=V0, h_max=h_max, K=K))
    s = out["solution"]
    return {"r": s.r, "h": s.h, "lam": s.lam, "mu": s.mu.tolist(),
            "total_inner_iter": s.total_inner_iter, "n_outer": s.n_outer,
            "enum_n_cases": out["enum"]["n_cases"], "enum_total_iter": out["enum"]["total_iter"],
            "enum_found": out["enum"]["found"], "f_val": out["f_val"]}


@st.cache_data(show_spinner=False)
def _reference_convergence():
    return ev.reference_convergence_check()


@st.cache_data(show_spinner=False)
def _combinatorics():
    return ev.combinatorics_scaling_check()


@st.cache_data(show_spinner=False)
def _synthetic_scaling():
    return ev.synthetic_scaling_check()


@st.cache_data(show_spinner=False)
def _complementarity():
    return ev.complementarity_check()


@st.cache_data(show_spinner=False)
def _gradient_check():
    return ev.gradient_check()


st.title("🗻 Innere-Punkte-Verfahren für NLP")
st.markdown(
    "Stück 6 (SQP) erriet je Iteration die aktive Menge neu. **Primal-duale Innere-Punkte-"
    "Verfahren** (Wächter & Biegler 2006, IPOPT) umgehen das Erraten komplett: Gleichungen, "
    "Ungleichungen (über Schlupfvariablen $s_j>0$) und ein schrumpfender Barriere-Parameter "
    "$\\bar\\mu$ werden GEMEINSAM in einem Newton-System gelöst. Der Zylinder-Container bekommt "
    "hier bis zu **drei** Ungleichungen (Höhenlimit, Mindestradius, max. Aspect-Ratio) — genug, "
    "um die Fallzahl-Kombinatorik der aktiven-Menge-Enumeration (Stück 4/6) tatsächlich wachsen "
    "zu lassen."
)
st.caption(
    "Stück 7 der 'Nichtlineare Optimierung'-Reihe. Folgestück: "
    "[Stochastische Gradientenverfahren](https://sebastianhanisch-stochastische-gradientenverfahren-demo.streamlit.app/) (letztes Stück, Brücke zu Deep Learning)."
)

with st.expander("So funktioniert das Innere-Punkte-Verfahren", expanded=True):
    st.markdown(
        "1. Jede Ungleichung $g_j(x)\\le0$ bekommt eine Schlupfvariable: $g_j(x)+s_j=0$, "
        "$s_j>0$.\n"
        "2. Statt exakter Komplementarität ($s_j\\mu_j=0$) wird eine GESTÖRTE Version gefordert: "
        "$s_j\\mu_j=\\bar\\mu$ für ein kleines $\\bar\\mu>0$ — dasselbe Barriere-Prinzip wie "
        "Stück 5, aber jetzt für PRIMALE und DUALE Variablen gemeinsam.\n"
        "3. EIN Newton-Schritt auf Stationarität + Gleichung + Schlupf-Nebenbedingungen + "
        "gestörter Komplementarität — mit einer Fraction-to-Boundary-Regel, die $s,\\mu>0$ "
        "erhält. $\\bar\\mu\\to0$ in einer äußeren Folge."
    )

st.caption("🎯 Schnellstart – ein Klick lädt ein durchgerechnetes Beispiel:")
preset_cols = st.columns(len(C.PRESETS))
for col, (key, preset) in zip(preset_cols, C.PRESETS.items()):
    with col:
        st.button(preset["label"], help=preset["help"], on_click=pr.apply_preset, args=(key,),
                  use_container_width=True)

st.caption("🔗 Die Adresszeile speichert deine Einstellungen als Permalink.")

pr.load_permalink_settings()
pr.init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    st.header("⚙️ Einstellungen")
    V0 = st.slider("Volumen V₀", C.V0_MIN, C.V0_MAX, ss["V0"], step=1.0, key="widget_V0",
                   on_change=pr.store_from_widget, args=("V0",))
    ss["V0"] = V0
    h_max = st.slider("Höhenlimit h_max", C.H_MAX_MIN, C.H_MAX_MAX, ss["h_max"], step=0.05,
                      key="widget_h_max", on_change=pr.store_from_widget, args=("h_max",))
    ss["h_max"] = h_max
    K = st.slider("Anzahl Ungleichungen K", C.K_MIN, C.K_MAX, ss["K"], step=1, key="widget_K",
                  on_change=pr.store_from_widget, args=("K",))
    ss["K"] = K
    st.caption("K=1: nur Höhenlimit. K=2: + Mindestradius. K=3: + max. Aspect-Ratio.")

pr.sync_query_params(dict(V0=V0, h_max=h_max, K=K))

out = _run(V0, h_max, K)

st.markdown("---")
st.subheader("🎯 Dieselbe Lösung, ohne Fallunterscheidung")
r_star_now = (V0 / (2 * 3.141592653589793)) ** (1 / 3)
pad_r = max(2.5, r_star_now * 2.2, C.R_MIN_DEFAULT * 1.3)
pad_h = max(6.0, 2 * r_star_now * 2.2, h_max * 1.3)
col_left, col_right = st.columns([3, 2])
with col_left:
    fig = viz.build_problem_figure(V0, h_max, C.R_MIN_DEFAULT, C.AR_MAX_DEFAULT, K, out["r"],
                                   out["h"], r_range=(0.05, pad_r), h_range=(0.05, pad_h),
                                   title=f"Volumen={V0:.0f}, Höhenlimit={h_max:.2f}, K={K}")
    st.plotly_chart(fig, key=f"problem_{V0}_{h_max}_{K}", use_container_width=True)
with col_right:
    st.metric("Radius r*", f"{out['r']:.4f}")
    st.metric("Höhe h*", f"{out['h']:.4f}")
    st.metric("Innere-Punkte: Newton-Iterationen (gesamt)", out["total_inner_iter"])
    st.metric("Aktive-Menge-Enumeration: Fälle / Iterationen (gesamt)",
             f"{out['enum_n_cases']} / {out['enum_total_iter']}")
    st.caption(f"Gefundene aktive Menge (Enumeration): {out['enum_found']}")

st.markdown("---")
st.subheader("🎯 Die zentrale Messung: wo liegt der Kreuzungspunkt?")
scal = _synthetic_scaling()
col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(viz.build_scaling_figure(scal), key=f"scaling_{V0}_{h_max}",
                    use_container_width=True)
with col_b:
    st.plotly_chart(viz.build_case_count_figure(scal), key=f"cases_{V0}_{h_max}",
                    use_container_width=True)
st.caption(
    "**Ehrlicher Befund:** bei der hier realistischen Portfolio-Größe (K≤3, echte Vehikel-B-"
    "Nebenbedingungen) ist die aktive-Menge-Enumeration NICHT langsamer als das Innere-Punkte-"
    "Verfahren — im Gegenteil, sie braucht insgesamt WENIGER Newton-Iterationen (siehe 📐). Der "
    "aus der Literatur bekannte Vorteil des Innere-Punkte-Verfahrens wird erst ab einer "
    "künstlich erweiterten Nebenbedingungszahl sichtbar (Kreuzungspunkt zwischen K=4 und K=5 in "
    "der Grafik links, gemessen an einer rein synthetischen Erweiterung — nicht mehr Teil von "
    "Vehikel B)."
)

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    "| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |\n"
    "|---|---|---|\n"
    "| Wenige Ungleichungen (K≤3) | Der Vorteil des Innere-Punkte-Verfahrens ist hier NICHT "
    "sichtbar — die Enumeration bleibt günstiger (siehe 🎯 oben) | Erst bei vielen "
    "Nebenbedingungen (industrielle NLPs) zahlt sich das Verfahren aus |\n"
    "| Fraction-to-Boundary hält s,μ>0 | Ohne diese Regel könnte ein voller Newton-Schritt s "
    "oder μ negativ machen — die Barriere wäre nicht mehr definiert | Dieselbe Sicherung wie "
    "Stück 5s Barriere-Verfahren, hier für primale UND duale Variablen |\n"
    "| Alle Ungleichungen linear | Nichtlineare Ungleichungen bräuchten eine eigene Hesse-Matrix "
    "je Term in $H_L$ | Analog zu Stück 6s $H_L=\\nabla^2f+\\lambda\\nabla^2g_1$ |\n"
)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Primal-duales System:** $\nabla f(x)+\lambda\nabla g_1(x)+\sum_j\mu_j\nabla g_j(x)=0$,
$g_1(x)=0$, $g_j(x)+s_j=0$, $s_j\mu_j=\bar\mu$ (für jedes $j=1,\dots,K$), gelöst per Newton
(numerische Jacobi-Matrix wie in Stück 4/6) mit Fraction-to-Boundary-Regel ($\tau=0{,}995$).
"""
    )
    st.markdown("**Referenz-Konvergenz** (K=1, beide Presets):")
    rc = _reference_convergence()
    for row in rc:
        st.caption(f"h_max={row['h_max']} ({row['case']}): Fehler={row['err']:.2e}, "
                  f"{row['total_inner_iter']} Newton-Iterationen gesamt")

    st.markdown("**Kombinatorik an der echten Vehikel-B-Erweiterung** (K=1,2,3):")
    combi = _combinatorics()
    for row in combi:
        st.caption(f"K={row['K']}: {row['n_cases']} Fälle (erwartet {row['expected_n_cases']}), "
                  f"Enumeration {row['enum_total_iter']} Iterationen, Innere-Punkte "
                  f"{row['ip_total_iter']} Iterationen — Innere-Punkte günstiger: "
                  f"{row['ip_cheaper']}, Übereinstimmung: {row['agreement_err']:.2e}")

    st.markdown("**Komplementarität** (K=3): $s_j\\mu_j$ konvergiert gegen $\\bar\\mu$, das "
               "selbst gegen 0 geht:")
    comp = _complementarity()
    c1, c2 = st.columns(2)
    c1.metric("Finale Lücke max(s·μ)", f"{comp['final_gap']:.2e}")
    c2.metric("Finales μ̄", f"{comp['final_mu_bar']:.2e}")
    st.caption(f"Monoton fallend: {comp['monotone_decreasing']}")

    grad_err = _gradient_check()
    g1, g2 = st.columns(2)
    g1.metric("Gradienten-Check f", f"{grad_err['f_max_rel_err']:.1e}")
    g2.metric("Gradienten-Check g₁", f"{grad_err['g1_max_rel_err']:.1e}")

    st.markdown(
        "**Literatur:** Wächter, A. & Biegler, L. T. (2006). *On the implementation of a "
        "primal-dual interior point filter line-search algorithm for large-scale nonlinear "
        "programming.* Mathematical Programming, 106(1), 25–57."
    )
    st.caption(
        "Implementiert in `ip_functions.py` (Zielfunktion, Nebenbedingungen), "
        "`ip_active_set.py` (Vergleichsverfahren: generalisierte Enumeration), `ip_solver.py` "
        "(primal-duales Innere-Punkte-Verfahren), `ip_evaluation.py` (Korrektheits-Kette), "
        "`ip_visualization.py` (Plots)."
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Nichtlineare Optimierung: acht Stücke, zwei Äste](https://sebastianhanisch.net/konzepte-nichtlineare-optimierung.html)."
)
