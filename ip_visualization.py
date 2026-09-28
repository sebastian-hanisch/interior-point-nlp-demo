"""Reine Plotly-Figure-Builder, keine Streamlit-Aufrufe. Achsen fest uebergeben (siehe
feedback_plotly_fixedrange_convention/feedback_plotly_scaleanchor_explicit_range). Log-Achsen
(Iterationen vs. K) explizit als type='log' gesetzt. Keine literalen "|" in Markdown-
Tabellenzellen (feedback_markdown_table_literal_pipe_breaks_columns)."""
import numpy as np
import plotly.graph_objects as go

import ip_functions as fn

COLOR_CONSTRAINT = "#d62728"
COLOR_LIMIT = "#9467bd"
COLOR_RMIN = "#8c564b"
COLOR_OPT = "#2ca02c"
COLOR_IP = "#1f77b4"
COLOR_ENUM = "#d62728"


def build_problem_figure(V0, h_max, r_min, ar_max, K, solution_r, solution_h, r_range=(0.1, 2.5),
                         h_range=(0.1, 6.0), title=""):
    rs = np.linspace(r_range[0], r_range[1], 150)
    hs = np.linspace(h_range[0], h_range[1], 150)
    Z = np.zeros((len(hs), len(rs)))
    for i, hv in enumerate(hs):
        for j, rv in enumerate(rs):
            Z[i, j] = fn.surface_area(np.array([rv, hv]))
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=rs, y=hs, z=Z, showscale=False, colorscale="Blues", contours=dict(coloring="fill"),
        opacity=0.5,
    ))
    h_curve = V0 / (np.pi * rs ** 2)
    mask = (h_curve >= h_range[0]) & (h_curve <= h_range[1])
    fig.add_trace(go.Scatter(x=rs[mask], y=h_curve[mask], mode="lines", name="Volumen V=V₀",
                             line=dict(color=COLOR_CONSTRAINT, width=2)))
    fig.add_trace(go.Scatter(x=list(r_range), y=[h_max, h_max], mode="lines",
                             name=f"Höhenlimit h≤{h_max:.2f}",
                             line=dict(color=COLOR_LIMIT, width=2, dash="dash")))
    if K >= 2:
        fig.add_trace(go.Scatter(x=[r_min, r_min], y=list(h_range), mode="lines",
                                 name=f"Mindestradius r≥{r_min:.2f}",
                                 line=dict(color=COLOR_RMIN, width=2, dash="dot")))
    if K >= 3:
        ar_h = [ar_max * r_range[0], ar_max * r_range[1]]
        fig.add_trace(go.Scatter(x=list(r_range), y=ar_h, mode="lines",
                                 name=f"Aspect-Ratio h≤{ar_max:.2f}·r",
                                 line=dict(color="#e377c2", width=2, dash="dashdot")))
    fig.add_trace(go.Scatter(x=[solution_r], y=[solution_h], mode="markers", name="Lösung",
                             marker=dict(color=COLOR_OPT, size=14, symbol="star")))
    fig.update_layout(
        title=title, xaxis=dict(range=list(r_range), fixedrange=True, title="Radius r"),
        yaxis=dict(range=list(h_range), fixedrange=True, title="Höhe h"),
        showlegend=True, height=440, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_scaling_figure(rows, title="Fallzahl/Iterationen vs. Anzahl Ungleichungen K"):
    K_vals = [r["K"] for r in rows]
    enum_vals = [r["enum_total_iter"] for r in rows]
    ip_vals = [r["ip_total_iter"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=K_vals, y=enum_vals, mode="lines+markers",
                             name="Aktive-Menge-Enumeration (Summe Newton-Iterationen)",
                             line=dict(color=COLOR_ENUM, width=2)))
    fig.add_trace(go.Scatter(x=K_vals, y=ip_vals, mode="lines+markers",
                             name="Innere-Punkte-Verfahren (Summe Newton-Iterationen)",
                             line=dict(color=COLOR_IP, width=2)))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Anzahl Ungleichungen K", fixedrange=True, dtick=1),
        yaxis=dict(title="Gesamt-Newton-Iterationen", type="log", fixedrange=True),
        showlegend=True, height=380, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_case_count_figure(rows, title="Fallzahl der aktiven-Menge-Enumeration: 2^K"):
    K_vals = [r["K"] for r in rows]
    n_cases = [r["n_cases"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=K_vals, y=n_cases, marker_color=COLOR_ENUM, text=n_cases,
                         textposition="outside"))
    fig.update_layout(
        title=title, xaxis=dict(title="Anzahl Ungleichungen K", fixedrange=True, dtick=1),
        yaxis=dict(title="Anzahl Teilmengen (Fälle)", type="log", fixedrange=True),
        showlegend=False, height=340, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
