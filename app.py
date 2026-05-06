import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model import simulate_pnl

st.set_page_config(layout="wide", page_title="CE stocastico")
st.title("Conto Economico stocastico – Monte Carlo")

BRENT_REF = 80.0
FX_REF = 1.08

KPIS = {
    "EBITDA": "ebitda",
    "EBIT": "ebit",
    "EBT": "ebt",
    "Utile Netto": "net",
}

if "scenarios" not in st.session_state:
    st.session_state.scenarios = {}

with st.sidebar:
    st.header("Simulazione")
    run = st.button("▶ Run simulazione")
    seed = st.number_input("Seed", value=42, step=1)

    years = st.slider("Orizzonte (anni)", 1, 5, 3)
    n_sims = st.select_slider("Simulazioni", [1000, 5000, 10000, 20000], 10000)

    st.subheader("Driver (livelli medi)")
    infl = st.slider("Inflazione annua", 0.0, 0.06, 0.025)
    brent = st.slider("Brent (USD/bbl)", 50.0, 120.0, 80.0)
    fx = st.slider("EUR/USD", 0.95, 1.20, 1.08)

    st.subheader("Volatilità")
    vol_brent = st.slider("Vol Brent", 0.05, 0.6, 0.25)
    vol_fx = st.slider("Vol FX", 0.03, 0.3, 0.10)
    vol_infl = st.slider("Vol Infl", 0.002, 0.05, 0.01)

    st.subheader("Correlazioni driver")
    rho_bf = st.slider("Corr Brent–FX", -0.9, 0.9, 0.1)
    rho_bi = st.slider("Corr Brent–Infl", -0.9, 0.9, 0.2)
    rho_fi = st.slider("Corr FX–Infl", -0.9, 0.9, 0.1)

    corr = np.array([
        [1, rho_bf, rho_bi],
        [rho_bf, 1, rho_fi],
        [rho_bi, rho_fi, 1]
    ])

    st.subheader("Base CE (milioni)")
    rev0 = st.number_input("Ricavi", value=1000.0)
    cogs0 = st.number_input("Costi", value=600.0)
    opex0 = st.number_input("Spese", value=250.0)
    da0 = st.number_input("Ammortamenti", value=80.0)
    extra_mean = st.number_input("Straordinari media", value=0.0)
    extra_vol = st.number_input("Straordinari vol", value=20.0)

    st.subheader("Elasticità")
    rev_infl = st.slider("Ricavi ~ Infl", 0.0, 1.5, 0.7)
    rev_fx = st.slider("Ricavi ~ FX", 0.0, 1.0, 0.2)
    cogs_brent = st.slider("Costi ~ Brent", 0.0, 1.5, 0.8)
    cogs_infl = st.slider("Costi ~ Infl", 0.0, 1.5, 0.5)
    cogs_fx = st.slider("Costi ~ FX", 0.0, 1.0, 0.1)
    opex_infl = st.slider("Spese ~ Infl", 0.0, 1.5, 0.8)
    da_infl = st.slider("DA ~ Infl", 0.0, 1.0, 0.3)

    tax_rate = st.slider("Tax rate", 0.0, 0.5, 0.28)

    st.subheader("Scenari")
    scn_name = st.text_input("Nome scenario", "Base")
    if st.button("Salva scenario"):
        st.session_state.scenarios[scn_name] = dict(locals())

# =======================
# BLOCCO 2/2
# =======================

def implied_levels_from_infl(infl_level, vol_brent, vol_fx, vol_infl, rho_bi, rho_fi):
    """
    Lega i driver: dato infl_level, calcola livelli "attesi" di Brent e FX
    coerenti con correlazione e volatilità.
    Approccio: cond. expectation di un MVN sui driver-shock.
    """
    # z-score dell'inflazione rispetto al riferimento
    z = 0.0 if vol_infl <= 0 else (infl_level - 0.025) / vol_infl

    # shock atteso su log-return Brent e FX
    brent_log_shift = rho_bi * vol_brent * z
    fx_log_shift    = rho_fi * vol_fx * z

    brent_level = float(BRENT_REF * np.exp(brent_log_shift))
    fx_level    = float(FX_REF * np.exp(fx_log_shift))

    # clamp a range plausibili (coerente con slider)
    brent_level = float(np.clip(brent_level, 50.0, 120.0))
    fx_level    = float(np.clip(fx_level, 0.95, 1.20))
    return brent_level, fx_level

def build_params(
    years, n_sims, seed,
    infl, brent, fx,
    vol_brent, vol_fx, vol_infl,
    rho_bf, rho_bi, rho_fi,
    rev0, cogs0, opex0, da0, extra_mean, extra_vol,
    rev_infl, rev_fx, cogs_brent, cogs_infl, cogs_fx, opex_infl, da_infl,
    tax_rate
):
    corr = np.array([[1.0, rho_bf, rho_bi],
                     [rho_bf, 1.0, rho_fi],
                     [rho_bi, rho_fi, 1.0]], dtype=float)

    base = {"rev": rev0, "cogs": cogs0, "opex": opex0, "da": da0, "extra": extra_mean, "extra_vol": extra_vol}
    betas = {
        "rev_infl": rev_infl, "rev_fx": rev_fx,
        "cogs_brent": cogs_brent, "cogs_infl": cogs_infl, "cogs_fx": cogs_fx,
        "opex_infl": opex_infl, "da_infl": da_infl,
    }

    params = dict(
        n_sims=int(n_sims),
        years=int(years),
        base=base,
        betas=betas,
        driver_mu={"brent": 0.0, "fx": 0.0, "infl": float(infl)},
        driver_vol={"brent": float(vol_brent), "fx": float(vol_fx), "infl": float(vol_infl)},
        corr=corr,
        level_shift={"brent_mult": float(brent) / BRENT_REF, "fx_mult": float(fx) / FX_REF},
        tax_rate=float(tax_rate),
        seed=int(seed),
    )
    return params

def summarize_percentiles(results, years):
    rows = []
    for label, key in KPIS.items():
        arr = results[key]  # (n_sims, years)
        for t in range(years):
            x = arr[:, t]
            p10, p50, p90 = np.percentile(x, [10, 50, 90])
            rows.append({
                "KPI": label, "Anno": t+1,
                "P10": p10, "P50": p50, "P90": p90,
                "Prob_<0": float(np.mean(x < 0))
            })
    return pd.DataFrame(rows)

def build_ce_table_p50(results, years):
    """
    Conto Economico tabellare (P50) per anno, con voci richieste:
    Ricavi, Costi, Spese, Ammortamenti, EBITDA, EBIT, Straordinari, EBT, Imposte, Utile Netto.
    Valori realistici e dinamici (derivano dal modello + driver).
    """
    rev_p50   = np.percentile(results["rev"],   50, axis=0)
    cogs_p50  = np.percentile(results["cogs"],  50, axis=0)
    opex_p50  = np.percentile(results["opex"],  50, axis=0)
    da_p50    = np.percentile(results["da"],    50, axis=0)
    extra_p50 = np.percentile(results["extra"], 50, axis=0)
    ebitda_p50 = rev_p50 - cogs_p50 - opex_p50
    ebit_p50   = ebitda_p50 - da_p50
    ebt_p50    = ebit_p50 + extra_p50

    tax_p50  = np.percentile(results["tax"], 50, axis=0)
    net_p50  = np.percentile(results["net"], 50, axis=0)

    df = pd.DataFrame({
        "Anno": np.arange(1, years+1),
        "Ricavi": rev_p50,
        "Costi": cogs_p50,
        "Spese operative": opex_p50,
        "Ammortamenti": da_p50,
        "EBITDA": ebitda_p50,
        "EBIT": ebit_p50,
        "Proventi/Oneri straordinari": extra_p50,
        "EBT": ebt_p50,
        "Imposte": tax_p50,
        "Utile Netto": net_p50,
    })
    return df

# -------------------------
# Driver linkati (richiesta)
# -------------------------
# Per legare davvero i driver: usiamo Inflazione come "master"
# e aggiorniamo Brent e FX automaticamente in base a corr+vol.
with st.sidebar:
    st.subheader("Driver linkati")
    link_drivers = st.toggle("Collega Brent & FX all'Inflazione", value=True)

# Se link_drivers ON, ricalcoliamo brent e fx coerenti con inflazione e corr/vol
# Nota: "infl, vol_*, rho_*" sono già definiti nel BLOCCO 1.
if link_drivers:
    brent_auto, fx_auto = implied_levels_from_infl(
        infl_level=infl,
        vol_brent=vol_brent, vol_fx=vol_fx, vol_infl=vol_infl,
        rho_bi=rho_bi, rho_fi=rho_fi
    )
    # Mostriamo a video (e usiamo nel modello) questi valori coerenti
    brent_used, fx_used = brent_auto, fx_auto
else:
    # Se non linkati, usa i livelli manuali selezionati nel BLOCCO 1
    brent_used, fx_used = brent, fx

# Piccolo pannello informativo
st.sidebar.write("---")
st.sidebar.write("**Driver effettivi usati**")
st.sidebar.write(f"Inflazione: **{infl:.3%}**")
st.sidebar.write(f"Brent: **{brent_used:.2f}** (ref {BRENT_REF})")
st.sidebar.write(f"EUR/USD: **{fx_used:.3f}** (ref {FX_REF})")

# -------------------------
# Costruzione parametri
# -------------------------
params = build_params(
    years=years, n_sims=n_sims, seed=seed,
    infl=infl, brent=brent_used, fx=fx_used,
    vol_brent=vol_brent, vol_fx=vol_fx, vol_infl=vol_infl,
    rho_bf=rho_bf, rho_bi=rho_bi, rho_fi=rho_fi,
    rev0=rev0, cogs0=cogs0, opex0=opex0, da0=da0, extra_mean=extra_mean, extra_vol=extra_vol,
    rev_infl=rev_infl, rev_fx=rev_fx,
    cogs_brent=cogs_brent, cogs_infl=cogs_infl, cogs_fx=cogs_fx,
    opex_infl=opex_infl, da_infl=da_infl,
    tax_rate=tax_rate
)

# -------------------------
# Run button + caching in session_state
# -------------------------
if "last_results" not in st.session_state:
    st.session_state.last_results = None
    st.session_state.last_params = None

# Regola: ricalcolo SOLO se premi Run o se non ho risultati
if run or (st.session_state.last_results is None):
    with st.spinner("Eseguo la simulazione Monte Carlo..."):
        results = simulate_pnl(**params)
    st.session_state.last_results = results
    st.session_state.last_params = params
else:
    results = st.session_state.last_results

# -------------------------
# Output principale: KPI selezionabile
# -------------------------
st.subheader("Risultati principali")

selected_kpi = st.selectbox("KPI per output", list(KPIS.keys()), index=0)
kpi_key = KPIS[selected_kpi]
kpi_sim = results[kpi_key]
last_year = kpi_sim[:, -1]

p10, p50, p90 = np.percentile(last_year, [10, 50, 90])
prob_negative = float(np.mean(last_year < 0))

col1, col2, col3, col4 = st.columns(4)
col1.metric("P10 ultimo anno", f"{p10:,.1f}")
col2.metric("P50 ultimo anno", f"{p50:,.1f}")
col3.metric("P90 ultimo anno", f"{p90:,.1f}")
col4.metric("P(KPI < 0)", f"{prob_negative*100:.1f}%")

# -------------------------
# Tabs: distribuzione / fan chart / CE tabellare / scenari / export
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Distribuzione (ultimo anno)",
    "Fan chart (anni)",
    "Conto Economico (tabella)",
    "Confronto scenari",
    "Export"
])

with tab1:
    fig_hist = px.histogram(
        last_year, nbins=70,
        title=f"Distribuzione {selected_kpi} nell'anno {years}",
        labels={"value": selected_kpi},
    )
    fig_hist.update_layout(bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    years_index = np.arange(1, years + 1)
    fan_df = pd.DataFrame({
        "Anno": years_index,
        "P10": np.percentile(kpi_sim, 10, axis=0),
        "P50": np.percentile(kpi_sim, 50, axis=0),
        "P90": np.percentile(kpi_sim, 90, axis=0),
    })

    fig_fan = go.Figure()
    fig_fan.add_trace(go.Scatter(
        x=fan_df["Anno"], y=fan_df["P90"],
        mode="lines", line=dict(color="rgba(255,100,100,0.8)"),
        name="P90"
    ))
    fig_fan.add_trace(go.Scatter(
        x=fan_df["Anno"], y=fan_df["P10"],
        mode="lines", fill="tonexty",
        line=dict(color="rgba(100,150,255,0.8)"),
        name="P10"
    ))
    fig_fan.add_trace(go.Scatter(
        x=fan_df["Anno"], y=fan_df["P50"],
        mode="lines+markers", line=dict(color="black", width=2),
        name="P50"
    ))
    fig_fan.update_layout(
        title=f"Fan chart {selected_kpi} anni 1–{years}",
        xaxis_title="Anno",
        yaxis_title=selected_kpi,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_fan, use_container_width=True)

with tab3:
    st.markdown("### Conto Economico (valori **P50** per anno, in milioni)")
    ce_df = build_ce_table_p50(results, years)

    # formattazione
    fmt_df = ce_df.copy()
    for col in fmt_df.columns:
        if col != "Anno":
            fmt_df[col] = fmt_df[col].map(lambda x: f"{x:,.1f}")
    st.dataframe(fmt_df, use_container_width=True)

    st.caption("I valori derivano dalla simulazione (P50) e quindi variano al variare dei driver e delle elasticità.")

with tab4:
    st.markdown("### Confronto scenari (PDF sovrapposte)")
    if len(st.session_state.scenarios) == 0:
        st.info("Nessuno scenario salvato. Usa la sidebar → 'Salva scenario'.")
    else:
        scn_list = list(st.session_state.scenarios.keys())
        compare = st.multiselect("Scegli scenari da confrontare", scn_list, default=scn_list[:1])
        kpi_comp = st.selectbox("KPI confronto", list(KPIS.keys()), index=0, key="kpi_compare")
        kpi_comp_key = KPIS[kpi_comp]

        if compare:
            fig = go.Figure()
            for scn in compare:
                s = st.session_state.scenarios[scn]

                # ricostruisci parametri scenario
                # (rispetta anche driver linkati: se vuoi linkati negli scenari, imposta link_drivers=True e salva)
                brent_s = s["brent"]
                fx_s = s["fx"]
                infl_s = s["infl"]

                corr_s = np.array([[1.0, s["rho_bf"], s["rho_bi"]],
                                   [s["rho_bf"], 1.0, s["rho_fi"]],
                                   [s["rho_bi"], s["rho_fi"], 1.0]], dtype=float)

                params_s = build_params(
                    years=s["years"], n_sims=s["n_sims"], seed=s["seed"],
                    infl=infl_s, brent=brent_s, fx=fx_s,
                    vol_brent=s["vol_brent"], vol_fx=s["vol_fx"], vol_infl=s["vol_infl"],
                    rho_bf=s["rho_bf"], rho_bi=s["rho_bi"], rho_fi=s["rho_fi"],
                    rev0=s["rev0"], cogs0=s["cogs0"], opex0=s["opex0"], da0=s["da0"],
                    extra_mean=s["extra_mean"], extra_vol=s["extra_vol"],
                    rev_infl=s["rev_infl"], rev_fx=s["rev_fx"],
                    cogs_brent=s["cogs_brent"], cogs_infl=s["cogs_infl"], cogs_fx=s["cogs_fx"],
                    opex_infl=s["opex_infl"], da_infl=s["da_infl"],
                    tax_rate=s["tax_rate"]
                )
                params_s["corr"] = corr_s

                r_s = simulate_pnl(**params_s)
                x = r_s[kpi_comp_key][:, -1]

                fig.add_trace(go.Histogram(
                    x=x, nbinsx=70, name=scn, opacity=0.45
                ))

            fig.update_layout(
                barmode="overlay",
                title=f"Confronto PDF {kpi_comp} (anno {years})",
                xaxis_title=kpi_comp, yaxis_title="Frequenza"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown("### Export")
    summary_df = summarize_percentiles(results, years)
    st.dataframe(summary_df, use_container_width=True)

    csv_summary = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Scarica percentili (CSV)",
        data=csv_summary,
        file_name="percentili_kpi.csv",
        mime="text/csv"
    )

    ce_df = build_ce_table_p50(results, years)
    csv_ce = ce_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Scarica CE tabellare (P50) (CSV)",
        data=csv_ce,
        file_name="conto_economico_p50.csv",
        mime="text/csv"
    )

st.write("---")
st.write("**Matrice di correlazione driver:**")
st.write(pd.DataFrame(
    params["corr"],
    index=["Brent", "FX", "Inflazione"],
    columns=["Brent", "FX", "Inflazione"]
))
