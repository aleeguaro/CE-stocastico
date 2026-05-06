import streamlit as st
import numpy as np
import plotly.express as px
from model import simulate_pnl

st.set_page_config(layout="wide")
st.title("Conto Economico stocastico – simulazioni PDF")

BRENT_REF = 80.0
FX_REF = 1.08

with st.sidebar:
    years = st.slider("Anni", 1, 5, 3)
    n_sims = st.select_slider("Simulazioni", [1000, 5000, 10000, 20000], 10000)

    brent = st.slider("Brent", 50.0, 120.0, 80.0)
    fx = st.slider("EUR/USD", 0.95, 1.20, 1.08)
    infl = st.slider("Inflazione", 0.0, 0.06, 0.025)

    vol_brent = st.slider("Vol Brent", 0.05, 0.6, 0.25)
    vol_fx = st.slider("Vol FX", 0.03, 0.3, 0.1)
    vol_infl = st.slider("Vol Infl", 0.002, 0.05, 0.01)

    rev0 = st.number_input("Ricavi", 1000.0)
    cogs0 = st.number_input("Costi", 600.0)
    opex0 = st.number_input("Spese", 250.0)
    da0 = st.number_input("Ammortamenti", 80.0)

    tax_rate = st.slider("Tax rate", 0.0, 0.5, 0.28)

corr = np.array([[1, 0.1, 0.2],
                 [0.1, 1, 0.1],
                 [0.2, 0.1, 1]])

res = simulate_pnl(
    n_sims, years,
    base={"rev": rev0, "cogs": cogs0, "opex": opex0, "da": da0,
          "extra": 0, "extra_vol": 20},
    betas={"rev_infl": 0.7, "rev_fx": 0.2,
           "cogs_brent": 0.8, "cogs_infl": 0.5, "cogs_fx": 0.1,
           "opex_infl": 0.8, "da_infl": 0.3},
    driver_mu={"brent": 0, "fx": 0, "infl": infl},
    driver_vol={"brent": vol_brent, "fx": vol_fx, "infl": vol_infl},
    corr=corr,
    level_shift={"brent_mult": brent/BRENT_REF, "fx_mult": fx/FX_REF},
    tax_rate=tax_rate
)

x = res["ebitda"][:, -1]
fig = px.histogram(x, nbins=60, title="Distribuzione EBITDA (ultimo anno)")
st.plotly_chart(fig, use_container_width=True)

p10, p50, p90 = np.percentile(x, [10, 50, 90])
st.write(f"P10: {p10:.1f} – P50: {p50:.1f} – P90: {p90:.1f}")
st.write(f"P(EBITDA < 0): {(x<0).mean()*100:.1f}%")
