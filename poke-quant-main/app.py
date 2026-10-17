import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import db
import data_generator
import price_ui
from scanner import CardScanner
from math_model import VolatilityForecaster

st.set_page_config(page_title="PokeQuant: Local CV & Volatility Engine", layout="wide")

@st.cache_resource
def init_engine():
    # First run: build the made-up card database from seed_cards.csv
    if not db.has_data():
        data_generator.seed_database()
    return CardScanner()

scanner = init_engine()

st.title("⚡ PokeQuant: Local Card Scanner & Volatility Engine")
st.markdown("Scan or upload a Pokémon card to analyze historical trends, quantitative risk, and AI-driven price simulations **completely offline**.")

MIN_HISTORY = 35   # the volatility model needs a 30-day window plus a few days

tab_analyze, tab_prices = st.tabs(["📷 Scan & Analyze", "💲 Load Real Prices"])

# (This tab comes first in the code so anything saved here is visible to the analysis below on the same run.)
with tab_prices:
    price_ui.render()

with tab_analyze:
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("1. Card Scanner & Detection Feed")
        input_mode = st.radio("Select Input Mode", ["Upload Image", "Simulate Live Camera Frame", "Browse Database"])

        scanned_card = None
    
        if input_mode == "Upload Image":
            uploaded_file = st.file_uploader("Upload Card Image", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
            
                # Process scanner
                warped, approx = scanner.extract_card_contour(frame)
                display_frame = frame.copy()
                if approx is not None:
                    cv2.drawContours(display_frame, [approx], -1, (0, 255, 0), 3)
            
                st.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), caption="Detected Card Outline", use_column_width=True)
            
                target_frame = warped if warped is not None else frame
                scanned_card = scanner.match_card(target_frame)

        elif input_mode == "Simulate Live Camera Frame":
            st.info("Simulating live camera detection feed from sample database...")
            # Mocking live camera pick from database items
            db_keys = list(scanner.reference_database.keys())
            if db_keys:
                sample_key = st.selectbox("Select Mock Camera Target", db_keys)
                sample_data = scanner.reference_database[sample_key]
                mock_frame = cv2.cvtColor(sample_data["image"], cv2.COLOR_GRAY2BGR)
                st.image(mock_frame, caption="Live Camera Frame", use_column_width=True)
                scanned_card = scanner.match_card(mock_frame)
            else:
                st.warning("Please add reference images to `assets/database/` to run camera detection.")

        else:
            # No image needed: pick any card straight from the database
            all_cards = db.list_cards()
            pick = st.selectbox("Select a card", list(all_cards["card_id"]),
                                format_func=lambda cid: all_cards.set_index("card_id").loc[cid, "name"])
            scanned_card = db.get_card(pick)
            scanned_card["score"] = 100.0

    with col_right:
        st.subheader("2. Identified Card Profile & Risk Metrics")
    
        if scanned_card:
            c1, c2, c3 = st.columns(3)
            c1.metric("Card Name", scanned_card["name"])
            c2.metric("Rarity", scanned_card["rarity"])
            c3.metric("Set", scanned_card["set"])

            # Load financial history from the database
            df_history = db.get_price_history(scanned_card["card_id"], days=180)

            if df_history.empty:
                st.warning(f"No price history in the database for '{scanned_card['card_id']}'. "
                           "Add it to `seed_cards.csv` and run `python data_generator.py`, "
                           "or insert rows with `db.add_price(...)`.")
            else:
                prices = df_history["Market Price ($)"]

                n_points, sources = db.get_card_sources(scanned_card["card_id"])
                if "mock" in sources:
                    st.caption(f"Data: made-up ({n_points} days)" + (" + real" if sources - {"mock"} else ""))
                else:
                    st.caption(f"Data: real prices ({n_points} days; source: {', '.join(sorted(sources))})")

                if len(prices) < MIN_HISTORY:
                    st.metric("Latest price", f"${prices.iloc[-1]:,.2f}")
                    st.info(f"Only {len(prices)} price point(s) so far - the volatility model needs at least "
                            f"{MIN_HISTORY}. Record prices daily on the **Load Real Prices** tab, or import "
                            "past prices from a CSV there.")
                    if len(prices) >= 2:
                        st.line_chart(prices)
                    st.stop()

                # Run Quantitative Volatility Model
                forecaster = VolatilityForecaster(prices)
                metrics = forecaster.calculate_metrics()

                st.markdown("---")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Volatility", f"{metrics['current_volatility']}%")
                m2.metric("EWMA Volatility", f"{metrics['ewma_volatility']}%")
                m3.metric("95% VaR (30d)", f"${metrics['var_95']}")
                m4.metric("Risk Score", f"{metrics['risk_score']} / 10", delta=f"{metrics['expected_roi']}% ROI")

                # Forecast Price Paths
                future_dates, mc_paths = forecaster.forecast_jump_diffusion(horizon=30)

                # Plotly Interactive Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_history.index, y=prices, mode='lines', name='Historical Price', line=dict(color='#1f77b4', width=2)))

                # Add Monte Carlo simulation bands
                for i in range(min(20, mc_paths.shape[0])):
                    fig.add_trace(go.Scatter(x=future_dates, y=mc_paths[i], mode='lines', line=dict(color='rgba(255, 127, 14, 0.15)'), showlegend=False))

                fig.update_layout(
                    title=f"180-Day History & 30-Day Jump-Diffusion Forecast ({scanned_card['name']})",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_dark",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
