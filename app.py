import streamlit as st
from datetime import date
import pandas as pd
import altair as alt

from Portafolio import (
    get_price_data,
    calc_portfolio_metrics,
    generate_text_report,
)

# --------------------------------------------------
# Configuración general de la app
# --------------------------------------------------
st.set_page_config(
    page_title="Calculadora de Portafolio",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Calculadora de Portafolio de Inversión Personal")
st.write(
    "Selecciona tus activos, pesos y periodo. "
    "La app descargará datos de mercado, calculará métricas de riesgo/rendimiento "
    "y generará un reporte de tu portafolio."
)

# --------------------------------------------------
# 1. Inputs del usuario (sidebar)
# --------------------------------------------------
st.sidebar.header("Configuración del portafolio")

tickers_input = st.sidebar.text_input(
    "Tickers (separados por comas)",
    value="AAPL, MSFT, VOO",
    help="Ejemplo: AAPL,MSFT,VOO",
)

weights_input = st.sidebar.text_input(
    "Pesos (mismos activos, separados por comas)",
    value="40,30,30",
    help="Puedes usar porcentajes (40,30,30) o proporciones (0.4,0.3,0.3).",
)

start_date = st.sidebar.date_input(
    "Fecha de inicio",
    value=date(2020, 1, 1),
)

end_date = st.sidebar.date_input(
    "Fecha de fin",
    value=date.today(),
    max_value=date.today(),
)

rf_input = st.sidebar.text_input(
    "Tasa libre de riesgo anual (ej. 0.04 = 4%)",
    value="0.04",
)

benchmark_input = st.sidebar.text_input(
    "Ticker benchmark (opcional, para beta)",
    value="^GSPC",
    help="Ejemplo: ^GSPC para S&P 500. Deja vacío si no quieres calcular beta.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Monte Carlo")

mc_sims = st.sidebar.number_input(
    "Número de simulaciones",
    min_value=0,
    max_value=20000,
    value=1000,
    step=500,
    help="Pon 0 si no quieres simulación Monte Carlo.",
)

mc_years = st.sidebar.number_input(
    "Horizonte Monte Carlo (años)",
    min_value=0.25,
    max_value=10.0,
    value=1.0,
    step=0.25,
)

analyze_button = st.sidebar.button("Calcular portafolio")


# --------------------------------------------------
# Funciones auxiliares para parsear inputs
# --------------------------------------------------
def parse_tickers(tickers_str: str):
    return [t.strip().upper() for t in tickers_str.split(",") if t.strip() != ""]


def parse_weights(weights_str: str, n_assets: int):
    if not weights_str:
        return None, "No ingresaste pesos."
    raw = [w.strip().replace("%", "") for w in weights_str.split(",") if w.strip() != ""]
    if len(raw) != n_assets:
        return None, f"Debes ingresar exactamente {n_assets} pesos."
    try:
        weights = [float(x) for x in raw]
    except ValueError:
        return None, "Todos los pesos deben ser numéricos."
    if sum(weights) <= 0:
        return None, "La suma de los pesos debe ser mayor que 0."
    return weights, None


# --------------------------------------------------
# 2. Lógica principal
# --------------------------------------------------
if analyze_button:
    try:
        # ---- Parsear tickers y pesos ----
        tickers = parse_tickers(tickers_input)
        if len(tickers) == 0:
            st.error("Debes ingresar al menos un ticker válido.")
        else:
            weights, err = parse_weights(weights_input, len(tickers))
            if err:
                st.error(err)
            else:
                # ---- Tasa libre de riesgo ----
                try:
                    risk_free = float(rf_input)
                except ValueError:
                    st.error("La tasa libre de riesgo debe ser un número (ej. 0.04).")
                    risk_free = None

                if risk_free is not None:
                    start_str = start_date.strftime("%Y-%m-%d")
                    end_str = end_date.strftime("%Y-%m-%d")

                    # ---- Descargar precios del portafolio ----
                    st.info("Descargando datos de mercado...")
                    prices = get_price_data(
                        tickers,
                        start_str,
                        end_str,
                    )

                    if prices.empty:
                        st.error("No se obtuvieron precios. Revisa tickers o fechas.")
                    else:
                        # ---- Descargar benchmark opcional ----
                        benchmark_prices = None
                        bench_ticker = benchmark_input.strip().upper()
                        if bench_ticker:
                            bench_df = get_price_data(
                                [bench_ticker],
                                start_str,
                                end_str,
                            )
                            if bench_df.empty:
                                st.warning(
                                    "No se pudieron obtener datos del benchmark; "
                                    "beta no se calculará."
                                )
                            else:
                                benchmark_prices = bench_df

                        st.success("Datos descargados correctamente. Calculando métricas...")

                        # Horizonte de Monte Carlo en días
                        mc_horizon_days = int(mc_years * 252)

                        # ---- Calcular métricas del portafolio ----
                        metrics = calc_portfolio_metrics(
                            prices,
                            weights,
                            risk_free=risk_free,
                            benchmark_prices=benchmark_prices,
                            mc_sims=int(mc_sims),
                            mc_horizon_days=mc_horizon_days,
                        )

                        # --------------------------------------------------
                        # 3. Mostrar métricas principales
                        # --------------------------------------------------
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Métricas del portafolio")
                            st.metric(
                                "Rendimiento esperado anual",
                                f"{metrics['mean_annual']*100:.2f} %",
                            )
                            st.metric(
                                "Desv. estándar (volatilidad) anual",
                                f"{metrics['vol_annual']*100:.2f} %",
                            )
                            st.metric(
                                "Desv. estándar diaria",
                                f"{metrics['vol_daily']*100:.2f} %",
                            )
                            st.metric(
                                "Sharpe ratio",
                                f"{metrics['sharpe']:.2f}",
                            )

                        with col2:
                            st.subheader("Riesgo")
                            st.metric(
                                "Máx. drawdown histórico",
                                f"{metrics['max_drawdown']*100:.2f} %",
                            )
                            st.metric(
                                "VaR 95% diario (histórico)",
                                f"{metrics['var_95']*100:.2f} %",
                            )
                            st.metric(
                                "CVaR 95% diario (histórico)",
                                f"{metrics['cvar_95']*100:.2f} %",
                            )
                            beta_val = metrics.get("beta")
                            if beta_val is not None and not pd.isna(beta_val):
                                st.metric(
                                    f"Beta vs {bench_ticker if bench_ticker else 'benchmark'}",
                                    f"{beta_val:.2f}",
                                )

                        # --------------------------------------------------
                        # 4. Tabla de composición
                        # --------------------------------------------------
                        st.subheader("Composición del portafolio")
                        st.dataframe(
                            metrics["asset_metrics"].style.format({
                                "Peso": "{:.2%}",
                                "Rend_Anual_Esp": "{:.2%}",
                                "Vol_Anual": "{:.2%}",
                            })
                        )

                        # --------------------------------------------------
                        # 5. Gráficas históricas
                        # --------------------------------------------------
                        st.subheader("Evolución del valor del portafolio")
                        st.line_chart(metrics["port_value"])

                        st.subheader("Drawdown")
                        port_value = metrics["port_value"]
                        running_max = port_value.cummax()
                        drawdown = port_value / running_max - 1
                        st.line_chart(drawdown)

                        # --------------------------------------------------
                        # 6. Monte Carlo (spaghetti plot)
                        # --------------------------------------------------
                        mc = metrics.get("mc_results")
                        if mc is not None:
                            st.subheader("Simulación Monte Carlo - Trayectorias del portafolio")

                            st.write(
                                f"{mc['n_sims']} simulaciones, "
                                f"horizonte de {mc['horizon_days']} días "
                                f"({mc['horizon_days']/252:.2f} años aprox.)"
                            )

                            col_mc1, col_mc2, col_mc3, col_mc4 = st.columns(4)
                            col_mc1.metric("Media simulada (retorno)", f"{mc['mean']*100:.2f} %")
                            col_mc2.metric(
                                "Desv. estándar simulada",
                                f"{mc['std']*100:.2f} %",
                            )
                            col_mc3.metric("VaR 95% (MC)", f"{mc['var_95']*100:.2f} %")
                            col_mc4.metric("CVaR 95% (MC)", f"{mc['cvar_95']*100:.2f} %")

                            # paths: matriz (horizon_days+1, n_sims)
                            paths = mc["paths"]
                            n_steps, n_paths = paths.shape

                            # Para que no sea pesado, graficamos máximo 50 trayectorias
                            n_plot = min(50, n_paths)
                            df_paths = pd.DataFrame(paths[:, :n_plot])
                            df_paths["Día"] = range(n_steps)

                            df_long = df_paths.melt(
                                id_vars="Día",
                                var_name="Simulación",
                                value_name="Valor_portafolio",
                            )

                            chart = (
                                alt.Chart(df_long)
                                .mark_line(opacity=0.4)
                                .encode(
                                    x=alt.X("Día:Q", title="Días"),
                                    y=alt.Y(
                                        "Valor_portafolio:Q",
                                        title="Valor del portafolio (inicial = 1)",
                                    ),
                                    color=alt.Color("Simulación:N", legend=None),
                                )
                            )

                            st.altair_chart(chart, use_container_width=True)

                        # --------------------------------------------------
                        # 7. Reporte en texto
                        # --------------------------------------------------
                        st.subheader("Reporte en texto")
                        report_text = generate_text_report(metrics, risk_free=risk_free)
                        st.text(report_text)

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
        st.stop()
else:
    st.info(
        "Configura tu portafolio en la barra lateral y haz clic en 'Calcular portafolio'."
    )
