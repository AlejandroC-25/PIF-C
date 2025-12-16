import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime


def get_price_data(tickers, start_date, end_date):
    """
    Descarga precios de cierre (ajustados) para una lista de tickers
    desde Yahoo Finance y devuelve un DataFrame con una columna por ticker.
    Es robusta a cambios de columnas de yfinance ('Adj Close', 'Close', MultiIndex, etc.).
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,   # ya ajusta dividendos/splits
        progress=False,
    )

    if data.empty:
        return pd.DataFrame()

    # Caso MultiIndex (varios tickers, OHLCV)
    if isinstance(data.columns, pd.MultiIndex):
        # Intentar usar nivel 'Close'
        if "Close" in data.columns.levels[0]:
            prices = data["Close"]
        else:
            # Si no existe, tomamos el primer nivel disponible
            first_level = data.columns.levels[0][0]
            prices = data[first_level]
    else:
        # Caso columnas simples
        if "Close" in data.columns:
            series = data["Close"]
            prices = series.to_frame() if isinstance(series, pd.Series) else data[["Close"]]
        else:
            # Último recurso: usamos todo el DataFrame
            prices = data

    prices = prices.sort_index(axis=1)
    return prices.dropna(how="all")


def _calc_beta(port_returns: pd.Series, benchmark_prices: pd.DataFrame | pd.Series | None):
    """
    Calcula beta del portafolio contra un benchmark (si se proporciona).
    """
    if benchmark_prices is None or len(benchmark_prices) == 0:
        return np.nan

    # Convertir a serie de rendimientos del benchmark
    if isinstance(benchmark_prices, pd.DataFrame):
        if benchmark_prices.shape[1] == 0:
            return np.nan
        bench_ret = benchmark_prices.iloc[:, 0].pct_change().dropna()
    else:
        bench_ret = benchmark_prices.pct_change().dropna()

    # Alinear fechas
    df = pd.concat([port_returns, bench_ret], axis=1, join="inner").dropna()
    if df.shape[0] < 2:
        return np.nan

    port_r = df.iloc[:, 0]
    bench_r = df.iloc[:, 1]
    var_bench = bench_r.var(ddof=1)
    if var_bench == 0:
        return np.nan

    cov = np.cov(port_r, bench_r, ddof=1)[0, 1]
    beta = cov / var_bench
    return beta


def _monte_carlo_bootstrap(port_returns: pd.Series, n_sims: int, horizon_days: int):
    """
    Monte Carlo simple por bootstrap de rendimientos diarios del portafolio.
    Devuelve:
      - paths: matriz (horizon_days+1, n_sims) con la trayectoria del valor del portafolio
      - returns: retorno total de cada simulación
    El valor inicial del portafolio es 1.0
    """
    if n_sims <= 0 or horizon_days <= 0:
        return None

    ret_values = port_returns.dropna().values
    if len(ret_values) == 0:
        return None

    # paths[fila, columna] = valor del portafolio en ese día y simulación
    paths = np.zeros((horizon_days + 1, n_sims), dtype=float)
    paths[0, :] = 1.0  # día 0, valor inicial

    for j in range(n_sims):
        # muestreamos rendimientos diarios con reemplazo
        sample = np.random.choice(ret_values, size=horizon_days, replace=True)
        # trayectoria de valor del portafolio
        path = np.cumprod(1 + sample)
        paths[1:, j] = path

    # retorno total de cada simulación (último valor - 1)
    final_returns = paths[-1, :] - 1.0

    mc_mean = final_returns.mean()
    mc_std = final_returns.std(ddof=1)
    mc_var_95 = np.percentile(final_returns, 5)
    mc_cvar_95 = final_returns[final_returns <= mc_var_95].mean()

    return {
        "paths": paths,           # NUEVO: trayectorias completas
        "returns": final_returns, # retornos totales
        "mean": mc_mean,
        "std": mc_std,
        "var_95": mc_var_95,
        "cvar_95": mc_cvar_95,
        "n_sims": n_sims,
        "horizon_days": horizon_days,
    }

    """
    Monte Carlo simple por bootstrap de rendimientos diarios del portafolio.
    Devuelve un dict con los resultados y un array de retornos totales del horizonte.
    """
    if n_sims <= 0 or horizon_days <= 0:
        return None

    ret_values = port_returns.values
    if len(ret_values) == 0:
        return None

    sims = []
    for _ in range(n_sims):
        sample = np.random.choice(ret_values, size=horizon_days, replace=True)
        total_return = np.prod(1 + sample) - 1  # retorno total del periodo
        sims.append(total_return)

    sims = np.array(sims)
    mc_mean = sims.mean()
    mc_std = sims.std(ddof=1)
    mc_var_95 = np.percentile(sims, 5)
    mc_cvar_95 = sims[sims <= mc_var_95].mean()

    return {
        "returns": sims,
        "mean": mc_mean,
        "std": mc_std,
        "var_95": mc_var_95,
        "cvar_95": mc_cvar_95,
        "n_sims": n_sims,
        "horizon_days": horizon_days,
    }


def calc_portfolio_metrics(
    prices,
    weights,
    risk_free: float = 0.04,
    benchmark_prices=None,
    mc_sims: int = 1000,
    mc_horizon_days: int = 252,
):
    """
    Calcula métricas de un portafolio a partir de precios históricos.
    Incluye:
    - rendimiento esperado anual
    - desviación estándar (volatilidad) anual
    - Sharpe
    - Máx. drawdown
    - VaR/CVaR históricos
    - Beta vs benchmark (si se da)
    - Simulación Monte Carlo (bootstrap) del retorno del horizonte.
    """
    prices = prices.dropna()
    returns = prices.pct_change().dropna()

    tickers = list(prices.columns)
    weights = np.array(weights, dtype=float)

    if len(weights) != len(tickers):
        raise ValueError("La cantidad de pesos no coincide con la cantidad de activos.")

    # Normalizar por si no suman exactamente 1
    weights = weights / weights.sum()

    # Rendimiento diario del portafolio
    port_returns = returns.dot(weights)

    # Métricas diarias
    mean_daily = port_returns.mean()
    vol_daily = port_returns.std(ddof=1)

    trading_days = 252
    mean_annual = mean_daily * trading_days
    vol_annual = vol_daily * np.sqrt(trading_days)

    # Sharpe ratio
    excess_return = mean_annual - risk_free
    sharpe = excess_return / vol_annual if vol_annual > 0 else np.nan

     # Sharpe ratio
    excess_return = mean_annual - risk_free
    sharpe = excess_return / vol_annual if vol_annual > 0 else np.nan

        # Matriz de covarianzas anual entre activos
    cov_daily = returns.cov()
    cov_annual = cov_daily * trading_days

    # Beta del portafolio vs benchmark (si hay)
    beta = _calc_beta(port_returns, benchmark_prices)

    # Treynor ratio: (Exceso de rendimiento) / beta
    if beta is None or np.isnan(beta) or beta == 0:
        treynor = np.nan
    else:
        treynor = excess_return / beta


    # Curva de valor del portafolio (iniciando en 1)
    port_value = (1 + port_returns).cumprod()

    # Máximo drawdown
    running_max = port_value.cummax()
    drawdown = port_value / running_max - 1
    max_drawdown = drawdown.min()

    # VaR y CVaR (95% histórico)
    var_level = 5  # 5% -> 95% de confianza
    var_95 = np.percentile(port_returns, var_level)
    cvar_95 = port_returns[port_returns <= var_95].mean()

    # Métricas por activo
    asset_mean_daily = returns.mean()
    asset_vol_daily = returns.std(ddof=1)

    asset_mean_annual = asset_mean_daily * trading_days
    asset_vol_annual = asset_vol_daily * np.sqrt(trading_days)

    asset_metrics = pd.DataFrame({
        "Ticker": tickers,
        "Peso": weights,
        "Rend_Anual_Esp": asset_mean_annual,
        "Vol_Anual": asset_vol_annual
    })

    # Beta
    beta = _calc_beta(port_returns, benchmark_prices)

    # Monte Carlo
    mc_results = _monte_carlo_bootstrap(port_returns, mc_sims, mc_horizon_days)

    metrics = {
        "tickers": tickers,
        "weights": weights,
        "start_date": prices.index.min(),
        "end_date": prices.index.max(),
        "returns": returns,
        "port_returns": port_returns,
        "port_value": port_value,
        "mean_annual": mean_annual,
        "vol_annual": vol_annual,
        "vol_daily": vol_daily,
        "sharpe": sharpe,
        "treynor": treynor,
        "cov_matrix": cov_annual,
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "asset_metrics": asset_metrics,
        "beta": beta,
        "mc_results": mc_results,
    }

    return metrics


def generate_text_report(metrics, risk_free: float = 0.04):
    """
    Genera un reporte de texto a partir del dict de métricas.
    """
    start = metrics["start_date"].strftime("%Y-%m-%d")
    end = metrics["end_date"].strftime("%Y-%m-%d")

    mean_annual = metrics["mean_annual"]
    vol_annual = metrics["vol_annual"]
    sharpe = metrics["sharpe"]
    max_dd = metrics["max_drawdown"]
    var_95 = metrics["var_95"]
    cvar_95 = metrics["cvar_95"]
    beta = metrics.get("beta", np.nan)

    asset_df = metrics["asset_metrics"].copy()
    asset_df = asset_df.sort_values("Peso", ascending=False)

    top_weight = asset_df.iloc[0]
    most_volatile = asset_df.sort_values("Vol_Anual", ascending=False).iloc[0]

    lines = []
    lines.append(f"Análisis de Portafolio ({start} a {end})")
    lines.append("-" * 50)
    lines.append(f"Rendimiento esperado anual del portafolio: {mean_annual*100:.2f}%")
    lines.append(f"Desv. estándar (volatilidad) anual:       {vol_annual*100:.2f}%")
    lines.append(f"Sharpe ratio (rf = {risk_free*100:.2f}%):            {sharpe:.2f}")
    if not np.isnan(beta):
        lines.append(f"Beta versus benchmark:                    {beta:.2f}")
    lines.append("")
    lines.append(f"Máximo drawdown histórico:               {max_dd*100:.2f}%")
    lines.append(f"VaR 95% diario (histórico):              {var_95*100:.2f}%")
    lines.append(f"CVaR 95% diario (histórico):             {cvar_95*100:.2f}%")
    lines.append("")
    lines.append("Composición del portafolio:")
    for _, row in asset_df.iterrows():
        lines.append(
            f"  - {row['Ticker']}: peso {row['Peso']*100:.2f}%, "
            f"rend. anual esp. {row['Rend_Anual_Esp']*100:.2f}%, "
            f"vol. anual {row['Vol_Anual']*100:.2f}%"
        )
    lines.append("")
    lines.append(
        f"El activo con mayor peso es {top_weight['Ticker']} "
        f"({top_weight['Peso']*100:.2f}% del portafolio)."
    )
    lines.append(
        f"El activo más volátil en el periodo fue {most_volatile['Ticker']} "
        f"con una volatilidad anual de {most_volatile['Vol_Anual']*100:.2f}%."
    )

    # Resumen Monte Carlo si existe
    mc = metrics.get("mc_results")
    if mc is not None:
        lines.append("")
        lines.append(
            f"Monte Carlo ({mc['n_sims']} simulaciones, "
            f"{mc['horizon_days']} días de horizonte):"
        )
        lines.append(f"  • Media de retornos simulados:         {mc['mean']*100:.2f}%")
        lines.append(f"  • Desv. estándar simulada:             {mc['std']*100:.2f}%")
        lines.append(f"  • VaR 95% (MC):                         {mc['var_95']*100:.2f}%")
        lines.append(f"  • CVaR 95% (MC):                        {mc['cvar_95']*100:.2f}%")

    return "\n".join(lines)
