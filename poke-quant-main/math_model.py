import numpy as np
import pandas as pd

class VolatilityForecaster:
    def __init__(self, price_series):
        self.prices = price_series
        self.log_returns = np.log(self.prices / self.prices.shift(1)).dropna()

    def calculate_metrics(self):
        # 1. Rolling 30-day Volatility (Annualized or standard deviation of returns)
        rolling_vol = self.log_returns.rolling(window=30).std() * np.sqrt(252)
        current_vol = float(rolling_vol.iloc[-1]) if not rolling_vol.empty else 0.20

        # 2. Exponentially Weighted Moving Average (EWMA) Volatility
        ewma_vol = float(self.log_returns.ewm(span=30).std().iloc[-1] * np.sqrt(252))

        # 3. Monte Carlo Value at Risk (VaR 95%)
        simulated_returns = np.random.normal(self.log_returns.mean(), self.log_returns.std(), (1000, 30))
        simulated_paths = self.prices.iloc[-1] * np.cumprod(1 + simulated_returns, axis=1)
        final_prices = simulated_paths[:, -1]
        var_95 = float(self.prices.iloc[-1] - np.percentile(final_prices, 5))

        # 4. Expected ROI & Risk Score (1-10)
        expected_roi = float((final_prices.mean() - self.prices.iloc[-1]) / self.prices.iloc[-1] * 100)
        risk_score = min(max(int(current_vol * 15), 1), 10)

        return {
            "current_volatility": round(current_vol * 100, 2),
            "ewma_volatility": round(ewma_vol * 100, 2),
            "var_95": round(var_95, 2),
            "expected_roi": round(expected_roi, 2),
            "risk_score": risk_score,
            "simulated_paths": simulated_paths
        }

    def forecast_jump_diffusion(self, horizon=30):
        """Geometric Brownian Motion with Jump Diffusion for future paths."""
        last_price = self.prices.iloc[-1]
        dt = 1 / 252
        mu = self.log_returns.mean()
        sigma = self.log_returns.std()
        
        # Jump parameters
        lam = 0.05  # Jump intensity
        jump_mean = 0.0
        jump_std = 0.15

        paths = np.zeros((50, horizon))
        for i in range(50):
            path = [last_price]
            for t in range(1, horizon):
                dz = np.random.normal(0, np.sqrt(dt))
                # Poisson jump trigger
                jump = np.random.poisson(lam * dt) * np.random.normal(jump_mean, jump_std)
                price_next = path[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dz + jump)
                path.append(price_next)
            paths[i, :] = path

        future_dates = pd.date_range(start=self.prices.index[-1] + pd.Timedelta(days=1), periods=horizon)
        return future_dates, paths
