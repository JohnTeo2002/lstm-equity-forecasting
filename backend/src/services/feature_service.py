from typing import List, Dict, Any, Tuple


class FeatureService:
    """Computes technical indicators and handles MinMax scaling for LSTM inputs."""

    @staticmethod
    def compute_sma(prices: List[float], window: int) -> List[float]:
        sma = []
        for i in range(len(prices)):
            if i < window - 1:
                sma.append(prices[i])
            else:
                window_prices = prices[i - window + 1 : i + 1]
                sma.append(sum(window_prices) / window)
        return sma

    @staticmethod
    def compute_rsi(prices: List[float], period: int = 14) -> List[float]:
        if len(prices) < 2:
            return [50.0] * len(prices)

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        rsi = [50.0]

        gains = [max(d, 0.0) for d in deltas]
        losses = [abs(min(d, 0.0)) for d in deltas]

        avg_gain = sum(gains[:period]) / max(period, 1)
        avg_loss = sum(losses[:period]) / max(period, 1)

        for i in range(len(deltas)):
            if i < period:
                rsi.append(50.0)
                continue

            gain = gains[i]
            loss = losses[i]

            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

            if avg_loss == 0:
                rsi.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100.0 - (100.0 / (1.0 + rs)))

        return rsi

    def prepare_features(self, bars: List[Dict[str, Any]]) -> Tuple[List[List[float]], float, float]:
        """
        Extract features from historical bars and compute MinMax bounds for price scaling.
        Returns:
            features: 2D list of feature vectors [step, feature_dim]
            min_price: Minimum close price for scaling
            max_price: Maximum close price for scaling
        """
        closes = [b["close"] for b in bars]
        volumes = [float(b["volume"]) for b in bars]

        min_price = min(closes)
        max_price = max(closes)
        price_range = max(max_price - min_price, 1e-5)

        min_vol = min(volumes)
        max_vol = max(volumes)
        vol_range = max(max_vol - min_vol, 1.0)

        rsi = self.compute_rsi(closes)
        sma = self.compute_sma(closes, window=10)

        feature_matrix = []
        for i in range(len(bars)):
            norm_close = (closes[i] - min_price) / price_range
            norm_vol = (volumes[i] - min_vol) / vol_range
            norm_rsi = rsi[i] / 100.0
            norm_sma = (sma[i] - min_price) / price_range

            feature_matrix.append([norm_close, norm_vol, norm_rsi, norm_sma])

        return feature_matrix, min_price, max_price

