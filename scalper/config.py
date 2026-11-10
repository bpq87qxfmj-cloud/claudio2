"""Configuración: mercados candidatos, costes y parámetros de la estrategia."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


@dataclass(frozen=True)
class Market:
    key: str
    name: str
    histdata: str  # código del par en histdata.com (histórico M1, cotización bid)
    yahoo: str  # ticker de Yahoo Finance para datos recientes / señal en vivo
    cost: float  # coste ida y vuelta en unidades de precio: spread + comisión + deslizamiento


# Costes ida y vuelta típicos de una cuenta minorista ECN / micro-futuros.
# Son supuestos: ajústalos a tu bróker con --cost.
MARKETS: dict[str, Market] = {
    "XAUUSD": Market("XAUUSD", "Oro spot (XAU/USD)", "XAUUSD", "GC=F", 0.30),
    "NAS100": Market("NAS100", "Nasdaq-100", "NSXUSD", "NQ=F", 1.00),
    "SPX500": Market("SPX500", "S&P 500", "SPXUSD", "ES=F", 0.50),
}


@dataclass(frozen=True)
class Params:
    """Parámetros fijados a priori (no se optimizan sobre el periodo de test)."""

    horizon: int = 10  # barras de 1 min que dura como máximo una operación
    barrier_mult: float = 1.0  # TP = SL = barrier_mult * sigma_1m * sqrt(horizon) * precio
    vol_span: int = 120  # span del EWM de volatilidad usado para dimensionar barreras
    min_width_cost: float = 4.0  # no operar si la barrera mide menos de N veces el coste

    # walk-forward
    train_months: int = 12
    val_frac: float = 0.15  # cola del tramo de entrenamiento usada para parada temprana y umbral
    first_test: str = "2024-01"
    lockbox_start: str = "2026-05"  # desde aquí: datos que no se miran durante el diseño

    # selección de umbral en validación
    quantiles: tuple[float, ...] = (0.20, 0.10, 0.05, 0.02, 0.01, 0.005)
    min_val_trades: int = 50
    min_val_tstat: float = 1.0

    lgbm: dict = field(
        default_factory=lambda: dict(
            objective="regression",
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=400,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=10.0,
            n_jobs=4,
            verbose=-1,
            random_state=7,
        )
    )


PARAMS = Params()
