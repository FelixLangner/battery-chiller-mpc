"""
Python-only (scipy-based) refit of the RC thermal / chiller power / T_sup /
fan coefficients from data/synthetic_store.parquet.

Run scripts/seed_synthetic_history.py first to populate the store this
reads from. Writes to src/battery_ems/controllers/*.json only with --write
(default off).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from battery_ems.emulators.synthetic_building.rc_plant import (
    ROOM_PARAMS,
    RCParams,
    discretize,
)
from battery_ems.emulators.synthetic_building.store import TimeSeriesStore

STORE_PATH = REPO / "data" / "synthetic_store.parquet"
CONTROLLERS_DIR = REPO / "src" / "battery_ems" / "controllers"
FIGURES_DIR = REPO / "figures"
ROOMS = list(ROOM_PARAMS.keys())

DT_SECONDS = 300


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _fit_percent(pred: np.ndarray, obs: np.ndarray) -> float:
    """Same "FitPercent" convention used throughout this codebase's MATLAB
    fitting scripts: 100 * (1 - ||obs-pred|| / ||obs-mean(obs)||)."""
    resid_norm = np.linalg.norm(obs - pred)
    var_norm = np.linalg.norm(obs - np.mean(obs))
    if var_norm == 0:
        return 0.0
    return float(100.0 * (1.0 - resid_norm / var_norm))


# ---------------------------------------------------------------------------
# RC thermal model refit (the actual reason MATLAB was needed)
# ---------------------------------------------------------------------------

def _simulate_room(params: RCParams, T_amb, Q_fan_kw, Solar, T_room0: float) -> np.ndarray:
    model = discretize(params)
    A = np.array(model["A"])
    B = np.array(model["B"])
    x = np.array([T_room0, T_room0])
    T_pred = np.empty(len(T_amb))
    for i in range(len(T_amb)):
        x = A @ x + B @ np.array([T_amb[i], Q_fan_kw[i], Solar[i]])
        T_pred[i] = x[0]
    return T_pred


def fit_room_rc(T_room, T_amb, Q_fan_kw, Solar) -> tuple[RCParams, float, float]:
    """Output-error (multi-step rollout) identification: minimize simulated
    vs. observed T_room over the whole trajectory, log-parameterized so
    Ci/Ce/Rie/Rea stay positive during the search."""
    T_room0 = float(T_room[0])

    def residuals(log_p):
        Ci, Ce, Rie, Rea, Aw = np.exp(log_p)
        params = RCParams(Ci=Ci, Ce=Ce, Rie=Rie, Rea=Rea, Aw=Aw)
        pred = _simulate_room(params, T_amb, Q_fan_kw, Solar, T_room0)
        return pred - T_room

    # Nominal starting guess: mid-range, plausible for a small residential room.
    p0 = np.log([0.45, 20.0, 1.0, 6.5, 1.5])
    result = least_squares(residuals, p0, method="lm", max_nfev=2000)
    Ci, Ce, Rie, Rea, Aw = np.exp(result.x)
    fitted = RCParams(Ci=Ci, Ce=Ce, Rie=Rie, Rea=Rea, Aw=Aw)

    pred = _simulate_room(fitted, T_amb, Q_fan_kw, Solar, T_room0)
    rmse = _rmse(pred, T_room)
    fit_pct = _fit_percent(pred, T_room)
    return fitted, rmse, fit_pct


# ---------------------------------------------------------------------------
# Chiller / T_sup / fan regressions (plain linear regression, as in the
# private repo's own Python retrain_*.py scripts -- no MATLAB involved there
# even in the private repo)
# ---------------------------------------------------------------------------

def fit_chiller_and_tsup(df, rooms) -> dict:
    chiller_on = df["chiller_cmp_hz"].to_numpy() > 5.0
    T_sup = df["T_sup"].to_numpy()
    T_amb = df["EnvTmp"].to_numpy()
    Total_Q = sum(df[f"{r}_Q_fan_w"].to_numpy() / 1000.0 for r in rooms)

    on_intercept = float(np.median(T_sup[chiller_on])) if chiller_on.any() else 8.0

    # OFF-regime AR(1)-with-inputs: T_sup[t+1] ~ 1 + T_sup[t] + EnvTmp[t+1] + Total_Q[t+1],
    # restricted to steps where the chiller was off at both t and t+1 (isolates the
    # passive-warming dynamic from the instant reset-to-constant when it turns on).
    # EnvTmp/Total_Q use the [t+1]-indexed (same row as T_sup[t+1]) values, not [t] --
    # plant.py's step() computes T_amb and Total_Q for the row it's about to produce,
    # then applies them in the same update that turns T_sup[t] into T_sup[t+1]; using
    # the previous row's T_amb/Total_Q instead is a genuine misalignment, confirmed via
    # exact residual test against the ground-truth OFF-regime formula (0.0109 mean
    # abs residual with [t]-indexing vs. exactly 0.0 with [t+1]-indexing).
    off_now = ~chiller_on[:-1]
    off_next = ~chiller_on[1:]
    mask = off_now & off_next
    X = np.column_stack([np.ones(mask.sum()), T_sup[:-1][mask], T_amb[1:][mask], Total_Q[1:][mask]])
    y = T_sup[1:][mask]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    off_rmse = _rmse(pred, y)
    off_fit_pct = _fit_percent(pred, y)

    return {
        "plant_thermal_model_on": {
            "Intercept": round(on_intercept, 4), "T_sup_current": 0.0,
            "EnvTmp": 0.0, "Total_Thermal_Load_kW": 0.0,
        },
        "plant_thermal_model_off": {
            "Intercept": round(float(coef[0]), 5), "T_sup_current": round(float(coef[1]), 5),
            "EnvTmp": round(float(coef[2]), 5), "Total_Thermal_Load_kW": round(float(coef[3]), 5),
        },
        "plant_thermal_model": {
            "Intercept": round(float(coef[0]), 5), "T_sup_current": round(float(coef[1]), 5),
            "Chiller_Command": round(on_intercept - float(coef[0]) - float(coef[1]) * float(np.mean(T_sup)), 3),
            "EnvTmp": round(float(coef[2]), 5), "Total_Thermal_Load_kW": round(float(coef[3]), 5),
        },
        "_diagnostics": {"off_regime_rmse": off_rmse, "off_regime_fit_percent": off_fit_pct,
                          "n_on_samples": int(chiller_on.sum()), "n_off_pairs": int(mask.sum()),
                          "off_regime_obs": y, "off_regime_pred": pred},
    }


def fit_chiller_power(df, rooms) -> dict:
    """
    NOT refit from data: SyntheticBuilding never simulates the chiller's
    electrical power draw (only its thermal effect on T_sup) -- there's no
    P_elec ground truth to regress against. Returns the hand-authored
    coefficients unchanged rather than fabricating a target. Refitting this
    would need extending SyntheticBuilding with a power model first.
    """
    standby_kw = 0.05
    return {
        "chiller_power_model": {"standby_kW": standby_kw, "active_intercept": 0.6, "Temp_Lift": 0.04,
                                 "Delta_Temp_Lift": 0.0, "Total_Thermal_Load": -0.6},
        "chiller_power_model_continuous": {"standby_kW": standby_kw, "active_intercept": 0.1,
                                            "Temp_Lift": 0.02, "Delta_Temp_Lift": 0.0,
                                            "Total_Thermal_Load": -0.5, "Chiller_Command": 1.0},
        "_diagnostics": {"note": "no P_elec ground truth simulated -- power coefficients not refit"},
    }


def fit_fan_slope(df, room: str) -> tuple[float, float, np.ndarray, np.ndarray]:
    T_room = df[f"{room}_temperature"].to_numpy()
    T_sup = df["T_sup"].to_numpy()
    Q_fan_kw = df[f"{room}_Q_fan_w"].to_numpy() / 1000.0
    # Q_fan[i] is computed in plant.py's step() from the PRE-update T_room/T_sup
    # (i.e. row i-1's values) before T_room/T_sup are advanced and stored as row
    # i -- so it must be paired with (T_room[i-1] - T_sup[i-1]), not the same-row
    # (T_room[i] - T_sup[i]). Confirmed via exact residual test against the
    # ground-truth slope: same-row gives mean|resid|=0.134, prev-row gives
    # exactly 0.0 (floating-point-perfect) across all 5 rooms.
    delta_prev = (T_room - T_sup)[:-1]
    q_next = Q_fan_kw[1:]
    is_on = np.abs(q_next) > 1e-6
    if is_on.sum() < 5:
        return -0.2, 0.0, np.array([]), np.array([])
    delta = delta_prev[is_on]
    q = q_next[is_on]
    # Q_fan = slope * delta, through the origin (matches gurobipy_mpc.py's Max_Cooling formula)
    slope = float(np.sum(delta * q) / np.sum(delta * delta))
    pred = slope * delta
    return slope, _rmse(pred, q), delta, q


# ---------------------------------------------------------------------------
# Validation figures
# ---------------------------------------------------------------------------

def _grid_axes(n: int, ncols: int, figsize_per_cell):
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


def _plot_rc_validation(rc_validation: dict, index) -> Path:
    """rc_validation: {room: (T_room_observed, T_room_pred, rmse, fit_pct)}"""
    fig, axes = _grid_axes(len(rc_validation), ncols=3, figsize_per_cell=(5, 3.5))
    for ax, (room, (T_obs, T_pred, rmse, fit_pct)) in zip(axes, rc_validation.items()):
        ax.plot(index, T_obs, linewidth=1.2, alpha=0.85, label="observed")
        ax.plot(index, T_pred, linewidth=1.2, alpha=0.85, linestyle="--", label="predicted (rollout)")
        ax.set_title(f"{room}: RMSE={rmse:.3f}°C, Fit%={fit_pct:.1f}%", fontsize=10)
        ax.set_ylabel("T_room (°C)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("RC thermal model validation: observed vs. predicted room temperature", fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "validation_rc_rooms.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_tsup_validation(y_obs: np.ndarray, y_pred: np.ndarray, rmse: float, fit_pct: float) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_obs, y_pred, s=10, alpha=0.35)
    lo, hi = min(y_obs.min(), y_pred.min()), max(y_obs.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
    ax.set_xlabel("Observed T_sup[t+1] (°C)")
    ax.set_ylabel("Predicted T_sup[t+1] (°C)")
    ax.set_title(f"OFF-regime T_sup regression: RMSE={rmse:.3f}°C, Fit%={fit_pct:.1f}%", fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIGURES_DIR / "validation_tsup_off_regime.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_fan_validation(fan_validation: dict) -> Path:
    """fan_validation: {room: (delta, q, slope, rmse)}"""
    fig, axes = _grid_axes(len(fan_validation), ncols=3, figsize_per_cell=(5, 4))
    for ax, (room, (delta, q, slope, rmse)) in zip(axes, fan_validation.items()):
        if delta.size == 0:
            ax.set_title(f"{room}: no active-fan samples", fontsize=10)
            continue
        ax.scatter(delta, q, s=10, alpha=0.35, label="observed")
        x_line = np.array([delta.min(), delta.max()])
        ax.plot(x_line, slope * x_line, "r--", linewidth=1.5, label=f"fit: slope={slope:.3f}")
        ax.set_title(f"{room}: RMSE={rmse:.4f} kW", fontsize=10)
        ax.set_xlabel("T_room - T_sup (°C)")
        ax.set_ylabel("Q_fan (kW)")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("Fan heat-flow slope validation (through-origin regression)", fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "validation_fan_slopes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main(write: bool) -> None:
    if not STORE_PATH.exists():
        print(f"ERROR: {STORE_PATH} not found -- run scripts/seed_synthetic_history.py first.")
        return
    store = TimeSeriesStore(STORE_PATH)
    df = store._df.dropna(how="all")
    print(f"Loaded {len(df)} rows from {STORE_PATH} ({df.index.min()} -> {df.index.max()})")

    print("\n=== RC thermal models (output-error rollout fit) ===")
    rc_models = {}
    rc_validation = {}
    for room in ROOMS:
        T_room = df[f"{room}_temperature"].to_numpy()
        T_amb = df["EnvTmp"].to_numpy()
        Q_fan_kw = df[f"{room}_Q_fan_w"].to_numpy() / 1000.0
        Solar = df["GlobIrradHoriz"].to_numpy() / 1000.0
        fitted, rmse, fit_pct = fit_room_rc(T_room, T_amb, Q_fan_kw, Solar)
        model = discretize(fitted)
        model["K"] = [0.9, 0.35]  # observer gain not identified here; kept at a plausible default
        rc_models[room] = model
        T_pred = _simulate_room(fitted, T_amb, Q_fan_kw, Solar, float(T_room[0]))
        rc_validation[room] = (T_room, T_pred, rmse, fit_pct)
        print(f"  {room}: RMSE={rmse:.4f}°C  FitPercent={fit_pct:.1f}%  "
              f"(Ci={fitted.Ci:.3f}, Ce={fitted.Ce:.2f}, Rie={fitted.Rie:.3f}, "
              f"Rea={fitted.Rea:.3f}, Aw={fitted.Aw:.3f})")
        if fit_pct < 80.0:
            print(f"    [WARNING] {room}: fit quality below 80% -- inspect before deploying.")

    print("\n=== Chiller T_sup regime-switch model (linear regression) ===")
    plant_coefs = fit_chiller_and_tsup(df, ROOMS)
    diag = plant_coefs.pop("_diagnostics")
    print(f"  ON regime: constant={plant_coefs['plant_thermal_model_on']['Intercept']:.2f}°C "
          f"(n={diag['n_on_samples']} samples)")
    print(f"  OFF regime: RMSE={diag['off_regime_rmse']:.4f}°C  FitPercent={diag['off_regime_fit_percent']:.1f}% "
          f"(n={diag['n_off_pairs']} pairs)")

    print("\n=== Chiller power model ===")
    power_coefs = fit_chiller_power(df, ROOMS)
    diag2 = power_coefs.pop("_diagnostics")
    print(f"  {diag2.get('note', 'regressed from data')}")

    print("\n=== Fan heat-flow slopes (regression through origin) ===")
    fan_coefs = {}
    fan_validation = {}
    for room in ROOMS:
        slope, rmse, delta, q = fit_fan_slope(df, room)
        fan_coefs[room] = {"unified_slope_W_per_C": round(slope * 1000, 2),
                            "unified_slope_kW_per_C": round(slope, 5)}
        fan_validation[room] = (delta, q, slope, rmse)
        print(f"  {room}: slope={slope:.4f} kW/°C  RMSE={rmse:.4f} kW")

    print("\n=== Validation figures ===")
    FIGURES_DIR.mkdir(exist_ok=True)
    rc_path = _plot_rc_validation(rc_validation, df.index)
    print(f"  Saved {rc_path}")
    if diag["n_off_pairs"] > 0:
        tsup_path = _plot_tsup_validation(diag["off_regime_obs"], diag["off_regime_pred"],
                                           diag["off_regime_rmse"], diag["off_regime_fit_percent"])
        print(f"  Saved {tsup_path}")
    fan_path = _plot_fan_validation(fan_validation)
    print(f"  Saved {fan_path}")

    if not write:
        print("\n(dry run -- pass --write to overwrite src/battery_ems/controllers/*.json)")
        return

    plant_out = {**plant_coefs}
    plant_out.update(power_coefs)
    plant_out["_meta"] = {"source": "fit_synthetic_models.py, refit from data/synthetic_store.parquet"}
    with open(CONTROLLERS_DIR / "mpc_plant_power_coefs.json", "w") as f:
        json.dump(plant_out, f, indent=2)
    fan_coefs["_meta"] = {"source": "fit_synthetic_models.py, refit from data/synthetic_store.parquet"}
    with open(CONTROLLERS_DIR / "mpc_unified_fan_coefficients.json", "w") as f:
        json.dump(fan_coefs, f, indent=2)
    with open(CONTROLLERS_DIR / "mpc_rc_models.json", "w") as f:
        json.dump(rc_models, f, indent=2)
    print(f"\nWrote refit coefficients to {CONTROLLERS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Overwrite the deployed controller JSONs.")
    args = parser.parse_args()
    main(args.write)
