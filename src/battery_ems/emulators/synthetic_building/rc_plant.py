"""
Ground-truth 2-state (TiTe) RC thermal plant for one room: ONLY the bare
discrete-time propagation used by the real `mpc_rc_models.json` state-space
format (A, B, C, D), adapted from the propagation math in
`battery_ems.mpc.rc_observer.RCObserver` (x = A @ x + B @ u) but stripped of
its Kalman-correction/persistence machinery, which is observer-specific, not
plant-specific.

Hand-authored, plausible, stable per-room parameters -- NOT fit from real
data. Discretized via zero-order hold at the 5-min physics step, matching
`gurobipy_mpc.build_parametric_mpc`'s BLOCK_SIZE=3 (15-min control) / 5-min
physics resolution.

State x = [T_i (air), T_e (envelope)]. Input u = [T_amb, Q_fan_kW, Solar_kW_m2, 0.0]
-- the 4th input is always literally 0.0 in the deployed MPC constraint
(`gurobipy_mpc.py`'s `u_t = [..., 0.0]`), so B's 4th column is unused by
construction; kept as an all-zero column only to match the JSON shape the
solver code expects.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import cont2discrete

DT_SECONDS = 300  # 5-min physics step, matches gurobipy_mpc.py


@dataclass
class RCParams:
    Ci: float   # kWh/°C, room air + fast furnishings thermal mass
    Ce: float   # kWh/°C, envelope (walls/floor/ceiling) thermal mass
    Rie: float  # °C/kW, air <-> envelope coupling resistance
    Rea: float  # °C/kW, envelope <-> ambient resistance
    Aw: float   # m^2-equivalent solar aperture, gain applied directly to the air node


def continuous_matrices(p: RCParams) -> tuple[np.ndarray, np.ndarray]:
    """Ac, Bc for state x=[Ti,Te], input u=[T_amb, Q_fan, Solar]."""
    Ac = np.array([
        [-1.0 / (p.Ci * p.Rie), 1.0 / (p.Ci * p.Rie)],
        [1.0 / (p.Ce * p.Rie), -(1.0 / (p.Ce * p.Rie) + 1.0 / (p.Ce * p.Rea))],
    ])
    Bc = np.array([
        [0.0, 1.0 / p.Ci, p.Aw / p.Ci],
        [1.0 / (p.Ce * p.Rea), 0.0, 0.0],
    ])
    return Ac, Bc


def discretize(p: RCParams, dt_seconds: float = DT_SECONDS) -> dict:
    """Returns the same {A, B, C, D} shape as mpc_rc_models.json entries."""
    Ac, Bc = continuous_matrices(p)
    Cc = np.array([[1.0, 0.0]])
    Dc = np.zeros((1, 3))
    dt_hours = dt_seconds / 3600.0
    Ad, Bd, Cd, Dd, _ = cont2discrete((Ac, Bc, Cc, Dc), dt=dt_hours, method="zoh")
    # Append the unused 4th input column (always multiplied by a literal 0.0
    # in gurobipy_mpc.py) so the shape matches the real JSON's B (n x 4).
    Bd_full = np.hstack([Bd, np.zeros((Bd.shape[0], 1))])
    return {
        "A": Ad.tolist(),
        "B": Bd_full.tolist(),
        "C": Cd.flatten().tolist(),
        "D": [0.0, 0.0, 0.0, 0.0],
        "structure": "TiTe_synthetic",
    }


class RCPlant:
    """Forward-simulates one room's ground-truth RC dynamics."""

    def __init__(self, params: RCParams, x0: tuple[float, float] = (23.0, 23.0)):
        self.params = params
        Ac, Bc = continuous_matrices(params)
        model = discretize(params)
        self.A = np.array(model["A"])
        self.B3 = np.array(model["B"])[:, :3]  # drop the always-zero 4th column for simulation
        self.C = np.array(model["C"])
        self.x = np.array(x0, dtype=float)

    def step(self, T_amb: float, Q_fan_kw: float, Solar_kw_m2: float) -> float:
        """Advance one 5-min step; Q_fan_kw is <=0 (cooling removes heat from the room air)."""
        u = np.array([T_amb, Q_fan_kw, Solar_kw_m2])
        self.x = self.A @ self.x + self.B3 @ u
        return float(self.C @ self.x)

    @property
    def T_room(self) -> float:
        return float(self.C @ self.x)


# Per-room ground-truth parameters, hand-authored (not fit from real data).
# Room "sizes" loosely mirror the private repo's 5-room layout (small offices
# through a larger living/room area) without reusing any real figures.
ROOM_PARAMS: dict[str, RCParams] = {
    "room_1": RCParams(Ci=0.35, Ce=18.0, Rie=1.0, Rea=7.0, Aw=1.0),
    "room_2": RCParams(Ci=0.35, Ce=18.0, Rie=1.0, Rea=7.0, Aw=1.0),
    "room_3": RCParams(Ci=0.45, Ce=22.0, Rie=0.9, Rea=6.5, Aw=1.3),
    "room_4": RCParams(Ci=0.55, Ce=26.0, Rie=0.8, Rea=6.0, Aw=1.6),
    "room_5": RCParams(Ci=0.65, Ce=30.0, Rie=0.75, Rea=5.5, Aw=2.0),
}
