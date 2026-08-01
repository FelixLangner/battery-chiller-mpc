import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class RCObserver:
    """
    Steady-state Kalman / Luenberger observer for each room's RC thermal model.

    Per-room update (called once per 15-min MPC step, Matlab current-state form):

        e[k]   = y[k] - C * x̂[k] - D * u[k]
        x̂[k+1] = A * x̂[k] + B * u[k] + K * e[k]

    where:
        y      — measured room temperature (°C) from the synthetic store
        u_prev — [T_amb, Q_fan, Solar] inputs applied at the previous step
        K      — observer gain vector, shape (n_states,), loaded from mpc_rc_models.json
        D      — feedthrough matrix, shape (1, n_inputs), loaded from mpc_rc_models.json

    State is persisted to a JSON file between process restarts.
    On first run (no state file), call warmup_from_history() before the MPC loop.
    """

    def __init__(self, rc_models: dict, state_file: Path):
        self.rc_models = rc_models
        self.state_file = state_file
        self._state, self._needs_warmup = self._load_or_init()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def needs_warmup(self) -> bool:
        """True if no valid state file was found and warmup_from_history() should be called."""
        return self._needs_warmup

    @property
    def x_state(self) -> dict:
        """Current state estimate {room_id: [x0, x1, ...]}. Feed directly into MPC."""
        return self._state["x_state"]

    @property
    def T_lift_last(self) -> float:
        """Temperature lift (T_amb - T_sup) from the previous step — MPC lag-1 fallback."""
        return self._state.get("T_lift_last", 8.5)

    def update(
        self,
        y_measured: dict,    # {room_id: list[float]}  3 buckets oldest→newest, NaN where missing
        u_prev: dict,        # {room_id: list}    [T_amb, Q_fan, Solar] at previous step
        T_amb: float,
        T_sup: float,
        n_substeps: int = 3,
        disturbance_threshold: float = 0.4,
    ) -> None:
        """
        Apply Kalman predict+correct for every room, then persist state.
        Call this once per 15-min MPC step, before building initial_states.
        """
        for r, model in self.rc_models.items():
            A = np.array(model["A"])
            B = np.array(model["B"])
            C = np.atleast_2d(np.array(model["C"]))
            K = np.array(model.get("K", [[0.0]] * A.shape[0]))
            D = np.atleast_2d(np.array(model.get("D", np.zeros(B.shape[1]))))

            x = np.array(self._state["x_state"][r])
            u = np.array(u_prev[r])

            y_val = y_measured[r]
            y_list = [float(y_val)] if isinstance(y_val, (int, float)) else list(y_val)

            for sub in range(n_substeps):
                y_sub = y_list[sub] if sub < len(y_list) else float("nan")
                if not math.isnan(y_sub):
                    e = y_sub - (C @ x).item() - (D @ u).item()
                    if abs(e) <= disturbance_threshold:
                        x = A @ x + B @ u + K.flatten() * e
                    else:
                        # Unmeasured disturbance: advance physics, snap air state only
                        x = A @ x + B @ u
                        x[0] = y_sub
                        log.info(
                            f"{r} sub-step {sub}: Kalman correction SKIPPED — disturbance "
                            f"|e|={abs(e):.2f}°C > {disturbance_threshold}°C. "
                            f"Air state snapped to measurement, hidden states held open-loop."
                        )
                else:
                    x = A @ x + B @ u

            self._state["x_state"][r] = x.tolist()

        self._state["T_lift_last"] = float(T_amb - T_sup)
        self._state["T_sup_last"] = float(T_sup)
        self._state["u_prev"] = u_prev
        self._state["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def get_u_prev(self, rooms: list, T_amb_fallback: float) -> dict:
        """
        Returns the u_prev stored from the last step.
        On first run (no prior state), returns a neutral vector.
        """
        stored = self._state.get("u_prev")
        if stored and all(r in stored for r in rooms):
            return stored
        return {r: [T_amb_fallback, 0.0, 0.0] for r in rooms}

    def warmup_from_history(self, state_reader, disturbance_threshold: float = 0.4) -> None:
        """
        Initialize state by running the Kalman filter over 24h of historical data.
        Called once on cold start (needs_warmup == True). After this, the state
        file exists and subsequent restarts load directly from it.
        """
        log.info("Cold start detected — running 24h observer warmup from history.")
        history = state_reader.read_history(hours=24)
        n_steps = len(history["T_amb"])

        if n_steps == 0:
            log.warning("No historical data returned — staying with cold-start state.")
            return

        log.info(f"Warming up observer over {n_steps} steps ({n_steps * 5} minutes of history).")

        for r, model in self.rc_models.items():
            A = np.array(model["A"])
            B = np.array(model["B"])
            C = np.atleast_2d(np.array(model["C"]))
            K = np.array(model.get("K", np.zeros(A.shape[0])))
            D = np.atleast_2d(np.array(model.get("D", np.zeros(B.shape[1]))))

            y_arr = history["room_temps"][r]
            Q_arr = history["Q_fan"][r]
            Solar_arr = history["Solar"]

            last_T_amb = 20.0
            last_Q = 0.0
            last_Solar = 0.0

            y0 = next((y_arr[t] for t in range(n_steps) if not np.isnan(y_arr[t])), 22.0)
            for t0 in range(n_steps):
                T0 = history["T_amb"][t0]
                Q0 = Q_arr[t0]
                S0 = Solar_arr[t0]
                if not np.isnan(T0) and not np.isnan(Q0):
                    last_T_amb, last_Q = T0, Q0
                    if not np.isnan(S0):
                        last_Solar = S0
                    break

            x = np.full(A.shape[0], y0)
            n_skipped = 0

            for t in range(n_steps):
                T_amb_t = history["T_amb"][t]
                if np.isnan(T_amb_t):
                    T_amb_t = last_T_amb
                else:
                    last_T_amb = T_amb_t

                Q_t = Q_arr[t]
                if np.isnan(Q_t):
                    Q_t = last_Q
                else:
                    last_Q = Q_t

                Solar_t = Solar_arr[t]
                if np.isnan(Solar_t):
                    Solar_t = last_Solar
                else:
                    last_Solar = Solar_t

                u = np.array([T_amb_t, Q_t, Solar_t])

                y_t = y_arr[t]
                if not np.isnan(y_t):
                    e = y_t - (C @ x).item() - (D @ u).item()
                    if abs(e) <= disturbance_threshold:
                        x = A @ x + B @ u + K.flatten() * e
                    else:
                        x = A @ x + B @ u
                        x[0] = y_t
                        n_skipped += 1
                else:
                    x = A @ x + B @ u

            self._state["x_state"][r] = x.tolist()
            log.info(f"  {r}: final x̂ = {[round(v, 2) for v in x.tolist()]}")
            log.info(
                f"  {r}: Kalman correction skipped on {n_skipped}/{n_steps} steps "
                f"({100 * n_skipped / n_steps:.1f}%) — hidden states ran open-loop those steps."
            )

        valid_T_sup = history["T_sup"][~np.isnan(history["T_sup"])]
        if len(valid_T_sup) > 0:
            self._state["T_sup_last"] = float(valid_T_sup[-1])

        valid_T_amb = history["T_amb"][~np.isnan(history["T_amb"])]
        valid_T_sup_arr = history["T_sup"][~np.isnan(history["T_sup"])]
        if len(valid_T_amb) > 0 and len(valid_T_sup_arr) > 0:
            self._state["T_lift_last"] = float(valid_T_amb[-1] - valid_T_sup_arr[-1])

        self._state["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._needs_warmup = False
        self._save()
        log.info("Observer warmup complete.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_or_init(self) -> tuple[dict, bool]:
        if self.state_file.exists():
            with open(self.state_file) as f:
                data = json.load(f)
            if set(data.get("x_state", {}).keys()) == set(self.rc_models.keys()):
                log.info(f"Loaded observer state from {self.state_file}")
                return data, False
            log.warning("Observer state file has wrong rooms — resetting.")

        log.info("No valid observer state file found — cold start pending warmup.")
        return self._cold_start(), True

    def _cold_start(self) -> dict:
        return {
            "x_state": {r: [22.0] * len(m["A"]) for r, m in self.rc_models.items()},
            "T_lift_last": 8.5,
            "T_sup_last": 14.0,
            "u_prev": None,
            "timestamp": None,
        }

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)
