# battery_chiller_ems (synthetic demo)

A synthetic demonstration of a Model Predictive Control
system for a residential building with PV, battery, and a chiller/fan-coil
cooling system. It minimizes electricity cost subject to thermal comfort,
re-solving a receding horizon every 15 minutes.

**Synthetic data.** For data privacy reasons, this repo contains 
only synthetic data: no real sensor data, no real credentials, no real fitted model
coefficients. Instead, every "measurement" is created by a digital twin
(`emulators/synthetic_building/`) built specifically for this demo. 
The control code itself (the MPC formulation, the Kalman
observer, the control loop, the fallback logic) is real as in the deployed controller.

## System overview

- A central chiller (binary on/off) cools a shared supply water loop.
- One fan coil unit per room (binary on/off, 5 rooms) blows air across the
  water to cool each room individually.
- A cooling + PV + battery MPC: chiller/fan decisions and battery
  charge/discharge share one objective (grid import cost − export revenue)
  and one grid-balance constraint, so the solver co-optimizes comfort,
  cooling cost, and battery arbitrage jointly.
- Supply water temperature, chiller power, and fan heat flow are linear
  regression models; each room's thermal response is a per-room RC
  (resistor-capacitor) state-space model.
- PV/load forecasting uses a simple deterministic clear-sky/diurnal model
  (see "What's simplified" below).
- The MPC is built once and re-solved every step using Gurobi.

## Model formulation

### Indices and sets

$$
r \in \mathcal{R} = \{\text{room}_1,\dots,\text{room}_5\}, \qquad
t \in \{0,\dots,H-1\},\ H = 288 \ \text{(5-min steps, 24h horizon)}, \qquad
k = \lfloor t/B \rfloor,\ B = 3 \ \text{(15-min control block)}
$$

### Horizon structure: binary head / continuous tail
> The model includes 6 binary variables per 5 min time step (5x fan on/off, 1x chiller on/off) which is intractable for real-time operation. Two approaches restore tractability:
>
>1. Control blocking: The model runs at 5 min time steps but the binary control decisions are restricted to 15 min steps
>
>2. Continuous approximation: For the first 2 hours of the prediction horizon, the MPC considers the full MILP. This regime is called the "binary head". Afterward,for hours 2-24, the binary variables are relaxed to continuous variables in [0,1]. This is called the "continuous tail". Note, that the "tail" is just an approximation and never gets executed due to the receding horizon approach.

$$
Z^{\mathrm{chiller}}_k \ \text{(chiller on/off)} \qquad Z^{\mathrm{fan}}_{r,k} \ \text{(per-room fan on/off)}
$$

are:
- **strictly binary** for the first $K_{\mathrm{bin}} = 8$ blocks (2h) — the
  "head". This is the only part of every solve that is ever executed.
- **relaxed to $[0,1]$ continuous** for $k \ge K_{\mathrm{bin}}$ — the
  "tail". This exists purely so the optimizer has *some* model of
  consequences beyond 2h and is re-solved from scratch every 15 minutes with fresh forecasts and
  measured state.

Several physical relationships are modeled differently in the two regions
specifically because of this:
- $T^{\mathrm{sup}}$ uses an *exact* regime-switch (separate ON/OFF fitted
  models, Big-M used to linearize the switch
  $Z^{\mathrm{chiller}} \in \{0,1\}$) in the head, vs. a single
  model with $Z^{\mathrm{chiller}}$ as a continuous linear feature in the
  tail.
- $\dot{Q}^{\mathrm{fan}}$ (heat delivered to a room) uses big-M to linearize
  $Z^{\mathrm{fan}} \cdot \gamma_r \cdot (T^{\mathrm{room}} - T^{\mathrm{sup}})$ in
  the head. For the tail: $\dot{Q}^{\mathrm{fan}} \ge \gamma_r(T^{\mathrm{room}} - T^{\mathrm{sup}})$.

### Objective

$$
\min \quad
\underbrace{\sum_{t=0}^{H-1} \Big( P^{\mathrm{grid,imp}}_t\, c^{\mathrm{buy}}_t - P^{\mathrm{grid,exp}}_t\, c^{\mathrm{sell}}_t \Big)\, \Delta t}_{\text{grid cost (chiller + fans + battery net of PV)}}
+
\underbrace{\sum_{r\in\mathcal{R}}\sum_{t=0}^{H-1} w^{T}_r\, s^{T}_{r,t} + \sum_{t=0}^{H-1} w^{\mathrm{Tsup}}\, s^{\mathrm{Tsup}}_t +\\sum_{r\in\mathcal{R}}\sum_{t\,:\,k(t)\,\ge\,K_{\mathrm{bin}}} w^{\mathrm{fan}}_r\, s^{\mathrm{fan}}_{r,t}}_{\text{comfort and physical-floor slack penalties}}
$$

where $\Delta t = 5/60$ h and $s^{(\cdot)}$ are soft-penalty slacks (a
violation stays feasible, just costed). 

### Key models

**Room thermal dynamics (RC model)**, `mpc_rc_models.json`:

$$
x_{r,t+1} = A_r x_{r,t} + B_r u_{r,t}, \qquad
T^{\mathrm{room}}_{r,t} = C_r x_{r,t}, \qquad
u_{r,t} = \big[T^{\mathrm{amb}}_t,\ \dot Q^{\mathrm{fan}}_{r,t},\ \dot q^{\mathrm{solar}}_t\big]^\top
$$

**Supply temperature dynamics**, `mpc_plant_power_coefs.json`: Illustration for the head/tail split:

$$
T^{\mathrm{sup}}_{t+1} =
\begin{cases}
\theta^{\mathrm{on}}, & Z^{\mathrm{chiller}}_k = 1 \\
\theta^{\mathrm{off}} + \theta^{\mathrm{off}}_{T}T^{\mathrm{sup}}_t + \theta^{\mathrm{off}}_{E}T^{\mathrm{amb}}_t + \theta^{\mathrm{off}}_{Q}\sum_{r\in\mathcal{R}} \dot Q^{\mathrm{fan}}_{r,t}, & Z^{\mathrm{chiller}}_k = 0
\end{cases}
\qquad \text{(head)}
$$

$$
T^{\mathrm{sup}}_{t+1} = \theta^{\mathrm{cont}}_0 + \theta^{\mathrm{cont}}_{T}T^{\mathrm{sup}}_t + \theta^{\mathrm{cont}}_{Z}Z^{\mathrm{chiller}}_k + \theta^{\mathrm{cont}}_{E}T^{\mathrm{amb}}_t + \theta^{\mathrm{cont}}_{Q}\sum_{r\in\mathcal{R}} \dot Q^{\mathrm{fan}}_{r,t}
\qquad \text{(tail)}
$$

(ON is a fitted constant, no state dependence — its ~2.7 min time
constant is far below the 15-min block, so $T^{\mathrm{sup}}$ settles
almost immediately.)

**Battery SOC dynamics**:

$$
SOC_{t+1} = SOC_t + \Big( \eta^{\mathrm{ch}} P^{\mathrm{ch}}_t - \frac{P^{\mathrm{dis}}_t}{\eta^{\mathrm{dis}}} \Big) \Delta t
$$

Fan heat flow and chiller electrical power: a fitted regression, exact/Big-M in the
head, relaxed in the tail — see `gurobipy_mpc.py` and the
JSON files in `controllers/` for the exact constraints and coefficients.

### Inputs for each solve

Forecasts (per step over the horizon): `T_amb`, `Solar`, per-room `T_min` /
`T_max` (comfort schedule), `T_sup_min` (floor), buy `tariffs` (and
optional sell tariffs), `PV_forecast`, `Load_forecast`.

Initial state (measured, at solve time): `T_sup_current`,
`Temp_Lift_historical` (previous step's `T_amb−T_sup`, for the `ΔT_lift`
term), per-room `x_state_current` (from the Kalman observer, not a raw
sensor reading), `Chiller_On_prev`, `SOC_current`.

## Repo layout

| Path | Contents                                                                                                                                                        |
|---|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `src/battery_ems/emulators/synthetic_building/` | The digital-twin simulator: RC thermal plant, chiller/T_sup response, PV/load/battery ground truth, and the time-series store standing in for a sensor database |
| `src/battery_ems/emulators/battery/` | Standalone battery SOC emulator used by the synthetic building                                                                                                  |
| `src/battery_ems/interfaces/` | Read "measurements" / Write "actuations" to the digital twin (or, in real deployment, to InfluxDB and a PLC)                                                    |
| `src/battery_ems/mpc/` | Control loop, forecast providers, Kalman observer, control-write de-duplication logic                                                                           |
| `src/battery_ems/controllers/` | The joint MPC model (`gurobipy_mpc.py`) and its coefficient JSONs                                                                                               |
| `src/scripts/` | Standalone scripts                                                                                                                                              |
| `tests/` | Unit tests (no Gurobi solve required to run them)                                                                                                               |
| `meters_demo.yaml` | Synthetic meter config                                                                                          |
| `.gitlab-ci.yml` | CI: lint + unit tests                                                                                                                                           |

## Getting started

1. Python ≥3.10:
   ```bash
   py -m venv venv
   venv/Scripts/activate       
   pip install -e .
   ```
2. A local Gurobi license is required to solve the MPC. Everything else (InfluxDB, PLC, weather) is
   either synthetic or a free public service (DWD weather via
   `wetterdienst`) -- except the buy tariff, which is real public market
   data (see "Synthetic data" above).

## Running it

```bash
python -m scripts.seed_synthetic_history --days 10   # bootstrap synthetic history
python -m battery_ems.mpc.run_mpc --dry-run           # one solve, saves plan plots to figures/
python -m scripts.run_synthetic_demo --cycles 6       # fast-forward closed control loop
python -m scripts.fit_synthetic_models                # refit coefficients from the seeded history (dry run)
python -m scripts.fit_synthetic_models --write         # ...and deploy them
```

`run_synthetic_demo.py` is the closed MPC loop.

## Testing

```bash
pytest tests/
```

No test builds or solves the Gurobi model, so the full suite runs without a
Gurobi license.

## Models: Ground truth vs. MPC model

`src/battery_ems/controllers/*.json` holds two conceptually distinct kinds
of coefficients:
- The **ground truth** the simulator actually uses to generate "measurement" data lives
  in `emulators/synthetic_building/plant.py` and `rc_plant.py`.

- The **MPC's internal model** lives in the JSON files,
  produced by `scripts/fit_synthetic_models.py` refitting from the
  simulator's own generated history via plain `scipy` regression. These are
  two independent things by design (a fitted model is never a perfect match
  to the system it controls, same as in reality).

`fit_synthetic_models.py` replaces the  MATLAB
`greyest`-based RC identification, which needs a MATLAB + System
Identification Toolbox license with scipy.

## What's simplified relative to the real deployment

- **PV/load forecasting** (`mpc/pv_load_forecast.py`) is a simple
  deterministic clear-sky model. In the real deployed code, it is a live foundation-model
  forecast fed by real meter history. However, this pipeline needs a whole
  real-meter preprocessing chain (aggregation, renaming, gap-filling) that's
  out of scope for this demo.
- **Chiller electrical power draw** is not simulated by the digital twin
  (only its thermal effect on supply temperature is).
- No baseline hysteresis controller, no real-hardware PLC client interfacing the actual building.
