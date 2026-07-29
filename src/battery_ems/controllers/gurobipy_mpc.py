from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

# Per-room comfort-slack penalty (€/°C-equivalent). Room 5's FCU is undersized for its
# thermal mass, so its weight is lowered to trade a little comfort margin for flexibility.
SLACK_T_WEIGHT_DEFAULT = 10.0
SLACK_T_WEIGHT_OVERRIDE = {"room_5": 1.0}

FAN_POWER_KW = 0.05  # assumed FCU electrical draw, not a measured value

# Battery power rating -- capacity/efficiency come from mpc_battery_coefs.json,
# but that file has no power rating, so it's set here directly.
BATTERY_MAX_CHARGE_KW = 4.5
BATTERY_MAX_DISCHARGE_KW = 4.5

BATTERY_FEED_IN_TARIFF_EUR_KWH = 0.08


SOC_LOW_FRAC = 0.10
SOC_HIGH_FRAC = 0.90
SOC_TERMINAL_MIN_FRAC = 0.50
SLACK_SOC_WEIGHT_EUR_KWH = 1.0  # €/kWh-step, deliberately firm relative to the tariff scale above

# Small cost on |P_charge[t]-P_charge[t-1]| and |P_discharge[t]-P_discharge[t-1]| --
# breaks ties among equally-priced schedules toward smooth ones. Whenever a stretch of
# time has flat/near-flat marginal economics (PV exceeds load the whole time, so
# charging now vs. later costs the same), the un-penalized LP has zero preference
# between a gentle ramp and an abrupt on/off burst
BATTERY_RAMP_COST_EUR_PER_KW = 0.01

T_ROOM_LB, T_ROOM_UB = 18.0, 28.0
T_SUP_LB, T_SUP_UB = 5.0, 40.0

# Physical envelope for outdoor temperature, used only to size the T_sup regime-switch
# Big-M below (see _add_cooling_dynamics)
ENVTMP_BIGM_LB, ENVTMP_BIGM_UB = -15.0, 45.0

# Slack to allow T_sup > T_room
SLACK_CROSS_WEIGHT_DEFAULT = 0.01
SLACK_CROSS_WEIGHT_OVERRIDE = {"room_5": 1.0}

# If consecutive solves fail, hold the previous solve's plan for this many
# 15-min blocks before falling back to a hard setpoint. Must stay < BINARY_BLOCKS
# so the holdover commands pulled from a previous solve are true binaries.
HOLDOVER_BLOCKS = 4


def _add_cooling_params(m, horizon_steps, rooms, rc_sys):
    """Forecast/initial-state injector vars for the cooling side (fixed lb=ub=0.0 at
    build time, mutated in step_mpc every solve)."""
    return {
        'T_amb': m.addVars(horizon_steps, lb=0.0, ub=0.0, name="Param_T_amb"),
        'Solar': m.addVars(horizon_steps, lb=0.0, ub=0.0, name="Param_Solar"),
        # Room-individual (see forecast_provider._comfort_schedule's per-room override)
        'T_min': {r: m.addVars(horizon_steps, lb=0.0, ub=0.0, name=f"Param_T_min_{r}") for r in rooms},
        'T_max': {r: m.addVars(horizon_steps, lb=0.0, ub=0.0, name=f"Param_T_max_{r}") for r in rooms},
        'T_sup_min': m.addVars(horizon_steps, lb=0.0, ub=0.0, name="P_T_sup_min"),
        'Init_T_sup': m.addVar(lb=0.0, ub=0.0, name="Init_T_sup"),
        'Init_Lift_Historical': m.addVar(lb=0.0, ub=0.0, name="Init_Lift_Hist"),
        'Init_x_state': {r: m.addVars(len(rc_sys[r]['A']), lb=0.0, ub=0.0, name=f"Init_x_{r}") for r in rooms},
        # Real chiller state just before t=0 -- needed by the hysteresis gate (an
        # OFF->ON transition needs T_sup already warm; staying ON does not).
        'Init_Chiller_On': m.addVar(lb=0.0, ub=0.0, name="Init_Chiller_On"),
    }


def _add_battery_params(m, horizon_steps):
    """Same parametric pattern as _add_cooling_params, for the battery/PV side."""
    return {
        'Init_SOC': m.addVar(lb=0.0, ub=0.0, name="Init_SOC"),
        'PV_forecast': m.addVars(horizon_steps, lb=0.0, ub=0.0, name="Param_PV_forecast"),
        'Load_forecast': m.addVars(horizon_steps, lb=0.0, ub=0.0, name="Param_Load_forecast"),
    }


def _add_decision_vars(m, horizon_steps, rooms):
    """Chiller/fan commands: strict binary for the first 2h (executed head), relaxed
    continuous [0,1] beyond that (terminal-cost approximation only)."""
    BLOCK_SIZE = 3  # 3 x 5-min physics steps = 15-min control resolution
    horizon_blocks = horizon_steps // BLOCK_SIZE
    BINARY_BLOCKS = 8  # first 2h

    Z_AC_15 = {}
    Z_fan_15 = {r: {} for r in rooms}
    for k in range(horizon_blocks):
        if k < BINARY_BLOCKS:
            Z_AC_15[k] = m.addVar(vtype=GRB.BINARY, name=f"Z_AC_bin_{k}")
            for r in rooms:
                Z_fan_15[r][k] = m.addVar(vtype=GRB.BINARY, name=f"Z_fan_bin_{r}_{k}")
        else:
            Z_AC_15[k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"Z_AC_cont_{k}")
            for r in rooms:
                Z_fan_15[r][k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"Z_fan_cont_{r}_{k}")

    return Z_AC_15, Z_fan_15, BLOCK_SIZE, horizon_blocks, BINARY_BLOCKS


def _add_cooling_state_vars(m, horizon_steps, rooms, rc_sys, fans, Z_AC_15, Z_fan_15):
    """T_sup/P_elec/room states, plus the McCormick box for each room's Q_fan."""
    vars_dict = {
        'T_sup': m.addVars(horizon_steps + 1, lb=T_SUP_LB, ub=T_SUP_UB, name="T_sup"),
        'P_elec': m.addVars(horizon_steps, lb=0.0, ub=4.0, name="P_elec"),
        'Temp_Lift': m.addVars(horizon_steps, lb=-40.0, ub=40.0, name="Temp_Lift"),
        'T_room': {}, 'x_state': {}, 'Q_fan': {}, 'Slack_T': {},
        'Z_AC_out': Z_AC_15, 'Z_fan_out': Z_fan_15,
        'Slack_T_sup': m.addVars(horizon_steps, lb=0.0, name="Slack_T_sup"),
    }

    # McCormick box [b_L, b_U] for Q_fan = Z_fan * Max_Cooling -- exact at binary Z_fan,
    Q_FAN_BOUNDS = {
        r: (-abs(fans[r]['unified_slope_kW_per_C']) * (T_ROOM_UB - T_SUP_LB),
            abs(fans[r]['unified_slope_kW_per_C']) * (T_SUP_UB - T_ROOM_LB))
        for r in rooms
    }

    for r in rooms:
        nx = len(rc_sys[r]['A'])
        vars_dict['x_state'][r] = m.addVars(nx, horizon_steps + 1, lb=-GRB.INFINITY, name=f"x_{r}")
        vars_dict['T_room'][r] = m.addVars(horizon_steps, lb=T_ROOM_LB, ub=T_ROOM_UB, name=f"T_room_{r}")
        b_L, b_U = Q_FAN_BOUNDS[r]
        vars_dict['Q_fan'][r] = m.addVars(horizon_steps, lb=b_L, ub=b_U, name=f"Q_fan_{r}")
        vars_dict['Slack_T'][r] = m.addVars(horizon_steps, lb=0.0, ub=50.0, name=f"Slack_T_{r}")

    return vars_dict, Q_FAN_BOUNDS


def _add_battery_state_vars(m, horizon_steps, capacity_kwh, max_charge_kw, max_discharge_kw):
    """SOC + charge/discharge + grid import/export No binary
    forcing charge/discharge mutual exclusivity either: with feed-in <= import tariff
    always, doing both at once only wastes round-trip efficiency for zero benefit."""
    return {
        'SOC': m.addVars(horizon_steps + 1, lb=0.0, ub=capacity_kwh, name="SOC"),
        'P_charge': m.addVars(horizon_steps, lb=0.0, ub=max_charge_kw, name="P_charge"),
        'P_discharge': m.addVars(horizon_steps, lb=0.0, ub=max_discharge_kw, name="P_discharge"),
        # Unbounded -- no site fuse/breaker limit modeled yet.
        'P_grid_import': m.addVars(horizon_steps, lb=0.0, name="P_grid_import"),
        'P_grid_export': m.addVars(horizon_steps, lb=0.0, name="P_grid_export"),
        # Soft SOC-band violation amounts
        'Slack_SOC_Low': m.addVars(horizon_steps + 1, lb=0.0, name="Slack_SOC_Low"),
        'Slack_SOC_High': m.addVars(horizon_steps + 1, lb=0.0, name="Slack_SOC_High"),
    }


def _add_initial_conditions(m, vars_dict, params, rc_sys, rooms):
    m.addConstr(vars_dict['T_sup'][0] == params['Init_T_sup'])
    m.addConstr(vars_dict['SOC'][0] == params['Init_SOC'])
    for r in rooms:
        for i in range(len(rc_sys[r]['A'])):
            m.addConstr(vars_dict['x_state'][r][i, 0] == params['Init_x_state'][r][i])


def _add_chiller_hysteresis_constraints(m, vars_dict, params, Z_AC_15, BINARY_BLOCKS, BLOCK_SIZE):
    """Binary-head-only gate approximating real chiller hysteresis: an OFF->ON
    transition needs T_sup already above T_SUP_ON_THRESHOLD; staying ON is
    unrestricted at the top but forced off below T_SUP_OFF_THRESHOLD regardless of
    history. Not applied in the continuous tail -- there Z_AC is a fractional
    value-function approximation, not a real state, and gating it there creates a
    spurious sawtooth via the T_sup feedback loop with no effect on what's executed.

    turn_on[k] is the standard max(0, x) hinge for (Z_AC[k]-prev_z): uncosted in the
    objective, so the solver always drives it to exactly 1 on a genuine OFF->ON
    transition and 0 otherwise -- no extra binaries needed.
    """
    M_TSUP_ON = 20.0
    T_SUP_ON_THRESHOLD = 19.0
    M_TSUP_OFF = 12.0
    T_SUP_OFF_THRESHOLD = 10.0

    Z_AC_prev = None
    for k in range(BINARY_BLOCKS):
        t0 = k * BLOCK_SIZE
        prev_z = Z_AC_prev if Z_AC_prev is not None else params['Init_Chiller_On']
        turn_on = m.addVar(lb=0.0, ub=1.0, name=f"Chiller_TurnOn_{k}")
        m.addConstr(turn_on >= Z_AC_15[k] - prev_z, name=f"Chiller_TurnOn_Def_{k}")
        m.addConstr(
            vars_dict['T_sup'][t0] >= T_SUP_ON_THRESHOLD - M_TSUP_ON * (1 - turn_on),
            name=f"Chiller_On_Threshold_{k}"
        )
        for i in range(BLOCK_SIZE):
            t = t0 + i
            m.addConstr(
                vars_dict['T_sup'][t] >= T_SUP_OFF_THRESHOLD - M_TSUP_OFF * (1 - Z_AC_15[k]),
                name=f"Chiller_Off_Threshold_{k}_{i}"
            )
        Z_AC_prev = Z_AC_15[k]


def _add_cooling_dynamics(m, vars_dict, params, config, Z_AC_15, Z_fan_15, Q_FAN_BOUNDS,
                           rooms, horizon_steps, BLOCK_SIZE, BINARY_BLOCKS):
    """Per-5-min-step T_sup/power/room/fan physics. Everything the battery doesn't touch."""
    plant = config['plant_physics']['plant_thermal_model']  # tail-only: blended, Z_AC as linear feature
    plant_on = config['plant_physics']['plant_thermal_model_on']  # head-only: exact regime, Z_AC=1
    plant_off = config['plant_physics']['plant_thermal_model_off']  # head-only: exact regime, Z_AC=0
    power_bin = config['plant_physics']['chiller_power_model']
    power_cont = config['plant_physics']['chiller_power_model_continuous']
    fans = config['fan_physics']
    rc_sys = config['rc_models']

    M_POWER = 4.0  # max physical electrical kW of the chiller, with margin over any observed peak
    Q_LOAD_CAP_LB = sum(Q_FAN_BOUNDS[r][0] for r in rooms)  # true worst-case total cooling

    # Big-M for the head's exact ON/OFF regime-switch, sized from the fitted
    # coefficients and variable bounds (tightest M that still guarantees correctness).
    _d_a = plant_on['Intercept'] - plant_off['Intercept']
    _d_t = plant_on['T_sup_current'] - plant_off['T_sup_current']
    _d_e = plant_on['EnvTmp'] - plant_off['EnvTmp']
    _d_q = plant_on['Total_Thermal_Load_kW'] - plant_off['Total_Thermal_Load_kW']
    _t_lo, _t_hi = sorted([_d_t * T_SUP_LB, _d_t * T_SUP_UB])
    _e_lo, _e_hi = sorted([_d_e * ENVTMP_BIGM_LB, _d_e * ENVTMP_BIGM_UB])
    _q_lo, _q_hi = sorted([_d_q * Q_LOAD_CAP_LB, _d_q * 0.0])
    _diff_max = _d_a + _t_hi + _e_hi + _q_hi
    _diff_min = _d_a + _t_lo + _e_lo + _q_lo
    M_TSUP = 1.1 * max(abs(_diff_max), abs(_diff_min))

    # Only exists for continuous-tail (room, t) pairs -- see the k>=BINARY_BLOCKS branch
    # below. Populated into vars_dict directly so update_objective can penalize it.
    vars_dict['Slack_Cross'] = {r: {} for r in rooms}

    for t in range(horizon_steps):
        k = t // BLOCK_SIZE

        # Q_fan is bounded <=0 for every room in both head and tail (see the per-room
        # loop below), so this sum is already <=0 by construction -- no separate capped
        # auxiliary needed (a one-sided capped auxiliary previously let the solver fake
        # an arbitrarily more negative load whenever that helped).
        Total_Q_Load = gp.quicksum(vars_dict['Q_fan'][r][t] for r in rooms)

        m.addConstr(vars_dict['Temp_Lift'][t] == params['T_amb'][t] - vars_dict['T_sup'][t])

        if k < BINARY_BLOCKS:
            # Binary head: exact ON/OFF regime-switch (Z_AC_15[k] is a true 0/1 here, so
            # this Big-M selection has zero error).
            on_expr = (plant_on['Intercept'] +
                       plant_on['T_sup_current'] * vars_dict['T_sup'][t] +
                       plant_on['EnvTmp'] * params['T_amb'][t] +
                       plant_on['Total_Thermal_Load_kW'] * Total_Q_Load)
            off_expr = (plant_off['Intercept'] +
                        plant_off['T_sup_current'] * vars_dict['T_sup'][t] +
                        plant_off['EnvTmp'] * params['T_amb'][t] +
                        plant_off['Total_Thermal_Load_kW'] * Total_Q_Load)
            m.addConstr(vars_dict['T_sup'][t + 1] <= on_expr + M_TSUP * (1 - Z_AC_15[k]), name=f"Tsup_On_Hi_{t}")
            m.addConstr(vars_dict['T_sup'][t + 1] >= on_expr - M_TSUP * (1 - Z_AC_15[k]), name=f"Tsup_On_Lo_{t}")
            m.addConstr(vars_dict['T_sup'][t + 1] <= off_expr + M_TSUP * Z_AC_15[k], name=f"Tsup_Off_Hi_{t}")
            m.addConstr(vars_dict['T_sup'][t + 1] >= off_expr - M_TSUP * Z_AC_15[k], name=f"Tsup_Off_Lo_{t}")
        else:
            # Continuous tail: single blended model, Z_AC as a linear feature (value-
            # function approximation only -- never executed).
            m.addConstr(
                vars_dict['T_sup'][t + 1] == plant['Intercept'] +
                plant['T_sup_current'] * vars_dict['T_sup'][t] +
                plant['Chiller_Command'] * Z_AC_15[k] +
                plant['Total_Thermal_Load_kW'] * Total_Q_Load +
                plant['EnvTmp'] * params['T_amb'][t],
                name=f"Tsup_Tail_{t}"
            )
        m.addConstr(
            vars_dict['T_sup'][t] + vars_dict['Slack_T_sup'][t] >= params['T_sup_min'][t],
            name=f"T_sup_Min_Bound_{t}"
        )

        lift_lag1 = vars_dict['Temp_Lift'][t - 1] if t > 0 else params['Init_Lift_Historical']
        delta_temp_lift = vars_dict['Temp_Lift'][t] - lift_lag1

        if k < BINARY_BLOCKS:
            # Binary head: active-only power model, standard Big-M mapping (exact
            # since Z_AC is pure binary here).
            P_active = (power_bin['active_intercept'] +
                        power_bin['Temp_Lift'] * vars_dict['Temp_Lift'][t] +
                        power_bin['Delta_Temp_Lift'] * delta_temp_lift +
                        power_bin['Total_Thermal_Load'] * Total_Q_Load)
            m.addConstr(vars_dict['P_elec'][t] >= power_bin["standby_kW"], name=f"P_Standby_{t}")
            m.addConstr(
                vars_dict['P_elec'][t] >= P_active - M_POWER * (1 - Z_AC_15[k]),
                name=f"P_Active_Standard_{t}"
            )
        else:
            # Continuous tail: linear model with Z_AC as a feature.
            P_reg = (power_cont['active_intercept'] +
                     power_cont['Temp_Lift'] * vars_dict['Temp_Lift'][t] +
                     power_cont['Delta_Temp_Lift'] * delta_temp_lift +
                     power_cont['Total_Thermal_Load'] * Total_Q_Load +
                     power_cont['Chiller_Command'] * Z_AC_15[k])
            m.addConstr(vars_dict['P_elec'][t] >= P_reg, name=f"P_Reg_Floor_{t}")
            m.addConstr(vars_dict['P_elec'][t] >= power_bin["standby_kW"], name=f"P_Standby_Cont_{t}")

        for r in rooms:
            m.addConstr(
                vars_dict['T_room'][r][t] - vars_dict['Slack_T'][r][t] <= params['T_max'][r][t],
                name=f"Comfort_Max_{r}_{t}"
            )
            m.addConstr(
                vars_dict['T_room'][r][t] + vars_dict['Slack_T'][r][t] >= params['T_min'][r][t],
                name=f"Comfort_Min_{r}_{t}"
            )

            A = np.atleast_2d(rc_sys[r]['A'])
            B = np.atleast_2d(rc_sys[r]['B'])
            C = np.atleast_2d(rc_sys[r]['C'])
            slope = fans[r]['unified_slope_kW_per_C']
            Max_Cooling = slope * (vars_dict['T_room'][r][t] - vars_dict['T_sup'][t])

            if k < BINARY_BLOCKS:
                # Binary head: McCormick for w=a*b, a=Z_fan[0,1], b=Max_Cooling[b_L,b_U].
                # Exact at Zf in {0,1} regardless of box tightness -- this is what
                # actually executes.
                b_L, b_U = Q_FAN_BOUNDS[r]
                Zf = Z_fan_15[r][k]
                vars_dict['Q_fan'][r][t].ub = 0.0
                m.addConstr(vars_dict['Q_fan'][r][t] >= Zf * b_L, name=f"Fan_McC_Lo1_{r}_{t}")
                m.addConstr(vars_dict['Q_fan'][r][t] >= Max_Cooling - b_U * (1 - Zf), name=f"Fan_McC_Lo2_{r}_{t}")
                m.addConstr(vars_dict['Q_fan'][r][t] <= Max_Cooling - b_L * (1 - Zf), name=f"Fan_McC_Hi1_{r}_{t}")
                m.addConstr(vars_dict['Q_fan'][r][t] <= Zf * b_U, name=f"Fan_McC_Hi2_{r}_{t}")
            else:
                # Continuous tail: tie Q_fan directly to physics (never positive,
                # never colder than what Max_Cooling genuinely allows), instead of a
                # McCormick relaxation whose slack at fractional Zf is prone to
                # hallucinated tail cooling.
                vars_dict['Q_fan'][r][t].ub = 0.0
                slack_cross = m.addVar(lb=0.0, name=f"Slack_Cross_{r}_{t}")
                vars_dict['Slack_Cross'][r][t] = slack_cross
                m.addConstr(vars_dict['Q_fan'][r][t] >= Max_Cooling - slack_cross,
                            name=f"Q_fan_Physical_Lo_{r}_{t}")

            u_t = [params['T_amb'][t], vars_dict['Q_fan'][r][t], params['Solar'][t], 0.0]
            for i in range(A.shape[0]):
                m.addConstr(vars_dict['x_state'][r][i, t + 1] ==
                            gp.quicksum(A[i, j] * vars_dict['x_state'][r][j, t] for j in range(A.shape[1])) +
                            gp.quicksum(B[i, j] * u_t[j] for j in range(B.shape[1])))
            m.addConstr(vars_dict['T_room'][r][t] == gp.quicksum(
                C[0, j] * vars_dict['x_state'][r][j, t] for j in range(C.shape[1])))


def _add_battery_dynamics(m, vars_dict, params, Z_fan_15, rooms, horizon_steps, BLOCK_SIZE,
                           eta_c, eta_d, dt_h):
    """SOC update + point-of-common-coupling (grid) balance."""
    for t in range(horizon_steps):
        k = t // BLOCK_SIZE
        Fan_Power_Total = gp.quicksum(FAN_POWER_KW * Z_fan_15[r][k] for r in rooms)

        m.addConstr(
            vars_dict['SOC'][t + 1] == vars_dict['SOC'][t]
            + (eta_c * vars_dict['P_charge'][t] - vars_dict['P_discharge'][t] / eta_d) * dt_h,
            name=f"SOC_Dynamics_{t}"
        )
        m.addConstr(
            vars_dict['P_grid_import'][t] - vars_dict['P_grid_export'][t] ==
            vars_dict['P_elec'][t] + Fan_Power_Total
            + vars_dict['P_charge'][t] - vars_dict['P_discharge'][t]
            + params['Load_forecast'][t] - params['PV_forecast'][t],
            name=f"Grid_Balance_{t}"
        )
        # Physical export cap: a grid-tie connection can only push out what's actually
        # generated/discharged on-site, never grid power just bought.
        m.addConstr(
            vars_dict['P_grid_export'][t] <= params['PV_forecast'][t] + vars_dict['P_discharge'][t],
            name=f"Export_Physical_Cap_{t}"
        )


def _add_soc_band_constraints(m, vars_dict, horizon_steps, capacity_kwh):
    """Soft SOC band at every step, with a higher 50% floor on the terminal
    SOC -- slack-penalized in update_objective, not a hard bound, so a genuinely
    necessary excursion stays feasible rather than blowing up the solve."""
    for t in range(horizon_steps + 1):
        low_frac = SOC_TERMINAL_MIN_FRAC if t == horizon_steps else SOC_LOW_FRAC
        m.addConstr(
            vars_dict['SOC'][t] + vars_dict['Slack_SOC_Low'][t] >= low_frac * capacity_kwh,
            name=f"SOC_Low_{t}"
        )
        m.addConstr(
            vars_dict['SOC'][t] - vars_dict['Slack_SOC_High'][t] <= SOC_HIGH_FRAC * capacity_kwh,
            name=f"SOC_High_{t}"
        )


def _add_battery_ramp_vars(m, vars_dict, horizon_steps):
    """|ΔP_charge|/|ΔP_discharge| between consecutive 5-min steps, via the standard
    two-sided linear hinge -- penalized in update_objective (see
    BATTERY_RAMP_COST_EUR_PER_KW)."""
    ramp_charge = m.addVars(horizon_steps - 1, lb=0.0, name="Ramp_Charge")
    ramp_discharge = m.addVars(horizon_steps - 1, lb=0.0, name="Ramp_Discharge")
    for t in range(1, horizon_steps):
        i = t - 1
        m.addConstr(ramp_charge[i] >= vars_dict['P_charge'][t] - vars_dict['P_charge'][t - 1],
                    name=f"Ramp_Charge_Up_{t}")
        m.addConstr(ramp_charge[i] >= vars_dict['P_charge'][t - 1] - vars_dict['P_charge'][t],
                    name=f"Ramp_Charge_Down_{t}")
        m.addConstr(ramp_discharge[i] >= vars_dict['P_discharge'][t] - vars_dict['P_discharge'][t - 1],
                    name=f"Ramp_Discharge_Up_{t}")
        m.addConstr(ramp_discharge[i] >= vars_dict['P_discharge'][t - 1] - vars_dict['P_discharge'][t],
                    name=f"Ramp_Discharge_Down_{t}")
    return {'Ramp_Charge': ramp_charge, 'Ramp_Discharge': ramp_discharge}


def build_parametric_mpc(horizon_steps, config):
    print("\n--- Compiling Joint Cooling+PV+Battery Parametric MPC Matrix ---")
    m = gp.Model("Parametric_Joint_MPC")

    m.setParam('OutputFlag', 0)
    m.setParam("MIPGap", 0.005)
    m.setParam("MIPGapAbs", 0.005)
    m.setParam("MIPFocus", 1)
    m.setParam("Heuristics", 0.3)
    m.setParam("Presolve", 2)
    m.setParam("TimeLimit", 120)

    rc_sys = config['rc_models']
    rooms = list(rc_sys.keys())
    fans = config['fan_physics']
    battery = config['battery_physics']  # power_offset_w is a sensor-bias fit artifact, unused here

    dt_h = 5.0 / 60.0  # hours per 5-min physics step (also used in update_objective)

    params = _add_cooling_params(m, horizon_steps, rooms, rc_sys)
    params.update(_add_battery_params(m, horizon_steps))

    Z_AC_15, Z_fan_15, BLOCK_SIZE, horizon_blocks, BINARY_BLOCKS = _add_decision_vars(m, horizon_steps, rooms)

    assert HOLDOVER_BLOCKS < BINARY_BLOCKS, (
        f"HOLDOVER_BLOCKS ({HOLDOVER_BLOCKS}) must stay within the binary head "
        f"(BINARY_BLOCKS={BINARY_BLOCKS}) so the holdover commands extracted from a "
        f"previous solve are true binaries, not rounded continuous-relaxation values."
    )

    vars_dict, Q_FAN_BOUNDS = _add_cooling_state_vars(m, horizon_steps, rooms, rc_sys, fans, Z_AC_15, Z_fan_15)
    vars_dict.update(_add_battery_state_vars(
        m, horizon_steps, battery['capacity_kwh'], BATTERY_MAX_CHARGE_KW, BATTERY_MAX_DISCHARGE_KW
    ))

    _add_initial_conditions(m, vars_dict, params, rc_sys, rooms)
    _add_chiller_hysteresis_constraints(m, vars_dict, params, Z_AC_15, BINARY_BLOCKS, BLOCK_SIZE)
    _add_cooling_dynamics(m, vars_dict, params, config, Z_AC_15, Z_fan_15, Q_FAN_BOUNDS,
                           rooms, horizon_steps, BLOCK_SIZE, BINARY_BLOCKS)
    _add_battery_dynamics(m, vars_dict, params, Z_fan_15, rooms, horizon_steps, BLOCK_SIZE,
                           battery['efficiency_charge'], battery['efficiency_discharge'], dt_h)
    _add_soc_band_constraints(m, vars_dict, horizon_steps, battery['capacity_kwh'])
    vars_dict.update(_add_battery_ramp_vars(m, vars_dict, horizon_steps))

    m.update()
    print("Static Parametric Matrix Compiled.")
    return m, params, vars_dict, BINARY_BLOCKS


def update_objective(m, vars_dict, new_forecasts, rooms, horizon_steps, BLOCK_SIZE=3):
    dt_h = 5.0 / 60.0
    obj = gp.LinExpr()

    sell_tariffs = new_forecasts.get("sell_tariffs")
    for t in range(horizon_steps):
        tariff = new_forecasts["tariffs"][t]
        sell_price = sell_tariffs[t] if sell_tariffs is not None else BATTERY_FEED_IN_TARIFF_EUR_KWH
        obj += vars_dict["P_grid_import"][t] * tariff * dt_h
        obj -= vars_dict["P_grid_export"][t] * sell_price * dt_h

    for r in rooms:
        slack_t_weight = SLACK_T_WEIGHT_OVERRIDE.get(r, SLACK_T_WEIGHT_DEFAULT)
        for t in range(horizon_steps):
            obj += vars_dict["Slack_T"][r][t] * slack_t_weight

    for t in range(horizon_steps):
        obj += vars_dict["Slack_T_sup"][t] * 10.0

    # Soft SOC band penalty (10-90%, 50% terminal floor -- see
    # _add_soc_band_constraints and SLACK_SOC_WEIGHT_EUR_KWH).
    for t in range(horizon_steps + 1):
        obj += (vars_dict["Slack_SOC_Low"][t] + vars_dict["Slack_SOC_High"][t]) * SLACK_SOC_WEIGHT_EUR_KWH

    # Ramp smoothing -- see BATTERY_RAMP_COST_EUR_PER_KW.
    for t in range(horizon_steps - 1):
        obj += (vars_dict["Ramp_Charge"][t] + vars_dict["Ramp_Discharge"][t]) * BATTERY_RAMP_COST_EUR_PER_KW

    # Continuous-tail Q_fan's physical lower-bound slack (see SLACK_CROSS_WEIGHT_DEFAULT/
    # SLACK_CROSS_WEIGHT_OVERRIDE) -- only exists for (room, t) pairs in the tail.
    for r in rooms:
        slack_cross_weight = SLACK_CROSS_WEIGHT_OVERRIDE.get(r, SLACK_CROSS_WEIGHT_DEFAULT)
        for t in vars_dict["Slack_Cross"][r]:
            obj += vars_dict["Slack_Cross"][r][t] * slack_cross_weight

    m.setObjective(obj, GRB.MINIMIZE)


def _preactivate_lagged_fans(vars_dict, rooms, chiller_cmd, fan_cmds):
    """Z_fan block-granularity lockout mitigation. Z_fan is one binary per 15-min
    control block, but T_sup/Max_Cooling move per 5-min substep. If the chiller is
    on this block but T_sup(t=0) hasn't yet dropped below a room's temperature,
    that room's Z_fan[k=0] can come back 0 for the whole block even though real
    cooling capacity appears mid-block once T_sup catches up -- pre-activate now
    instead of waiting an extra ~15 min for something already known to be coming
    (the solver's own next-block decision, Z_fan_out[r][1], reveals it). Only ever
    flips a fan 0->1; never overrides a fan the solver chose to leave off.
    """
    if chiller_cmd != 1:
        return fan_cmds
    t_sup_now = vars_dict['T_sup'][0].X
    for r in rooms:
        if (fan_cmds[r] == 0 and vars_dict['T_room'][r][0].X < t_sup_now and
                round(vars_dict['Z_fan_out'][r][1].X) == 1):
            fan_cmds[r] = 1
    return fan_cmds


def _extract_optimal_action(vars_dict, rooms, holdover_blocks):
    chiller_cmd = round(vars_dict['Z_AC_out'][0].X)
    fan_cmds = {r: round(vars_dict['Z_fan_out'][r][0].X) for r in rooms}
    fan_cmds = _preactivate_lagged_fans(vars_dict, rooms, chiller_cmd, fan_cmds)
    return {
        'Chiller_Command': chiller_cmd,
        'Fan_Commands': fan_cmds,
        # Battery command nets to a signed power (+discharge, -charge), not
        # rounded like Chiller/Fan -- it's a genuinely continuous decision.
        'Battery_Power_kW': vars_dict['P_discharge'][0].X - vars_dict['P_charge'][0].X,
        # Blocks k=1..holdover_blocks stored so run_mpc.py can coast through
        # up to holdover_blocks consecutive solve failures.
        'Holdover_Chiller_Commands': [
            round(vars_dict['Z_AC_out'][k].X) for k in range(1, holdover_blocks + 1)
        ],
        'Holdover_Fan_Commands': [
            {r: round(vars_dict['Z_fan_out'][r][k].X) for r in rooms}
            for k in range(1, holdover_blocks + 1)
        ],
        'Holdover_Battery_Power_kW': [
            vars_dict['P_discharge'][k * 3].X - vars_dict['P_charge'][k * 3].X
            for k in range(1, holdover_blocks + 1)
        ],
    }


def step_mpc(m, params, vars_dict, new_initial_states, new_forecasts, horizon_steps,
             holdover_blocks=HOLDOVER_BLOCKS):
    rooms = list(params['T_min'].keys())

    # 1. Forecasts
    for t in range(horizon_steps):
        params['T_amb'][t].lb = params['T_amb'][t].ub = new_forecasts['T_amb'][t]
        params['Solar'][t].lb = params['Solar'][t].ub = new_forecasts['Solar'][t]
        for r in rooms:
            params['T_min'][r][t].lb = params['T_min'][r][t].ub = new_forecasts['T_min'][r][t]
            params['T_max'][r][t].lb = params['T_max'][r][t].ub = new_forecasts['T_max'][r][t]
        params['T_sup_min'][t].lb = params['T_sup_min'][t].ub = new_forecasts['T_sup_min'][t]
        params['PV_forecast'][t].lb = params['PV_forecast'][t].ub = new_forecasts['PV_forecast'][t]
        params['Load_forecast'][t].lb = params['Load_forecast'][t].ub = new_forecasts['Load_forecast'][t]

    # 2. Initial states
    params['Init_T_sup'].lb = params['Init_T_sup'].ub = new_initial_states['T_sup_current']
    params['Init_Lift_Historical'].lb = params['Init_Lift_Historical'].ub = new_initial_states['Temp_Lift_historical']
    chiller_on_prev = float(new_initial_states.get('Chiller_On_prev', 0.0))
    params['Init_Chiller_On'].lb = params['Init_Chiller_On'].ub = chiller_on_prev
    params['Init_SOC'].lb = params['Init_SOC'].ub = new_initial_states['SOC_current']

    for r in params['Init_x_state'].keys():
        for i in range(len(params['Init_x_state'][r])):
            val = new_initial_states['x_state_current'][r][i]
            params['Init_x_state'][r][i].lb = params['Init_x_state'][r][i].ub = val

    print("\n--- Solving Parametric MPC for the Next Control Step ---")

    update_objective(m, vars_dict, new_forecasts, rooms, horizon_steps)

    m.optimize()

    if m.Status == GRB.INFEASIBLE:
        print("\n!!! MPC IS INFEASIBLE !!!")
        m.computeIIS()
        for c in m.getConstrs():
            if c.IISConstr:
                print(f"Constraint: {c.ConstrName}")
        for v in m.getVars():
            if v.IISLB > 0 or v.IISUB > 0:
                print(f"Variable Bound Conflict: {v.VarName} (LB: {v.LB}, UB: {v.UB})")

    accept = (m.Status == GRB.OPTIMAL) or (
        m.Status == GRB.TIME_LIMIT and m.SolCount > 0 and m.MIPGap < 0.10
    )
    if accept:
        return _extract_optimal_action(vars_dict, rooms, holdover_blocks)
    else:
        gap_str = f"{m.MIPGap:.4f}" if m.SolCount > 0 else "N/A"
        print(f"Solver rejected: status={m.Status}, SolCount={m.SolCount}, gap={gap_str}")
        return None


def plot_mpc_results(vars_dict, params, forecasts, horizon_steps, BLOCK_SIZE=3, save_path=None,
                      BINARY_BLOCKS=8):
    """Two separate figures -- cooling (chiller/temperatures) and battery/PV/grid
    economics are largely independent stories, and one tall stack made each panel
    harder to read."""
    print("\nExtracting optimization results for visualization...")
    t_hours = np.arange(horizon_steps) * (5.0 / 60.0)

    T_sup = [vars_dict['T_sup'][t].X for t in range(horizon_steps)]
    Z_AC = [vars_dict['Z_AC_out'][t // BLOCK_SIZE].X for t in range(horizon_steps)]

    rooms = list(vars_dict['Q_fan'].keys())
    T_room = {r: [vars_dict['T_room'][r][t].X for t in range(horizon_steps)] for r in rooms}

    T_min = {r: [params['T_min'][r][t].LB for t in range(horizon_steps)] for r in rooms}
    T_max = {r: [params['T_max'][r][t].LB for t in range(horizon_steps)] for r in rooms}
    T_sup_min = [params['T_sup_min'][t].LB for t in range(horizon_steps)]
    Q_fan = {r: [vars_dict['Q_fan'][r][t].X for t in range(horizon_steps)] for r in rooms}
    binary_head_hours = BINARY_BLOCKS * BLOCK_SIZE * (5.0 / 60.0)

    Tariff_buy = [forecasts["tariffs"][t] for t in range(horizon_steps)]
    sell_tariffs = forecasts.get("sell_tariffs")
    Tariff_sell = [sell_tariffs[t] for t in range(horizon_steps)] if sell_tariffs is not None \
        else [BATTERY_FEED_IN_TARIFF_EUR_KWH] * horizon_steps

    SOC_kwh = [vars_dict['SOC'][t].X for t in range(horizon_steps)]
    soc_capacity_kwh = vars_dict['SOC'][0].UB  # UB was fixed to capacity_kwh at build time
    SOC_norm = [s / soc_capacity_kwh for s in SOC_kwh]
    P_charge = [vars_dict['P_charge'][t].X for t in range(horizon_steps)]
    P_discharge = [vars_dict['P_discharge'][t].X for t in range(horizon_steps)]
    P_grid_import = [vars_dict['P_grid_import'][t].X for t in range(horizon_steps)]
    P_grid_export = [vars_dict['P_grid_export'][t].X for t in range(horizon_steps)]
    PV_forecast = [params['PV_forecast'][t].LB for t in range(horizon_steps)]
    Load_forecast = [params['Load_forecast'][t].LB for t in range(horizon_steps)]

    # ---- Figure 1: Cooling -- chiller actuation, temperatures & fan heat flow ----
    fig1, (ax1, ax2, ax6) = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    ax1.set_title("Supply Water Temperature & Chiller Actuation", fontweight='bold')
    line1 = ax1.plot(t_hours, T_sup, color='#0072BD', linewidth=2.5, label='T_sup (°C)')
    line1b = ax1.plot(t_hours, T_sup_min, 'r--', linewidth=1.5, alpha=0.7, label='T_sup Min Bound')
    ax1.set_ylabel("Supply Temp (°C)", color='#0072BD', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax1_twin = ax1.twinx()
    line2 = ax1_twin.step(t_hours, Z_AC, color='#D95319', where='post', linewidth=2, alpha=0.7,
                           label='Chiller (ON/OFF/CONT)')
    ax1_twin.set_ylabel("Chiller Command", color='#D95319', fontweight='bold')
    ax1_twin.set_ylim(-0.1, 1.1)

    lines1 = line1 + line1b + line2
    ax1.legend(lines1, [ln.get_label() for ln in lines1], loc='upper right')

    ax2.set_title("Indoor Air Temperatures & Comfort Bounds (room-individual)", fontweight='bold')
    for r in rooms:
        line, = ax2.plot(t_hours, T_room[r], linewidth=2, alpha=0.8, label=f'{r} Temp')
        color = line.get_color()
        ax2.plot(t_hours, T_min[r], linestyle='--', linewidth=1, color=color, alpha=0.5)
        ax2.plot(t_hours, T_max[r], linestyle='--', linewidth=1, color=color, alpha=0.5)
    ax2.set_ylabel("Room Temp (°C)", fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', ncol=len(rooms), fontsize='small')

    ax6.set_title("Fan Heat Flow Q_fan (tail physics-tie diagnostic)", fontweight='bold')
    ax6.axvspan(0, binary_head_hours, color='gray', alpha=0.12, label='Binary head (executed)')
    ax6.axhline(0, color='black', linewidth=1, alpha=0.6)
    for r in rooms:
        ax6.plot(t_hours, Q_fan[r], linewidth=1.8, alpha=0.85, label=f'{r}')
    ax6.set_ylabel("Q_fan (kW, +=heating/hallucinated)", fontweight='bold')
    ax6.set_xlabel("Time (Hours)", fontweight='bold')
    ax6.grid(True, linestyle='--', alpha=0.5)
    ax6.legend(loc='upper right', ncol=len(rooms) + 1, fontsize='small')

    plt.tight_layout()

    # ---- Figure 2: Battery/PV/grid economics ----
    fig2, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    ax3.set_title("Grid: Bought/Sold Power & Tariffs", fontweight='bold')
    line3 = ax3.step(t_hours, P_grid_import, color='#7E2F8E', where='post', linewidth=2, label='Bought (import, kW)')
    line3b = ax3.step(t_hours, P_grid_export, color='#2a78d6', where='post', linewidth=2, label='Sold (export, kW)')
    ax3.set_ylabel("Power (kW)", fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)

    ax3_twin = ax3.twinx()
    line4 = ax3_twin.step(t_hours, Tariff_buy, color='black', where='post', linewidth=1.5, alpha=0.6,
                           linestyle='--', label='Buy tariff (€/kWh)')
    line4b = ax3_twin.step(t_hours, Tariff_sell, color='gray', where='post', linewidth=1.5, alpha=0.6,
                            linestyle=':', label='Sell tariff (€/kWh)')
    ax3_twin.set_ylabel("Tariff (€/kWh)", fontweight='bold')

    lines3 = line3 + line3b + line4 + line4b
    ax3.legend(lines3, [ln.get_label() for ln in lines3], loc='upper right', fontsize='small')

    ax4.set_title("PV Generation & Uncontrollable Load Forecast", fontweight='bold')
    ax4.step(t_hours, PV_forecast, color='#EDB120', where='post', linewidth=2, label='PV generation (kW)')
    ax4.step(t_hours, Load_forecast, color='#7E2F8E', where='post', linewidth=2, label='Load (kW)')
    ax4.set_ylabel("Power (kW)", fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(loc='upper right', fontsize='small')

    ax5.set_title("Battery: Charge/Discharge & State of Charge", fontweight='bold')
    line5 = ax5.step(t_hours, P_charge, color='#77AC30', where='post', linewidth=2, label='Charge (kW)')
    line5b = ax5.step(t_hours, P_discharge, color='#D95319', where='post', linewidth=2, label='Discharge (kW)')
    ax5.set_ylabel("Power (kW)", fontweight='bold')
    ax5.grid(True, linestyle='--', alpha=0.5)

    ax5_twin = ax5.twinx()
    line6 = ax5_twin.plot(t_hours, SOC_norm, color='#2a78d6', linewidth=2.5, alpha=0.9, label='SOC (0-1)')
    ax5_twin.set_ylabel("SOC (normalized)", color='#2a78d6', fontweight='bold')
    ax5_twin.set_ylim(-0.05, 1.05)

    lines5 = line5 + line5b + line6
    ax5.legend(lines5, [ln.get_label() for ln in lines5], loc='upper right', fontsize='small')
    ax5.set_xlabel("Time (Hours)", fontweight='bold')

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig1.savefig(save_path.with_name(f"{save_path.stem}_cooling{save_path.suffix}"),
                      dpi=150, bbox_inches="tight")
        fig2.savefig(save_path.with_name(f"{save_path.stem}_battery{save_path.suffix}"),
                      dpi=150, bbox_inches="tight")
        plt.close(fig1)
        plt.close(fig2)
    else:
        plt.show()
