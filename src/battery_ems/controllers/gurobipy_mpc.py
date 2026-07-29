import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

# Per-room comfort-slack penalty (€/°C-equivalent weight on Slack_T). Default
# matches the previous global value; override rooms individually here.
SLACK_T_WEIGHT_DEFAULT = 1.0
SLACK_T_WEIGHT_OVERRIDE = {
    "room_5": 1.0,
}

# Assumed FCU fan electrical draw (kW), priced at the real tariff like the
# chiller's P_elec. 50W is an estimate, not a measured value.
FAN_POWER_KW = 0.05

# Small robustness bias against STARTING the chiller purely to win a marginal,
# near-noise-level economic tie-break (e.g., to avoid the chiller turning on because the model thinks this saves 0.01 EUR)
CHILLER_SWITCH_BIAS = 0.05  # EUR-equivalent per OFF->ON transition

# small slack to allow T_sup > T_room. Large enough to avoid "cheating of free cooling" while
# low enough to not enforce T_sup < T_room
SLACK_CROSS_WEIGHT_DEFAULT = 0.01
SLACK_CROSS_WEIGHT_OVERRIDE = {"room_5": 1.0}

# If consecutive solves fail, continue to execute the last available MPC plan for "HOLDOVER_BLOCKS" time steps before
# using a fallback mechanism. Must stay <= BINARY_BLOCKS so these are true binaries,
# not rounded continuous-relaxation values.
HOLDOVER_BLOCKS = 8


def build_parametric_mpc(horizon_steps, config):
    print("\n--- Compiling Parametric MPC Matrix ---")
    m = gp.Model("Parametric_Supervisory_MPC")

    # Suppress console output for the live loop
    m.setParam('OutputFlag', 0)
    m.setParam("MIPGap", 0.005)
    m.setParam("MIPGapAbs", 0.001)
    m.setParam("MIPFocus", 1)
    m.setParam("Heuristics", 0.3)
    m.setParam("Presolve", 2)
    # Hard time limit of 2 min = 120s. If the gap is still >=10%
    # when this fires, step_mpc() rejects the incumbent (see below) and the
    # we implement the fallback: The solution of the previous execution step at
    # t=1.
    m.setParam("TimeLimit", 120)

    plant = config['plant_physics']['plant_thermal_model']  # tail-only: blended, Z_AC as linear feature
    plant_on = config['plant_physics']['plant_thermal_model_on']  # head-only: exact regime, Z_AC=1
    plant_off = config['plant_physics']['plant_thermal_model_off']  # head-only: exact regime, Z_AC=0
    power_bin = config['plant_physics']['chiller_power_model']
    power_cont = config['plant_physics']['chiller_power_model_continuous']
    fans = config['fan_physics']
    rc_sys = config['rc_models']
    rooms = list(rc_sys.keys())

    # =========================================================================
    # 1. PARAMETER INJECTORS (Fixed Variables for easy updating)
    # =========================================================================
    params = {
        'T_amb': m.addVars(horizon_steps, lb=0.0, ub=0.0, name="Param_T_amb"),
        'Solar': m.addVars(horizon_steps, lb=0.0, ub=0.0, name="Param_Solar"),
        # Room-individual comfort bounds (see forecast_provider._comfort_schedule
        # for the per-room override, e.g. room_5's relaxed T_max)
        'T_min': {r: m.addVars(horizon_steps, lb=0.0, ub=0.0, name=f"Param_T_min_{r}") for r in rooms},
        'T_max': {r: m.addVars(horizon_steps, lb=0.0, ub=0.0, name=f"Param_T_max_{r}") for r in rooms},
        'T_sup_min': m.addVars(horizon_steps, lb=0.0, ub=0.0, name="P_T_sup_min"),

        # Initial States
        'Init_T_sup': m.addVar(lb=0.0, ub=0.0, name="Init_T_sup"),
        'Init_Lift_Historical': m.addVar(lb=0.0, ub=0.0, name="Init_Lift_Hist"),
        'Init_x_state': {r: m.addVars(len(rc_sys[r]['A']), lb=0.0, ub=0.0, name=f"Init_x_{r}") for r in rooms},
        # Was the chiller physically ON in the 15-min block right before t=0?
        # Needed for the hysteresis-aware Chiller_On_Threshold gate below (only
        # an OFF->ON transition needs to satisfy T_sup >= 18; staying ON does not).
        'Init_Chiller_On': m.addVar(lb=0.0, ub=0.0, name="Init_Chiller_On")
    }

    # =========================================================================
    # 2. DECISION & STATE VARIABLES
    # =========================================================================
    BLOCK_SIZE = 3  # 3 steps * 5 mins = 15 minute control resolution
    horizon_blocks = horizon_steps // BLOCK_SIZE
    BINARY_BLOCKS = 12

    assert HOLDOVER_BLOCKS < BINARY_BLOCKS, (
        f"HOLDOVER_BLOCKS ({HOLDOVER_BLOCKS}) must stay within the binary head "
        f"(BINARY_BLOCKS={BINARY_BLOCKS}) so step_mpc's Holdover_*_Commands are "
        f"true binaries, not rounded continuous-relaxation values."
    )

    # Use standard dicts so we can mix Binary and Continuous variables seamlessly
    Z_AC_15 = {}
    Z_fan_15 = {r: {} for r in rooms}

    for k in range(horizon_blocks):
        if k < BINARY_BLOCKS:
            # Hours 0-2: Strict Binary for hardware execution
            Z_AC_15[k] = m.addVar(vtype=GRB.BINARY, name=f"Z_AC_bin_{k}")
            for r in rooms:
                Z_fan_15[r][k] = m.addVar(vtype=GRB.BINARY, name=f"Z_fan_bin_{r}_{k}")
        else:
            # Hours 2-24: Relaxed Continuous to eliminate Branch-and-Bound tree
            Z_AC_15[k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"Z_AC_cont_{k}")
            for r in rooms:
                Z_fan_15[r][k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"Z_fan_cont_{r}_{k}")

    T_ROOM_LB = 18.0
    T_ROOM_UB = 28.0
    T_SUP_LB = 5.0
    T_SUP_UB = 40.0

    # Physical envelope for outdoor temperature, used only to size the T_sup regime-switch
    # Big-M below -- not a hard variable bound, just a safe range for this location's
    # climate (measured summer data already touches 33C).
    ENVTMP_BIGM_LB, ENVTMP_BIGM_UB = -15.0, 45.0

    # PHYSICS VARIABLES (High Resolution - 5 mins)
    vars_dict = {
        'T_sup': m.addVars(horizon_steps + 1, lb=T_SUP_LB, ub=T_SUP_UB, name="T_sup"),
        'P_elec': m.addVars(horizon_steps, lb=0.0, ub=3.0, name="P_elec"),  # Watts
        'Temp_Lift': m.addVars(horizon_steps, lb=-40.0, ub=40.0, name="Temp_Lift"),
        'T_room': {}, 'x_state': {}, 'Q_fan': {}, 'Slack_T': {},
        'Z_AC_out': Z_AC_15, 'Z_fan_out': Z_fan_15,  # Expose dicts for extraction
        'Slack_T_sup': m.addVars(horizon_steps, lb=0.0, name="Slack_T_sup")
    }

    # McCormick box [b_L, b_U] for Q_fan = Z_fan * Max_Cooling, per room
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

    # =========================================================================
    # 3. STATIC CONSTRAINTS
    # =========================================================================
    m.addConstr(vars_dict['T_sup'][0] == params['Init_T_sup'])
    M_POWER = 3.0  # Max physical electrical kW of the chiller
    M_TSUP_ON = 20.0  # Big-M for the chiller-on threshold (T_sup ub=35, threshold=18 -> need >= 17)
    T_SUP_ON_THRESHOLD = 19.0  # Chiller can only be commanded ON (from OFF) if T_sup was already above this
    M_TSUP_OFF = 12.0  # Big-M for the forced-shutoff threshold (T_sup lb=2.0, threshold=10 -> need >= 8)
    T_SUP_OFF_THRESHOLD = 10.0  # Chiller cannot run (ON or OFF) once T_sup has dropped below this

    # Hysteresis-aware chiller-on gate: only an OFF->ON *transition* needs
    # T_sup (at the start of that 15-min block) to already be above
    # T_SUP_ON_THRESHOLD. Staying ON (already running) is unrestricted at the
    # top end -- the real unit can keep running as T_sup drops further once
    # engaged -- but is unconditionally cut off once T_sup drops below
    # T_SUP_OFF_THRESHOLD (bottom-end hysteresis limit, applies regardless of
    # transition/history).
    #
    # turn_on[k] is the standard max(0, x) hinge for (Z_AC[k] - prev_z): since
    # turn_on has no cost in the objective and only tightens the gate as it
    # grows, the solver always drives it down to exactly max(0, Z_AC[k] - prev_z)
    # -- 1 only on a genuine off->on transition, 0 otherwise (staying on/off,
    # or turning off). No extra binaries needed.
    #
    # Restricted to the BINARY head only: these gates approximate the real
    # hardware's hysteresis for the decision that actually gets executed. In
    # the continuous relaxed tail, Z_AC is a fractional value-function
    # approximation, not a real chiller state -- applying the force-off gate
    # there turns it into a continuous cap Z[k] <= (T_sup[t0]+2)/10 that
    # feeds back through the T_sup plant dynamics (Z[k] -> T_sup[t0(k+1)] ->
    # cap on Z[k+1] -> ...), producing a spurious sawtooth oscillation with
    # no basis in real hardware behavior and no effect on what's executed.
    vars_dict['Chiller_TurnOn'] = {}
    Z_AC_prev = None
    for k in range(BINARY_BLOCKS):
        t0 = k * BLOCK_SIZE
        prev_z = Z_AC_prev if Z_AC_prev is not None else params['Init_Chiller_On']
        turn_on = m.addVar(lb=0.0, ub=1.0, name=f"Chiller_TurnOn_{k}")
        m.addConstr(turn_on >= Z_AC_15[k] - prev_z, name=f"Chiller_TurnOn_Def_{k}")
        vars_dict['Chiller_TurnOn'][k] = turn_on
        m.addConstr(
            vars_dict['T_sup'][t0] >= T_SUP_ON_THRESHOLD - M_TSUP_ON * (1 - turn_on),
            name=f"Chiller_On_Threshold_{k}"
        )

        for i in range(BLOCK_SIZE):
            t_sub = t0 + i
            m.addConstr(
                vars_dict['T_sup'][t_sub] >= T_SUP_OFF_THRESHOLD - M_TSUP_OFF * (1 - Z_AC_15[k]),
                name=f"Chiller_Off_Threshold_{k}_{i}"
            )
        Z_AC_prev = Z_AC_15[k]

    # Big-M for the head's exact ON/OFF regime-switch: the tightest M that still
    # guarantees correctness is the max possible |on_expr - off_expr| across the full
    # range T_sup/EnvTmp/Total_Q_Load can take, computed from the actual fitted
    # coefficients and variable bounds rather than an arbitrary large constant. ON
    # is a plain constant (plant_thermal_model_on), so only OFF's own T_sup/EnvTmp/
    # Total_Thermal_Load terms vary. Total_Q_Load's range comes from Q_FAN_BOUNDS
    # (each room's McCormick box), summed across rooms: [sum(b_L), 0].
    _d_a = plant_on['Constant'] - plant_off['Intercept']
    _d_t = 0.0 - plant_off['T_sup_current']
    _d_e = 0.0 - plant_off['EnvTmp']
    _d_q = 0.0 - plant_off.get('Total_Thermal_Load', 0.0)
    _t_lo, _t_hi = sorted([_d_t * T_SUP_LB, _d_t * T_SUP_UB])
    _e_lo, _e_hi = sorted([_d_e * ENVTMP_BIGM_LB, _d_e * ENVTMP_BIGM_UB])
    _q_total_lb = sum(Q_FAN_BOUNDS[r][0] for r in rooms)
    _q_lo, _q_hi = sorted([_d_q * _q_total_lb, _d_q * 0.0])
    _diff_max = _d_a + _t_hi + _e_hi + _q_hi
    _diff_min = _d_a + _t_lo + _e_lo + _q_lo
    M_TSUP = 1.1 * max(abs(_diff_max), abs(_diff_min))

    # Only exists for continuous-tail (room, t) pairs -- see the k>=BINARY_BLOCKS branch
    # below. Populated into vars_dict directly so update_objective can penalize it.
    vars_dict['Slack_Cross'] = {r: {} for r in rooms}

    for t in range(horizon_steps):
        # MATHEMATICAL LINK: Find which 15-minute decision block governs this 5-minute step
        k = t // BLOCK_SIZE

        # Q_fan is bounded <=0 for every room in both head and tail (see the per-room
        # loop below, Q_fan[r][t].ub is forced to 0.0 in both branches), so this sum is
        # already <=0 by construction -- no separate capped auxiliary needed. (A
        # previous version used a one-sided `<=Total_Q_Load` auxiliary meant to block
        # positive-Q exploits; being one-sided, it let the solver instead fake an
        # arbitrarily *more negative* load whenever that helped -- confirmed via
        # diagnose_tail_room5.py on feature/pv-battery-mpc, same physics, ported here.)
        Total_Q_Load = gp.quicksum(vars_dict['Q_fan'][r][t] for r in rooms)

        m.addConstr(vars_dict['Temp_Lift'][t] == params['T_amb'][t] - vars_dict['T_sup'][t])

        # Central Plant Physics
        if k < BINARY_BLOCKS:
            # Binary head: exact ON/OFF regime-switch (Z_AC_15[k] is a true 0/1 here, so
            # this Big-M selection has zero relaxation error). ON is a plain constant --
            # its time constant (~2.7min) is far below the 15-min block, so T_sup sits at
            # its steady value almost immediately, and no load or ambient term survived a
            # held-out refit (Q_fan is itself slope*(T_room-T_sup), so regressing T_sup on
            # delivered cooling regresses it on its own consequence -- confirmed by the
            # fitted sign flipping across held-out splits; dropping it entirely improved
            # deployable propagated cooling/power R2 over every load-based variant tried).
            # OFF keeps a genuine AR term: its ~15-20min time constant is comparable to
            # the 15-min block, so passive warm-up needs real memory. Both
            # plant_thermal_model_on/_off were fit at 15-min block resolution then
            # converted to this 5-min physics-step resolution via cube-root compounding
            # (D5=D15**(1/3), K5=K15/(1+D5+D5**2), verified to reproduce the validated
            # 15-min transition after 3 steps to 1e-15) -- see mpc_plant_power_coefs.json's
            # _15min_fit fields and session retrain notes for the derivation.
            on_expr = plant_on['Constant']
            off_expr = (plant_off['Intercept'] +
                        plant_off['T_sup_current'] * vars_dict['T_sup'][t] +
                        plant_off['EnvTmp'] * params['T_amb'][t] +
                        plant_off.get('Total_Thermal_Load', 0.0) * Total_Q_Load)
            m.addConstr(vars_dict['T_sup'][t + 1] <= on_expr + M_TSUP * (1 - Z_AC_15[k]), name=f"Tsup_On_Hi_{t}")
            m.addConstr(vars_dict['T_sup'][t + 1] >= on_expr - M_TSUP * (1 - Z_AC_15[k]), name=f"Tsup_On_Lo_{t}")
            m.addConstr(vars_dict['T_sup'][t + 1] <= off_expr + M_TSUP * Z_AC_15[k], name=f"Tsup_Off_Hi_{t}")
            m.addConstr(vars_dict['T_sup'][t + 1] >= off_expr - M_TSUP * Z_AC_15[k], name=f"Tsup_Off_Lo_{t}")
        else:
            # Continuous tail: single blended model, additive in Z_AC (value-function
            # approximation only -- never executed). No load term, same reverse-causation
            # reasoning as the head; also converted from its 15-min block fit to this
            # 5-min step resolution the same way as OFF above.
            m.addConstr(
                vars_dict['T_sup'][t + 1] == plant['Intercept'] +
                plant['T_sup_current'] * vars_dict['T_sup'][t] +
                plant['Chiller_Command'] * Z_AC_15[k] +
                plant['EnvTmp'] * params['T_amb'][t] +
                plant.get('Total_Thermal_Load', 0.0) * Total_Q_Load,
                name=f"Tsup_Tail_{t}"
            )
        m.addConstr(
            vars_dict['T_sup'][t] + vars_dict['Slack_T_sup'][t] >= params['T_sup_min'][t],
            name=f"T_sup_Min_Bound_{t}"
        )

        # Power Physics
        if k < BINARY_BLOCKS:
            # --- HOURS 0-2 (Binary Head): Active-Only Power Model ---
            # Refit this session on MPC-era-only data, static (no Delta_Temp_Lift ramp
            # term -- unvalidated at 15-min block resolution, dropped for simplicity
            # pending evidence it earns its place back in).
            P_active = (power_bin['active_intercept'] +
                        power_bin['Temp_Lift'] * vars_dict['Temp_Lift'][t] +
                        power_bin['Total_Thermal_Load'] * Total_Q_Load)

            # Minimum standby limit
            m.addConstr(vars_dict['P_elec'][t] >= power_bin["standby_kW"], name=f"P_Standby_{t}")

            # Standard Big-M mapping (Since Z_AC is pure binary here, this is perfect)
            m.addConstr(
                vars_dict['P_elec'][t] >= P_active - M_POWER * (1 - Z_AC_15[k]),
                name=f"P_Active_Standard_{t}"
            )
        else:
            # --- HOURS 2-24 (Continuous): Linear Model including Chiller_Command (Z_AC) as feature
            P_reg = (power_cont['active_intercept'] +
                     power_cont['Temp_Lift'] * vars_dict['Temp_Lift'][t] +
                     power_cont['Total_Thermal_Load'] * Total_Q_Load +
                     power_cont['Chiller_Command'] * Z_AC_15[k])

            # Perspective/Convex Hull Bounds to protect continuous Z_AC from "Free Energy" exploits
            m.addConstr(vars_dict['P_elec'][t] >= P_reg, name=f"P_Reg_Floor_{t}")
            m.addConstr(vars_dict['P_elec'][t] >= power_bin["standby_kW"], name=f"P_Standby_Cont_{t}")

        # Room & Fan Physics
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
                # actually executes, so it's left exactly as the static box gives it.
                # Q_fan.ub is capped at 0 here too (matching the tail) -- FCUs are
                # cooling-only, can't "heat" a room -- but b_U itself is left positive:
                # b_U is a McCormick envelope bound (needs to cover Max_Cooling's full
                # possible range, which is genuinely positive when T_sup>T_room), not
                # the physical output cap. Capping b_U itself to 0 instead would collapse
                # the Lo2 inequality's Zf=0 slack term and force Q_fan>=Max_Cooling
                # unconditionally -- making Zf=1 infeasible AND Zf=0 infeasible whenever
                # T_sup>T_room, which would wrongly pressure the chiller to keep
                # T_sup<T_room even when no room wants cooling.
                b_L, b_U = Q_FAN_BOUNDS[r]
                Zf = Z_fan_15[r][k]
                vars_dict['Q_fan'][r][t].ub = 0.0
                m.addConstr(vars_dict['Q_fan'][r][t] >= Zf * b_L, name=f"Fan_McC_Lo1_{r}_{t}")
                m.addConstr(vars_dict['Q_fan'][r][t] >= Max_Cooling - b_U * (1 - Zf), name=f"Fan_McC_Lo2_{r}_{t}")
                m.addConstr(vars_dict['Q_fan'][r][t] <= Max_Cooling - b_L * (1 - Zf), name=f"Fan_McC_Hi1_{r}_{t}")
                m.addConstr(vars_dict['Q_fan'][r][t] <= Zf * b_U, name=f"Fan_McC_Hi2_{r}_{t}")
            else:
                # Continuous tail: Zf is a fractional value-function artifact (its own
                # tiny electricity cost aside), so instead of relaxing Q_fan=Zf*Max_Cooling
                # via McCormick -- whose envelope has real slack at fractional Zf, and was
                # confirmed as the dominant source of hallucinated tail heating -- tie
                # Q_fan directly to the exact physics: never positive (a chilled-water
                # coil cannot heat a room), never colder than what THIS state's
                # Max_Cooling genuinely allows. The one gap: T_sup can genuinely drift
                # above T_room in the tail (e.g. chiller idle a while), making Max_Cooling
                # briefly positive -- Slack_Cross absorbs that rather than making the
                # constraint infeasible, which would otherwise force the chiller to run
                # purely to keep T_sup<T_room, for no real cooling reason. Q_fan's own
                # declared lb (from the static box, Q_FAN_BOUNDS) still backs this up as a
                # physical floor, so Slack_Cross can't be exploited to claim cooling
                # beyond real capacity (see SLACK_CROSS_WEIGHT_DEFAULT/_OVERRIDE).
                vars_dict['Q_fan'][r][t].ub = 0.0
                slack_cross = m.addVar(lb=0.0, name=f"Slack_Cross_{r}_{t}")
                vars_dict['Slack_Cross'][r][t] = slack_cross
                m.addConstr(vars_dict['Q_fan'][r][t] >= Max_Cooling - slack_cross,
                            name=f"Q_fan_Physical_Lo_{r}_{t}")

            # RC State Transition
            u_t = [params['T_amb'][t], vars_dict['Q_fan'][r][t], params['Solar'][t], 0.0]
            for i in range(A.shape[0]):
                m.addConstr(vars_dict['x_state'][r][i, t + 1] ==
                            gp.quicksum(A[i, j] * vars_dict['x_state'][r][j, t] for j in range(A.shape[1])) +
                            gp.quicksum(B[i, j] * u_t[j] for j in range(B.shape[1])))

            m.addConstr(vars_dict['T_room'][r][t] == gp.quicksum(
                C[0, j] * vars_dict['x_state'][r][j, t] for j in range(C.shape[1])))

    for r in rooms:
        for i in range(len(rc_sys[r]['A'])):
            m.addConstr(vars_dict['x_state'][r][i, 0] == params['Init_x_state'][r][i])

    m.update()
    print("Static Parametric Matrix Compiled.")
    return m, params, vars_dict


def update_objective(m, vars_dict, new_forecasts, rooms, horizon_steps, BLOCK_SIZE=3):
    dt_h = 5.0 / 60.0  # 5 min physics steps in hours

    obj = gp.LinExpr()

    for t in range(horizon_steps):
        tariff = new_forecasts["tariffs"][t]
        obj += vars_dict["P_elec"][t] * tariff * dt_h

    for turn_on in vars_dict["Chiller_TurnOn"].values():
        obj += CHILLER_SWITCH_BIAS * turn_on

    for r in rooms:
        slack_t_weight = SLACK_T_WEIGHT_OVERRIDE.get(r, SLACK_T_WEIGHT_DEFAULT)
        for t in range(horizon_steps):
            obj += vars_dict["Slack_T"][r][t] * slack_t_weight
            k = t // BLOCK_SIZE
            obj += FAN_POWER_KW * vars_dict["Z_fan_out"][r][k] * new_forecasts["tariffs"][t] * dt_h

    for t in range(horizon_steps):
        obj += vars_dict["Slack_T_sup"][t] * 10.0

    # Continuous-tail Q_fan's physical lower-bound slack (see SLACK_CROSS_WEIGHT_DEFAULT/
    # SLACK_CROSS_WEIGHT_OVERRIDE) -- only exists for (room, t) pairs in the tail, see
    # build_parametric_mpc's per-timestep room loop.
    for r in rooms:
        slack_cross_weight = SLACK_CROSS_WEIGHT_OVERRIDE.get(r, SLACK_CROSS_WEIGHT_DEFAULT)
        for t in vars_dict["Slack_Cross"][r]:
            obj += vars_dict["Slack_Cross"][r][t] * slack_cross_weight

    m.setObjective(obj, GRB.MINIMIZE)


def step_mpc(m, params, vars_dict, new_initial_states, new_forecasts, horizon_steps):
    # 1. Update Forecast Parameters
    for t in range(horizon_steps):
        params['T_amb'][t].lb = new_forecasts['T_amb'][t]
        params['T_amb'][t].ub = new_forecasts['T_amb'][t]
        params['Solar'][t].lb = new_forecasts['Solar'][t]
        params['Solar'][t].ub = new_forecasts['Solar'][t]
        for r in params['T_min'].keys():
            params['T_min'][r][t].lb = new_forecasts['T_min'][r][t]
            params['T_min'][r][t].ub = new_forecasts['T_min'][r][t]
            params['T_max'][r][t].lb = new_forecasts['T_max'][r][t]
            params['T_max'][r][t].ub = new_forecasts['T_max'][r][t]
        params['T_sup_min'][t].lb = new_forecasts['T_sup_min'][t]
        params['T_sup_min'][t].ub = new_forecasts['T_sup_min'][t]

    # 2. Update Sensor Initial States
    params['Init_T_sup'].lb = new_initial_states['T_sup_current']
    params['Init_T_sup'].ub = new_initial_states['T_sup_current']
    params['Init_Lift_Historical'].lb = new_initial_states['Temp_Lift_historical']
    params['Init_Lift_Historical'].ub = new_initial_states['Temp_Lift_historical']
    chiller_on_prev = float(new_initial_states.get('Chiller_On_prev', 0.0))
    params['Init_Chiller_On'].lb = chiller_on_prev
    params['Init_Chiller_On'].ub = chiller_on_prev

    for r in params['Init_x_state'].keys():
        for i in range(len(params['Init_x_state'][r])):
            val = new_initial_states['x_state_current'][r][i]
            params['Init_x_state'][r][i].lb = val
            params['Init_x_state'][r][i].ub = val

    print("\n--- Solving Parametric MPC for the Next Control Step ---")

    rooms = list(vars_dict["Q_fan"].keys())
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

    # 4. Extract commands for t=0 (executed now) and t=1 (held as fallback)
    # Accept either a genuine optimum (converged to the MIPGap target), or a
    # TimeLimit incumbent that's still reasonably close (<2% gap) -- anything
    # worse is rejected so the caller falls back to the previous solve's
    # held-over plan (see run_mpc.py's _handle_infeasible) rather than
    # executing a decision the solver never got close to trusting.
    accept = (m.Status == GRB.OPTIMAL) or (
        m.Status == GRB.TIME_LIMIT and m.SolCount > 0 and m.MIPGap < 0.02
    )
    if accept:
        rooms = list(vars_dict['Z_fan_out'].keys())
        optimal_action = {
            'Chiller_Command': round(vars_dict['Z_AC_out'][0].X),
            'Fan_Commands': {r: round(vars_dict['Z_fan_out'][r][0].X) for r in rooms},
            # Blocks k=1..HOLDOVER_BLOCKS stored so run_mpc.py can coast through
            # up to HOLDOVER_BLOCKS consecutive solve failures, advancing one
            # block of this plan's own trajectory per failed cycle, before
            # giving up to a hard fallback (see run_mpc.py's _handle_infeasible).
            'Holdover_Chiller_Commands': [
                round(vars_dict['Z_AC_out'][k].X) for k in range(1, HOLDOVER_BLOCKS + 1)
            ],
            'Holdover_Fan_Commands': [
                {r: round(vars_dict['Z_fan_out'][r][k].X) for r in rooms}
                for k in range(1, HOLDOVER_BLOCKS + 1)
            ],
        }
        return optimal_action
    else:
        gap_str = f"{m.MIPGap:.4f}" if m.SolCount > 0 else "N/A"
        print(f"Solver rejected: status={m.Status}, SolCount={m.SolCount}, gap={gap_str}")
        return None


def plot_mpc_results(vars_dict, params, forecasts, horizon_steps, BLOCK_SIZE=3, save_path=None):
    print("\nExtracting optimization results for visualization...")
    t_hours = np.arange(horizon_steps) * (5.0 / 60.0)

    T_sup = [vars_dict['T_sup'][t].X for t in range(horizon_steps)]
    Z_AC = [vars_dict['Z_AC_out'][t // BLOCK_SIZE].X for t in range(horizon_steps)]

    rooms = list(vars_dict['Q_fan'].keys())
    Q_fan = {r: [vars_dict['Q_fan'][r][t].X for t in range(horizon_steps)] for r in rooms}
    T_room = {r: [vars_dict['T_room'][r][t].X for t in range(horizon_steps)] for r in rooms}

    # Room-individual now (see forecast_provider._comfort_schedule's per-room override)
    T_min = {r: [params['T_min'][r][t].LB for t in range(horizon_steps)] for r in rooms}
    T_max = {r: [params['T_max'][r][t].LB for t in range(horizon_steps)] for r in rooms}
    T_sup_min = [params['T_sup_min'][t].LB for t in range(horizon_steps)]

    P_elec = [vars_dict['P_elec'][t].X for t in range(horizon_steps)]
    Tariff = [forecasts["tariffs"][t] for t in range(horizon_steps)]

    fig, axs = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    ax1 = axs[0]
    ax1.set_title("Plant Layer: Supply Water Temperature & Chiller Actuation", fontweight='bold')
    line1 = ax1.plot(t_hours, T_sup, color='#0072BD', linewidth=2.5, label='T_sup (°C)')
    ax1.plot(t_hours, T_sup_min, 'r--', linewidth=1.5, alpha=0.7, label='T_sup Min Bound (10°C)')
    ax1.set_ylabel("Supply Temp (°C)", color='#0072BD', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax1_twin = ax1.twinx()
    line2 = ax1_twin.step(t_hours, Z_AC, color='#D95319', where='post', linewidth=2, alpha=0.7,
                          label='Chiller (ON/OFF/CONT)')
    ax1_twin.set_ylabel("Chiller Command", color='#D95319', fontweight='bold')
    ax1_twin.set_ylim(-0.1, 1.1)

    lines = line1 + [ax1.lines[1]] + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')

    ax2 = axs[1]
    ax2.set_title("Delivery Layer: Fan Coil Heat Extraction", fontweight='bold')
    for r in rooms:
        ax2.step(t_hours, Q_fan[r], where='post', linewidth=1.5, alpha=0.8, label=f'{r} Cooling')
    ax2.set_ylabel("Heat Flow (kW)", fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', ncol=len(rooms), fontsize='small')

    ax3 = axs[2]
    ax3.set_title("Zone Layer: Indoor Air Temperatures & Comfort Bounds (room-individual)", fontweight='bold')
    for r in rooms:
        line, = ax3.plot(t_hours, T_room[r], linewidth=2, alpha=0.8, label=f'{r} Temp')
        color = line.get_color()
        ax3.plot(t_hours, T_min[r], linestyle='--', linewidth=1, color=color, alpha=0.5)
        ax3.plot(t_hours, T_max[r], linestyle='--', linewidth=1, color=color, alpha=0.5)
    ax3.set_ylabel("Room Temp (°C)", fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='lower left', ncol=len(rooms) + 1, fontsize='small')

    ax4 = axs[3]
    ax4.set_title("Economic Layer: Electrical Power vs. Electricity Tariff", fontweight='bold')
    line3 = ax4.plot(t_hours, P_elec, color='#7E2F8E', linewidth=2.5, label='Power (kW)')
    ax4.set_ylabel("Power Draw (kW)", color='#7E2F8E', fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)

    ax4_twin = ax4.twinx()
    line4 = ax4_twin.step(t_hours, Tariff, color='#EDB120', where='post', linewidth=2, alpha=0.9,
                          label='Tariff (€/kWh)')
    ax4_twin.set_ylabel("Tariff (€/kWh)", color='#EDB120', fontweight='bold')

    lines_eco = line3 + line4
    labels_eco = [line.get_label() for line in lines_eco]
    ax4.legend(lines_eco, labels_eco, loc='upper right')
    ax4.set_xlabel("Time (Hours)", fontweight='bold')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
