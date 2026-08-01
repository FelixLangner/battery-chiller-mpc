import gurobipy as gp
import numpy as np
from gurobipy import GRB

from battery_ems.controllers.mpc_config import MPCConfig

DEFAULT_CONFIG = MPCConfig()


class ParametricMPC:
    """Joint cooling+PV+battery MPC. The Gurobi model is built once (__init__);
    call .step(...) to re-solve for each control step by mutating variable
    bounds on self.params (warm-startable parametric pattern) rather than
    rebuilding from scratch."""

    def __init__(self, horizon_steps, config, cfg: MPCConfig = DEFAULT_CONFIG):
        print("\n--- Compiling Joint Cooling+PV+Battery Parametric MPC Matrix ---")
        self.horizon_steps = horizon_steps
        self.cfg = cfg

        self.rc_sys = config['rc_models']
        self.rooms = list(self.rc_sys.keys())
        self.fans = config['fan_physics']
        self.battery = config['battery_physics']  # power_offset_w is a sensor-bias fit artifact, unused here
        self.plant_physics = config['plant_physics']

        self.dt_h = cfg.dt_h_hours  # hours per 5-min physics step (also used in update_objective)

        self.m = gp.Model("Parametric_Joint_MPC")
        self.m.setParam('OutputFlag', cfg.solver_output_flag)
        self.m.setParam("MIPGap", cfg.solver_mip_gap)
        self.m.setParam("MIPGapAbs", cfg.solver_mip_gap_abs)
        self.m.setParam("MIPFocus", cfg.solver_mip_focus)
        self.m.setParam("Heuristics", cfg.solver_heuristics)
        self.m.setParam("Presolve", cfg.solver_presolve)
        self.m.setParam("TimeLimit", cfg.solver_time_limit_s)

        self.params = {}
        self._add_cooling_params()
        self._add_battery_params()

        self._add_decision_vars()  # sets self.Z_AC_15, self.Z_fan_15, self.block_size, self.binary_blocks

        assert cfg.holdover_blocks < self.binary_blocks, (
            f"holdover_blocks ({cfg.holdover_blocks}) must stay within the binary head "
            f"(binary_blocks={self.binary_blocks}) so the holdover commands extracted from a "
            f"previous solve are true binaries, not rounded continuous-relaxation values."
        )
        self.holdover_blocks = cfg.holdover_blocks

        self.vars_dict = {}
        self._add_cooling_state_vars()  # sets self.Q_FAN_BOUNDS
        self._add_battery_state_vars()

        self._add_initial_conditions()
        self._add_chiller_hysteresis_constraints()
        self._add_cooling_dynamics()
        self._add_battery_dynamics()
        self._add_soc_band_constraints()
        self._add_battery_ramp_vars()

        self.m.update()
        print("Static Parametric Matrix Compiled.")

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _add_cooling_params(self) -> None:
        """Forecast/initial-state injector vars for the cooling side (fixed lb=ub=0.0 at
        build time, mutated in step() every solve)."""
        m, horizon_steps, rooms, rc_sys = self.m, self.horizon_steps, self.rooms, self.rc_sys
        self.params.update({
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
        })

    def _add_battery_params(self) -> None:
        """Same parametric pattern as _add_cooling_params, for the battery/PV side."""
        m, horizon_steps = self.m, self.horizon_steps
        self.params.update({
            'Init_SOC': m.addVar(lb=0.0, ub=0.0, name="Init_SOC"),
            'PV_forecast': m.addVars(horizon_steps, lb=0.0, ub=0.0, name="Param_PV_forecast"),
            'Load_forecast': m.addVars(horizon_steps, lb=0.0, ub=0.0, name="Param_Load_forecast"),
        })

    def _add_decision_vars(self) -> None:
        """Chiller/fan commands: strict binary for the first 2h (executed head), relaxed
        continuous [0,1] beyond that (terminal-cost approximation only)."""
        m, horizon_steps, rooms, cfg = self.m, self.horizon_steps, self.rooms, self.cfg
        self.block_size = cfg.block_size  # 3 x 5-min physics steps = 15-min control resolution
        self.horizon_blocks = horizon_steps // self.block_size
        self.binary_blocks = cfg.binary_blocks  # first 2h

        Z_AC_15 = {}
        Z_fan_15 = {r: {} for r in rooms}
        for k in range(self.horizon_blocks):
            if k < self.binary_blocks:
                Z_AC_15[k] = m.addVar(vtype=GRB.BINARY, name=f"Z_AC_bin_{k}")
                for r in rooms:
                    Z_fan_15[r][k] = m.addVar(vtype=GRB.BINARY, name=f"Z_fan_bin_{r}_{k}")
            else:
                Z_AC_15[k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"Z_AC_cont_{k}")
                for r in rooms:
                    Z_fan_15[r][k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"Z_fan_cont_{r}_{k}")

        self.Z_AC_15 = Z_AC_15
        self.Z_fan_15 = Z_fan_15

    def _add_cooling_state_vars(self) -> None:
        """T_sup/P_elec/room states, plus the McCormick box for each room's Q_fan."""
        m, horizon_steps, rooms, cfg = self.m, self.horizon_steps, self.rooms, self.cfg
        rc_sys, fans = self.rc_sys, self.fans
        vd = self.vars_dict
        vd.update({
            'T_sup': m.addVars(horizon_steps + 1, lb=cfg.t_sup_lb, ub=cfg.t_sup_ub, name="T_sup"),
            'P_elec': m.addVars(horizon_steps, lb=0.0, ub=4.0, name="P_elec"),
            'Temp_Lift': m.addVars(horizon_steps, lb=-40.0, ub=40.0, name="Temp_Lift"),
            'T_room': {}, 'x_state': {}, 'Q_fan': {}, 'Slack_T': {},
            'Z_AC_out': self.Z_AC_15, 'Z_fan_out': self.Z_fan_15,
            'Slack_T_sup': m.addVars(horizon_steps, lb=0.0, name="Slack_T_sup"),
        })

        # McCormick box [b_L, b_U] for Q_fan = Z_fan * Max_Cooling -- exact at binary Z_fan,
        self.Q_FAN_BOUNDS = {
            r: (-abs(fans[r]['unified_slope_kW_per_C']) * (cfg.t_room_ub - cfg.t_sup_lb),
                abs(fans[r]['unified_slope_kW_per_C']) * (cfg.t_sup_ub - cfg.t_room_lb))
            for r in rooms
        }

        for r in rooms:
            nx = len(rc_sys[r]['A'])
            vd['x_state'][r] = m.addVars(nx, horizon_steps + 1, lb=-GRB.INFINITY, name=f"x_{r}")
            vd['T_room'][r] = m.addVars(horizon_steps, lb=cfg.t_room_lb, ub=cfg.t_room_ub, name=f"T_room_{r}")
            b_L, b_U = self.Q_FAN_BOUNDS[r]
            vd['Q_fan'][r] = m.addVars(horizon_steps, lb=b_L, ub=b_U, name=f"Q_fan_{r}")
            vd['Slack_T'][r] = m.addVars(horizon_steps, lb=0.0, ub=50.0, name=f"Slack_T_{r}")

    def _add_battery_state_vars(self) -> None:
        """SOC + charge/discharge + grid import/export. No binary
        forcing charge/discharge mutual exclusivity either: with feed-in <= import tariff
        always, doing both at once only wastes round-trip efficiency for zero benefit."""
        m, horizon_steps, cfg = self.m, self.horizon_steps, self.cfg
        capacity_kwh = self.battery['capacity_kwh']
        self.vars_dict.update({
            'SOC': m.addVars(horizon_steps + 1, lb=0.0, ub=capacity_kwh, name="SOC"),
            'P_charge': m.addVars(horizon_steps, lb=0.0, ub=cfg.battery_max_charge_kw, name="P_charge"),
            'P_discharge': m.addVars(horizon_steps, lb=0.0, ub=cfg.battery_max_discharge_kw, name="P_discharge"),
            # Unbounded -- no site fuse/breaker limit modeled yet.
            'P_grid_import': m.addVars(horizon_steps, lb=0.0, name="P_grid_import"),
            'P_grid_export': m.addVars(horizon_steps, lb=0.0, name="P_grid_export"),
            # Soft SOC-band violation amounts
            'Slack_SOC_Low': m.addVars(horizon_steps + 1, lb=0.0, name="Slack_SOC_Low"),
            'Slack_SOC_High': m.addVars(horizon_steps + 1, lb=0.0, name="Slack_SOC_High"),
        })

    def _add_initial_conditions(self) -> None:
        m, vd, params, rc_sys, rooms = self.m, self.vars_dict, self.params, self.rc_sys, self.rooms
        m.addConstr(vd['T_sup'][0] == params['Init_T_sup'])
        m.addConstr(vd['SOC'][0] == params['Init_SOC'])
        for r in rooms:
            for i in range(len(rc_sys[r]['A'])):
                m.addConstr(vd['x_state'][r][i, 0] == params['Init_x_state'][r][i])

    def _add_chiller_hysteresis_constraints(self) -> None:
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
        m, vd, params, cfg = self.m, self.vars_dict, self.params, self.cfg
        Z_AC_15, binary_blocks, block_size = self.Z_AC_15, self.binary_blocks, self.block_size
        M_TSUP_ON = cfg.m_tsup_on
        T_SUP_ON_THRESHOLD = cfg.t_sup_on_threshold
        M_TSUP_OFF = cfg.m_tsup_off
        T_SUP_OFF_THRESHOLD = cfg.t_sup_off_threshold

        Z_AC_prev = None
        for k in range(binary_blocks):
            t0 = k * block_size
            prev_z = Z_AC_prev if Z_AC_prev is not None else params['Init_Chiller_On']
            turn_on = m.addVar(lb=0.0, ub=1.0, name=f"Chiller_TurnOn_{k}")
            m.addConstr(turn_on >= Z_AC_15[k] - prev_z, name=f"Chiller_TurnOn_Def_{k}")
            m.addConstr(
                vd['T_sup'][t0] >= T_SUP_ON_THRESHOLD - M_TSUP_ON * (1 - turn_on),
                name=f"Chiller_On_Threshold_{k}"
            )
            for i in range(block_size):
                t = t0 + i
                m.addConstr(
                    vd['T_sup'][t] >= T_SUP_OFF_THRESHOLD - M_TSUP_OFF * (1 - Z_AC_15[k]),
                    name=f"Chiller_Off_Threshold_{k}_{i}"
                )
            Z_AC_prev = Z_AC_15[k]

    def _add_cooling_dynamics(self) -> None:
        """Per-5-min-step T_sup/power/room/fan physics. Everything the battery doesn't touch."""
        m, vd, params, cfg = self.m, self.vars_dict, self.params, self.cfg
        rooms, horizon_steps = self.rooms, self.horizon_steps
        block_size, binary_blocks = self.block_size, self.binary_blocks
        Z_AC_15, Z_fan_15, Q_FAN_BOUNDS = self.Z_AC_15, self.Z_fan_15, self.Q_FAN_BOUNDS

        plant = self.plant_physics['plant_thermal_model']  # tail-only: blended, Z_AC as linear feature
        plant_on = self.plant_physics['plant_thermal_model_on']  # head-only: exact regime, Z_AC=1
        plant_off = self.plant_physics['plant_thermal_model_off']  # head-only: exact regime, Z_AC=0
        power_bin = self.plant_physics['chiller_power_model']
        power_cont = self.plant_physics['chiller_power_model_continuous']
        fans = self.fans
        rc_sys = self.rc_sys

        M_POWER = cfg.m_power  # max physical electrical kW of the chiller, with margin over any observed peak
        Q_LOAD_CAP_LB = sum(Q_FAN_BOUNDS[r][0] for r in rooms)  # true worst-case total cooling

        # Big-M for the head's exact ON/OFF regime-switch, sized from the fitted
        # coefficients and variable bounds (tightest M that still guarantees correctness).
        _d_a = plant_on['Intercept'] - plant_off['Intercept']
        _d_t = plant_on['T_sup_current'] - plant_off['T_sup_current']
        _d_e = plant_on['EnvTmp'] - plant_off['EnvTmp']
        _d_q = plant_on['Total_Thermal_Load_kW'] - plant_off['Total_Thermal_Load_kW']
        _t_lo, _t_hi = sorted([_d_t * cfg.t_sup_lb, _d_t * cfg.t_sup_ub])
        _e_lo, _e_hi = sorted([_d_e * cfg.envtmp_bigm_lb, _d_e * cfg.envtmp_bigm_ub])
        _q_lo, _q_hi = sorted([_d_q * Q_LOAD_CAP_LB, _d_q * 0.0])
        _diff_max = _d_a + _t_hi + _e_hi + _q_hi
        _diff_min = _d_a + _t_lo + _e_lo + _q_lo
        M_TSUP = 1.1 * max(abs(_diff_max), abs(_diff_min))

        # Only exists for continuous-tail (room, t) pairs -- see the k>=binary_blocks branch
        # below. Populated into vars_dict directly so update_objective can penalize it.
        vd['Slack_Cross'] = {r: {} for r in rooms}

        for t in range(horizon_steps):
            k = t // block_size

            # Q_fan is bounded <=0 for every room in both head and tail (see the per-room
            # loop below), so this sum is already <=0 by construction -- no separate capped
            # auxiliary needed (a one-sided capped auxiliary previously let the solver fake
            # an arbitrarily more negative load whenever that helped).
            Total_Q_Load = gp.quicksum(vd['Q_fan'][r][t] for r in rooms)

            m.addConstr(vd['Temp_Lift'][t] == params['T_amb'][t] - vd['T_sup'][t])

            if k < binary_blocks:
                # Binary head: exact ON/OFF regime-switch (Z_AC_15[k] is a true 0/1 here, so
                # this Big-M selection has zero error).
                on_expr = (plant_on['Intercept'] +
                           plant_on['T_sup_current'] * vd['T_sup'][t] +
                           plant_on['EnvTmp'] * params['T_amb'][t] +
                           plant_on['Total_Thermal_Load_kW'] * Total_Q_Load)
                off_expr = (plant_off['Intercept'] +
                            plant_off['T_sup_current'] * vd['T_sup'][t] +
                            plant_off['EnvTmp'] * params['T_amb'][t] +
                            plant_off['Total_Thermal_Load_kW'] * Total_Q_Load)
                m.addConstr(vd['T_sup'][t + 1] <= on_expr + M_TSUP * (1 - Z_AC_15[k]), name=f"Tsup_On_Hi_{t}")
                m.addConstr(vd['T_sup'][t + 1] >= on_expr - M_TSUP * (1 - Z_AC_15[k]), name=f"Tsup_On_Lo_{t}")
                m.addConstr(vd['T_sup'][t + 1] <= off_expr + M_TSUP * Z_AC_15[k], name=f"Tsup_Off_Hi_{t}")
                m.addConstr(vd['T_sup'][t + 1] >= off_expr - M_TSUP * Z_AC_15[k], name=f"Tsup_Off_Lo_{t}")
            else:
                # Continuous tail: single blended model, Z_AC as a linear feature (value-
                # function approximation only -- never executed).
                m.addConstr(
                    vd['T_sup'][t + 1] == plant['Intercept'] +
                    plant['T_sup_current'] * vd['T_sup'][t] +
                    plant['Chiller_Command'] * Z_AC_15[k] +
                    plant['Total_Thermal_Load_kW'] * Total_Q_Load +
                    plant['EnvTmp'] * params['T_amb'][t],
                    name=f"Tsup_Tail_{t}"
                )
            m.addConstr(
                vd['T_sup'][t] + vd['Slack_T_sup'][t] >= params['T_sup_min'][t],
                name=f"T_sup_Min_Bound_{t}"
            )

            lift_lag1 = vd['Temp_Lift'][t - 1] if t > 0 else params['Init_Lift_Historical']
            delta_temp_lift = vd['Temp_Lift'][t] - lift_lag1

            if k < binary_blocks:
                # Binary head: active-only power model, standard Big-M mapping (exact
                # since Z_AC is pure binary here).
                P_active = (power_bin['active_intercept'] +
                            power_bin['Temp_Lift'] * vd['Temp_Lift'][t] +
                            power_bin['Delta_Temp_Lift'] * delta_temp_lift +
                            power_bin['Total_Thermal_Load'] * Total_Q_Load)
                m.addConstr(vd['P_elec'][t] >= power_bin["standby_kW"], name=f"P_Standby_{t}")
                m.addConstr(
                    vd['P_elec'][t] >= P_active - M_POWER * (1 - Z_AC_15[k]),
                    name=f"P_Active_Standard_{t}"
                )
            else:
                # Continuous tail: linear model with Z_AC as a feature.
                P_reg = (power_cont['active_intercept'] +
                         power_cont['Temp_Lift'] * vd['Temp_Lift'][t] +
                         power_cont['Delta_Temp_Lift'] * delta_temp_lift +
                         power_cont['Total_Thermal_Load'] * Total_Q_Load +
                         power_cont['Chiller_Command'] * Z_AC_15[k])
                m.addConstr(vd['P_elec'][t] >= P_reg, name=f"P_Reg_Floor_{t}")
                m.addConstr(vd['P_elec'][t] >= power_bin["standby_kW"], name=f"P_Standby_Cont_{t}")

            for r in rooms:
                m.addConstr(
                    vd['T_room'][r][t] - vd['Slack_T'][r][t] <= params['T_max'][r][t],
                    name=f"Comfort_Max_{r}_{t}"
                )
                m.addConstr(
                    vd['T_room'][r][t] + vd['Slack_T'][r][t] >= params['T_min'][r][t],
                    name=f"Comfort_Min_{r}_{t}"
                )

                A = np.atleast_2d(rc_sys[r]['A'])
                B = np.atleast_2d(rc_sys[r]['B'])
                C = np.atleast_2d(rc_sys[r]['C'])
                slope = fans[r]['unified_slope_kW_per_C']
                Max_Cooling = slope * (vd['T_room'][r][t] - vd['T_sup'][t])

                if k < binary_blocks:
                    # Binary head: McCormick for w=a*b, a=Z_fan[0,1], b=Max_Cooling[b_L,b_U].
                    # Exact at Zf in {0,1} regardless of box tightness -- this is what
                    # actually executes.
                    b_L, b_U = Q_FAN_BOUNDS[r]
                    Zf = Z_fan_15[r][k]
                    vd['Q_fan'][r][t].ub = 0.0
                    m.addConstr(vd['Q_fan'][r][t] >= Zf * b_L, name=f"Fan_McC_Lo1_{r}_{t}")
                    m.addConstr(vd['Q_fan'][r][t] >= Max_Cooling - b_U * (1 - Zf), name=f"Fan_McC_Lo2_{r}_{t}")
                    m.addConstr(vd['Q_fan'][r][t] <= Max_Cooling - b_L * (1 - Zf), name=f"Fan_McC_Hi1_{r}_{t}")
                    m.addConstr(vd['Q_fan'][r][t] <= Zf * b_U, name=f"Fan_McC_Hi2_{r}_{t}")
                else:
                    # Continuous tail: tie Q_fan directly to physics (never positive,
                    # never colder than what Max_Cooling genuinely allows), instead of a
                    # McCormick relaxation whose slack at fractional Zf is prone to
                    # hallucinated tail cooling.
                    vd['Q_fan'][r][t].ub = 0.0
                    slack_cross = m.addVar(lb=0.0, name=f"Slack_Cross_{r}_{t}")
                    vd['Slack_Cross'][r][t] = slack_cross
                    m.addConstr(vd['Q_fan'][r][t] >= Max_Cooling - slack_cross,
                                name=f"Q_fan_Physical_Lo_{r}_{t}")

                u_t = [params['T_amb'][t], vd['Q_fan'][r][t], params['Solar'][t]]
                for i in range(A.shape[0]):
                    m.addConstr(vd['x_state'][r][i, t + 1] ==
                                gp.quicksum(A[i, j] * vd['x_state'][r][j, t] for j in range(A.shape[1])) +
                                gp.quicksum(B[i, j] * u_t[j] for j in range(B.shape[1])))
                m.addConstr(vd['T_room'][r][t] == gp.quicksum(
                    C[0, j] * vd['x_state'][r][j, t] for j in range(C.shape[1])))

    def _add_battery_dynamics(self) -> None:
        """SOC update + point-of-common-coupling (grid) balance."""
        m, vd, params, cfg = self.m, self.vars_dict, self.params, self.cfg
        Z_fan_15, rooms, horizon_steps, block_size = self.Z_fan_15, self.rooms, self.horizon_steps, self.block_size
        eta_c, eta_d, dt_h = self.battery['efficiency_charge'], self.battery['efficiency_discharge'], self.dt_h

        for t in range(horizon_steps):
            k = t // block_size
            Fan_Power_Total = gp.quicksum(cfg.fan_power_kw * Z_fan_15[r][k] for r in rooms)

            m.addConstr(
                vd['SOC'][t + 1] == vd['SOC'][t]
                + (eta_c * vd['P_charge'][t] - vd['P_discharge'][t] / eta_d) * dt_h,
                name=f"SOC_Dynamics_{t}"
            )
            m.addConstr(
                vd['P_grid_import'][t] - vd['P_grid_export'][t] ==
                vd['P_elec'][t] + Fan_Power_Total
                + vd['P_charge'][t] - vd['P_discharge'][t]
                + params['Load_forecast'][t] - params['PV_forecast'][t],
                name=f"Grid_Balance_{t}"
            )
            # Physical export cap: a grid-tie connection can only push out what's actually
            # generated/discharged on-site, never grid power just bought.
            m.addConstr(
                vd['P_grid_export'][t] <= params['PV_forecast'][t] + vd['P_discharge'][t],
                name=f"Export_Physical_Cap_{t}"
            )

    def _add_soc_band_constraints(self) -> None:
        """Soft SOC band at every step, with a higher 50% floor on the terminal
        SOC -- slack-penalized in update_objective, not a hard bound, so a genuinely
        necessary excursion stays feasible rather than blowing up the solve."""
        m, vd, cfg = self.m, self.vars_dict, self.cfg
        horizon_steps, capacity_kwh = self.horizon_steps, self.battery['capacity_kwh']
        for t in range(horizon_steps + 1):
            low_frac = cfg.soc_terminal_min_frac if t == horizon_steps else cfg.soc_low_frac
            m.addConstr(
                vd['SOC'][t] + vd['Slack_SOC_Low'][t] >= low_frac * capacity_kwh,
                name=f"SOC_Low_{t}"
            )
            m.addConstr(
                vd['SOC'][t] - vd['Slack_SOC_High'][t] <= cfg.soc_high_frac * capacity_kwh,
                name=f"SOC_High_{t}"
            )

    def _add_battery_ramp_vars(self) -> None:
        """|ΔP_charge|/|ΔP_discharge| between consecutive 5-min steps, via the standard
        two-sided linear hinge -- penalized in update_objective (see
        MPCConfig.battery_ramp_cost_eur_per_kw)."""
        m, vd, horizon_steps = self.m, self.vars_dict, self.horizon_steps
        ramp_charge = m.addVars(horizon_steps - 1, lb=0.0, name="Ramp_Charge")
        ramp_discharge = m.addVars(horizon_steps - 1, lb=0.0, name="Ramp_Discharge")
        for t in range(1, horizon_steps):
            i = t - 1
            m.addConstr(ramp_charge[i] >= vd['P_charge'][t] - vd['P_charge'][t - 1],
                        name=f"Ramp_Charge_Up_{t}")
            m.addConstr(ramp_charge[i] >= vd['P_charge'][t - 1] - vd['P_charge'][t],
                        name=f"Ramp_Charge_Down_{t}")
            m.addConstr(ramp_discharge[i] >= vd['P_discharge'][t] - vd['P_discharge'][t - 1],
                        name=f"Ramp_Discharge_Up_{t}")
            m.addConstr(ramp_discharge[i] >= vd['P_discharge'][t - 1] - vd['P_discharge'][t],
                        name=f"Ramp_Discharge_Down_{t}")
        vd['Ramp_Charge'] = ramp_charge
        vd['Ramp_Discharge'] = ramp_discharge

    # ------------------------------------------------------------------
    # Objective + solve
    # ------------------------------------------------------------------

    def update_objective(self, new_forecasts: dict) -> None:
        m, vd, cfg = self.m, self.vars_dict, self.cfg
        rooms, horizon_steps, dt_h = self.rooms, self.horizon_steps, self.dt_h
        obj = gp.LinExpr()

        sell_tariffs = new_forecasts.get("sell_tariffs")
        for t in range(horizon_steps):
            tariff = new_forecasts["tariffs"][t]
            sell_price = sell_tariffs[t] if sell_tariffs is not None else cfg.battery_feed_in_tariff_eur_kwh
            obj += vd["P_grid_import"][t] * tariff * dt_h
            obj -= vd["P_grid_export"][t] * sell_price * dt_h

        for r in rooms:
            slack_t_weight = cfg.slack_t_weight_override.get(r, cfg.slack_t_weight_default)
            for t in range(horizon_steps):
                obj += vd["Slack_T"][r][t] * slack_t_weight

        for t in range(horizon_steps):
            obj += vd["Slack_T_sup"][t] * cfg.slack_t_sup_weight

        # Soft SOC band penalty (10-90%, 50% terminal floor -- see
        # _add_soc_band_constraints and slack_soc_weight_eur_kwh).
        for t in range(horizon_steps + 1):
            obj += (vd["Slack_SOC_Low"][t] + vd["Slack_SOC_High"][t]) * cfg.slack_soc_weight_eur_kwh

        # Ramp smoothing -- see battery_ramp_cost_eur_per_kw.
        for t in range(horizon_steps - 1):
            obj += (vd["Ramp_Charge"][t] + vd["Ramp_Discharge"][t]) * cfg.battery_ramp_cost_eur_per_kw

        # Continuous-tail Q_fan's physical lower-bound slack (see slack_cross_weight_default/
        # slack_cross_weight_override) -- only exists for (room, t) pairs in the tail.
        for r in rooms:
            slack_cross_weight = cfg.slack_cross_weight_override.get(r, cfg.slack_cross_weight_default)
            for t in vd["Slack_Cross"][r]:
                obj += vd["Slack_Cross"][r][t] * slack_cross_weight

        m.setObjective(obj, GRB.MINIMIZE)

    def _preactivate_lagged_fans(self, chiller_cmd: int, fan_cmds: dict) -> dict:
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
        vd = self.vars_dict
        t_sup_now = vd['T_sup'][0].X
        for r in self.rooms:
            if (fan_cmds[r] == 0 and vd['T_room'][r][0].X < t_sup_now and
                    round(vd['Z_fan_out'][r][1].X) == 1):
                fan_cmds[r] = 1
        return fan_cmds

    def _extract_optimal_action(self) -> dict:
        vd, rooms, holdover_blocks = self.vars_dict, self.rooms, self.holdover_blocks
        chiller_cmd = round(vd['Z_AC_out'][0].X)
        fan_cmds = {r: round(vd['Z_fan_out'][r][0].X) for r in rooms}
        fan_cmds = self._preactivate_lagged_fans(chiller_cmd, fan_cmds)
        return {
            'Chiller_Command': chiller_cmd,
            'Fan_Commands': fan_cmds,
            # Battery command nets to a signed power (+discharge, -charge), not
            # rounded like Chiller/Fan -- it's a genuinely continuous decision.
            'Battery_Power_kW': vd['P_discharge'][0].X - vd['P_charge'][0].X,
            # Blocks k=1..holdover_blocks stored so run_mpc.py can coast through
            # up to holdover_blocks consecutive solve failures.
            'Holdover_Chiller_Commands': [
                round(vd['Z_AC_out'][k].X) for k in range(1, holdover_blocks + 1)
            ],
            'Holdover_Fan_Commands': [
                {r: round(vd['Z_fan_out'][r][k].X) for r in rooms}
                for k in range(1, holdover_blocks + 1)
            ],
            'Holdover_Battery_Power_kW': [
                vd['P_discharge'][k * 3].X - vd['P_charge'][k * 3].X
                for k in range(1, holdover_blocks + 1)
            ],
        }

    def step(self, new_initial_states: dict, new_forecasts: dict) -> dict | None:
        m, params, rooms = self.m, self.params, self.rooms

        # 1. Forecasts
        for t in range(self.horizon_steps):
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
        params['Init_Lift_Historical'].lb = params['Init_Lift_Historical'].ub = \
            new_initial_states['Temp_Lift_historical']
        chiller_on_prev = float(new_initial_states.get('Chiller_On_prev', 0.0))
        params['Init_Chiller_On'].lb = params['Init_Chiller_On'].ub = chiller_on_prev
        params['Init_SOC'].lb = params['Init_SOC'].ub = new_initial_states['SOC_current']

        for r in params['Init_x_state']:
            for i in range(len(params['Init_x_state'][r])):
                val = new_initial_states['x_state_current'][r][i]
                params['Init_x_state'][r][i].lb = params['Init_x_state'][r][i].ub = val

        print("\n--- Solving Parametric MPC for the Next Control Step ---")

        self.update_objective(new_forecasts)

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
            return self._extract_optimal_action()
        else:
            gap_str = f"{m.MIPGap:.4f}" if m.SolCount > 0 else "N/A"
            print(f"Solver rejected: status={m.Status}, SolCount={m.SolCount}, gap={gap_str}")
            return None
