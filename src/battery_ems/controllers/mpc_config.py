"""
MPC config file
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MPCConfig:
    # --- Cooling comfort ---
    slack_t_weight_default: float = 10.0  # €/°C-equivalent comfort slack penalty
    slack_t_weight_override: dict = field(default_factory=lambda: {"room_5": 1.0})
    # room_5's FCU is undersized for its thermal mass, so its weight is
    # lowered to trade a little comfort margin for flexibility.
    slack_t_sup_weight: float = 10.0  # penalty on breaching T_sup_min

    t_room_lb: float = 18.0
    t_room_ub: float = 28.0
    t_sup_lb: float = 5.0
    t_sup_ub: float = 40.0

    # Slack to allow T_sup > T_room (continuous-tail Q_fan physical lower bound)
    slack_cross_weight_default: float = 0.01
    slack_cross_weight_override: dict = field(default_factory=lambda: {"room_5": 1.0})

    # --- Battery / PV ---
    fan_power_kw: float = 0.05  # assumed FCU electrical draw, not a measured value

    # Power rating -- capacity/efficiency come from mpc_battery_coefs.json,
    # but that file has no power rating, so it's set here directly.
    battery_max_charge_kw: float = 4.5
    battery_max_discharge_kw: float = 4.5

    battery_feed_in_tariff_eur_kwh: float = 0.08

    soc_low_frac: float = 0.10
    soc_high_frac: float = 0.90
    soc_terminal_min_frac: float = 0.50
    slack_soc_weight_eur_kwh: float = 1.0  # €/kWh-step, deliberately firm relative to tariff scale

    # Small cost on |ΔP_charge|/|ΔP_discharge| between consecutive steps --
    # breaks ties among equally-priced schedules toward smooth ones.
    battery_ramp_cost_eur_per_kw: float = 0.01

    # --- Physical envelope for outdoor temperature, used only to size the
    # T_sup regime-switch Big-M (see _add_cooling_dynamics) ---
    envtmp_bigm_lb: float = -15.0
    envtmp_bigm_ub: float = 45.0

    # --- Control blocking / binary head ---
    block_size: int = 3  # 5-min physics steps per 15-min control block
    binary_blocks: int = 8  # first 2h of the horizon kept strictly binary
    # If consecutive solves fail, hold the previous solve's plan for this
    # many 15-min blocks before falling back to a hard setpoint. Must stay
    # < binary_blocks so the holdover commands are true binaries.
    holdover_blocks: int = 4

    # --- Chiller hysteresis
    m_tsup_on: float = 20.0
    t_sup_on_threshold: float = 19.0
    m_tsup_off: float = 12.0
    t_sup_off_threshold: float = 6.0

    # --- Chiller power model ---
    m_power: float = 4.0  # max physical electrical kW of the chiller, with margin

    # --- Physics step size (also used by update_objective for the €/step cost) ---
    dt_h_hours: float = 5.0 / 60.0  # hours per 5-min physics step

    # --- Gurobi solver params ---
    solver_output_flag: int = 0
    solver_mip_gap: float = 0.01
    solver_mip_gap_abs: float = 0.005
    solver_mip_focus: int = 1
    solver_heuristics: float = 0.3
    solver_presolve: int = 2
    solver_time_limit_s: float = 120
