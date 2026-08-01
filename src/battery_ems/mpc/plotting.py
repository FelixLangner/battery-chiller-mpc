"""Plot MPC planning."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from battery_ems.controllers.gurobipy_mpc import ParametricMPC


def plot_mpc_results(mpc: ParametricMPC, forecasts: dict, save_path=None) -> None:
    """Two separate figures -- cooling (chiller/temperatures) and battery/PV/grid
    economics are largely independent stories, and one tall stack made each panel
    harder to read."""
    print("\nExtracting optimization results for visualization...")
    vars_dict, params, cfg = mpc.vars_dict, mpc.params, mpc.cfg
    horizon_steps, BLOCK_SIZE, BINARY_BLOCKS = mpc.horizon_steps, mpc.block_size, mpc.binary_blocks
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
        else [cfg.battery_feed_in_tariff_eur_kwh] * horizon_steps

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
