import numpy as np

class Battery:
    """
    Emulator for a residential battery energy storage system.
    Tracks the SOC.
    """

    def __init__(
            self,
            capacity_kwh: float = 7.0,
            power_charge_max_kw: float = 1.0,
            power_discharge_max_kw: float = 1.0,
            efficiency_charge: float = 0.9,
            efficiency_discharge: float = 0.9,
            initial_soc_kwh: float = 0.0,
    ) -> None:

        # 1. Static Physical Parameters
        self.capacity_kwh = capacity_kwh
        self.power_charge_max_kw = power_charge_max_kw
        self.power_discharge_max_kw = power_discharge_max_kw
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge

        # 2. Dynamic State Variables
        self.soc_kwh = initial_soc_kwh

    def step(self, power_kw: float = 0.0, delta_t_hours: float = 0.0) -> None:
        """
            Update the state of charge (SOC) based on the charging and discharging power.

            The SOC is updated as follows:
            - When charging, the effective energy added to the battery is reduced by the charging efficiency.
            - When discharging, the effective energy removed from the battery is increased by the discharging efficiency.

            Physical limits apply:
               1. SOC is also constrained to be between 0 and the battery capacity.
               2. The charging and discharging power cannot exceed their respective maximum limits.
        """
        if power_kw >= 0:  # charging
            self.soc_kwh += power_kw * self.efficiency_charge * delta_t_hours
        else:              # discharging
            self.soc_kwh += power_kw * delta_t_hours / self.efficiency_discharge

        self.soc_kwh = np.clip(self.soc_kwh, 0, self.capacity_kwh)
