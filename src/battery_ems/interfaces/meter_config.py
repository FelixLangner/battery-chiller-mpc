import yaml
from pathlib import Path

def load_meter_config(file_name: str = "meters_demo.yaml") -> dict:
    """
    Locates and loads the YAML configuration file from the project root.
    """
    # Find the root of the project by looking two directories up from this file
    # (src/battery_ems/interfaces -> src/battery_ems -> src -> root)
    current_dir = Path(__file__).parent
    root_dir = current_dir.parent.parent.parent
    yaml_path = root_dir / file_name

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {yaml_path}. "
            "Expected meters_demo.yaml at the project root (committed, synthetic-only)."
        )

    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)
