from pathlib import Path
import yaml

default_path = Path(__file__).parent / 'config.yaml'

with open(default_path) as f:
    default_configs = yaml.load(f, Loader=yaml.FullLoader)
