import yaml

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def override_config_with_args(config, args):
    """Overrides configuration with command-line arguments."""
    for key, value in vars(args).items():
        if value is not None and key in config:
            config[key] = value
        elif value is not None and key in config['training']:
            config['training'][key] = value
        elif value is not None: # For special cases like checkpoint path
             config[key] = value
    return config
