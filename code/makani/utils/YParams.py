from ruamel.yaml import YAML
import json
import re


class ParamsBase:
    """Convenience wrapper around a dictionary

    Allows referring to dictionary items as attributes, and tracking which
    attributes are modified.
    """

    def __init__(self):
        self._original_attrs = None
        self.params = {}
        self._original_attrs = list(self.__dict__)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        self.__setattr__(key, val)

    def __contains__(self, key):
        return key in self.params

    def get(self, key, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return self.params.get(key, default)

    def to_dict(self):
        new_attrs = {key: val for key, val in vars(self).items() if key not in self._original_attrs}
        return {**self.params, **new_attrs}

    @staticmethod
    def from_json(path: str) -> "ParamsBase":
        with open(path) as f:
            c = json.load(f)
        params = ParamsBase()
        params.update_params(c)
        return params

    def update_params(self, config):
        for key, val in config.items():
            if val == "None":
                val = None
            self.params[key] = val
            self.__setattr__(key, val)


class YParams(ParamsBase):
    def __init__(self, yaml_filename, config_name, print_params=False):
        """Open parameters stored with ``config_name`` in the yaml file ``yaml_filename``"""
        super().__init__()
        self._yaml_filename = yaml_filename
        self._config_name = config_name

        if print_params:
            print("------------------ Configuration ------------------")

        with open(yaml_filename) as _file:
            d = YAML().load(_file)[config_name]

        d = self._resolve_interpolations(d)
        self.update_params(d)

        if print_params:
            for key, val in d.items():
                print(key, val)
            print("---------------------------------------------------")

    def _resolve_interpolations(self, config):
        """
        Resolve ${key} references using other keys in the same config block.
        Supports repeated/nested substitutions such as:
            base_dir: /a/b
            dataset_name: physical
            train_data_path: ${base_dir}/${dataset_name}/train
        """
        pattern = re.compile(r"\$\{([^}]+)\}")

        def resolve_one(value, mapping):
            if not isinstance(value, str):
                return value

            while True:
                matches = pattern.findall(value)
                if not matches:
                    break

                new_value = value
                for ref_key in matches:
                    if ref_key not in mapping:
                        raise KeyError(
                            f"Interpolation key '{ref_key}' not found while resolving '{value}'"
                        )

                    ref_val = mapping[ref_key]
                    if not isinstance(ref_val, str):
                        ref_val = str(ref_val)

                    new_value = new_value.replace(f"${{{ref_key}}}", ref_val)

                if new_value == value:
                    break
                value = new_value

            return value

        resolved = dict(config)

        # Iterate until stable so nested references also resolve
        for _ in range(20):
            changed = False
            for key, val in resolved.items():
                new_val = resolve_one(val, resolved)
                if new_val != val:
                    resolved[key] = new_val
                    changed = True
            if not changed:
                break

        # Optional safety check: fail if anything is still unresolved
        for key, val in resolved.items():
            if isinstance(val, str) and "${" in val:
                raise ValueError(f"Unresolved interpolation for key '{key}': {val}")

        return resolved

    def log(self, logger):
        logger.info("------------------ Configuration ------------------")
        logger.info("Configuration file: " + str(self._yaml_filename))
        logger.info("Configuration name: " + str(self._config_name))
        for key, val in self.to_dict().items():
            logger.info(str(key) + " " + str(val))
        logger.info("---------------------------------------------------")