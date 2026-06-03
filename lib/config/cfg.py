import yaml
import os
import os.path as osp
import sys
from argparse import ArgumentParser
from importlib import import_module
from addict import Dict
sys.path.insert(0, './../../')

_MISSING = object()

class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError as e:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from e
        return value


def add_args(parser, cfg, prefix=''):
    for k, v in cfg.items():
        arg_name = '--' + prefix + k

        if isinstance(v, bool):
            parser.add_argument(arg_name, type=yaml.safe_load)
        elif isinstance(v, int):
            parser.add_argument(arg_name, type=int)
        elif isinstance(v, float):
            parser.add_argument(arg_name, type=float)
        elif isinstance(v, str):
            parser.add_argument(arg_name, type=str)
        elif v is None:
            parser.add_argument(arg_name, type=yaml.safe_load)
        elif isinstance(v, dict):
            add_args(parser, v, k + '.')
        elif isinstance(v, list):
            parser.add_argument(arg_name, type=yaml.safe_load)
        else:
            print(f'cannot parse key {prefix + k} of type {type(v)}')
    return parser

class Config(object):
    @staticmethod
    def _wrap(obj):
        """
        Recursively convert plain dicts into ConfigDict,
        and recurse into lists/tuples. None stays None.
        """
        if isinstance(obj, ConfigDict):
            return obj
        elif isinstance(obj, dict):
            wrapped = ConfigDict()
            for k, v in obj.items():
                wrapped[k] = Config._wrap(v)
            return wrapped
        elif isinstance(obj, list):
            return [Config._wrap(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(Config._wrap(v) for v in obj)
        else:
            return obj

    @staticmethod
    def fromfile(filename):
        if filename.endswith('.py'):
            module_name = osp.basename(filename)[:-3]
            if '.' in module_name:
                raise ValueError('Dots are not allowed in config file path.')
            config_dir = osp.dirname(filename)
            sys.path.insert(0, config_dir)
            mod = import_module(module_name)
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            }
        elif filename.endswith(('.yml', '.yaml')):
            with open(filename, 'r') as file:
                cfg_dict = yaml.safe_load(file)
            if cfg_dict is None:
                cfg_dict = {}
        else:
            raise IOError('Only py/yml/yaml/json type are supported now!')

        return Config(cfg_dict, filename=filename)

    @staticmethod
    def auto_argparser(description=None):
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument('config', help='config file path')
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.fromfile(cfg_file)

        parser = ArgumentParser(description=description)
        parser.add_argument('config', help='config file path')
        add_args(parser, cfg)
        return parser, cfg

    def __init__(self, cfg_dict=None, filename=None):
        if cfg_dict is None:
            cfg_dict = {}
        elif not isinstance(cfg_dict, dict):
            raise TypeError(f'cfg_dict must be a dict, but got {type(cfg_dict)}')

        super(Config, self).__setattr__('_cfg_dict', self._wrap(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)

        if filename:
            with open(filename, 'r') as f:
                super(Config, self).__setattr__('_text', f.read())
        else:
            super(Config, self).__setattr__('_text', '')

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def __repr__(self):
        return f'Config (path: {self.filename}): {self._cfg_dict!r}'

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict[name]

    def __setattr__(self, name, value):
        self._cfg_dict.__setattr__(name, self._wrap(value))

    def __setitem__(self, name, value):
        self._cfg_dict.__setitem__(name, self._wrap(value))

    def __iter__(self):
        return iter(self._cfg_dict)

    def _to_plain_dict(self, obj=_MISSING, _visited=None):
        """
        Recursively convert ConfigDict / dict / list / tuple into plain Python
        containers. Detect cycles explicitly to avoid infinite recursion.
        """
        if obj is _MISSING:
            obj = self._cfg_dict
            
        if _visited is None:
            _visited = set()

        if isinstance(obj, (ConfigDict, dict, list, tuple)):
            obj_id = id(obj)
            if obj_id in _visited:
                raise ValueError(
                    "Cyclic reference detected while converting config to a plain dict. "
                    "Some non-config object may have been inserted into cfg."
                )
            _visited.add(obj_id)

            try:
                if isinstance(obj, (ConfigDict, dict)):
                    return {
                        key: self._to_plain_dict(val, _visited)
                        for key, val in obj.items()
                    }
                elif isinstance(obj, list):
                    return [self._to_plain_dict(val, _visited) for val in obj]
                elif isinstance(obj, tuple):
                    return tuple(self._to_plain_dict(val, _visited) for val in obj)
            finally:
                _visited.remove(obj_id)

        return obj

    def to_dict(self):
        return self._to_plain_dict()

    def save_config(self, filename):
        dirname = osp.dirname(filename)
        if dirname and not osp.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        cfg_dict = self.to_dict()

        with open(filename, 'w') as f:
            yaml.safe_dump(
                cfg_dict,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True
            )