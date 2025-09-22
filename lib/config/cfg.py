import yaml
import os
import os.path as osp
import sys
from argparse import ArgumentParser
from importlib import import_module
from addict import Dict
sys.path.insert(0, './../../')


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError("'{}' object has no attribute '{}'".format(
                self.__class__.__name__, name))
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


def add_args(parser, cfg, prefix=''):
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument('--' + prefix + k)
        elif isinstance(v, int):
            parser.add_argument('--' + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument('--' + prefix + k, type=float)
        elif isinstance(v, bool):
            parser.add_argument('--' + prefix + k, action='store_true')
        elif isinstance(v, dict):
            add_args(parser, v, k + '.')
        else:
            print('connot parse key {} of type {}'.format(prefix + k, type(v)))
    return parser


class Config(object):
    @staticmethod
    def fromfile(filename):
        # filename = osp.abspath(osp.expanduser(filename))
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
            # cfg_dict = mmcv.load(filename)
        else:
            raise IOError('Only py/yml/yaml/json type are supported now!')
        return Config(cfg_dict, filename=filename)

    @staticmethod
    def auto_argparser(description=None):
        """Generate argparser from config file automatically (experimental)
        """
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument('config', help='config file path')
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.from_file(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument('config', help='config file path')
        add_args(parser, cfg)
        return parser, cfg

    def __init__(self, cfg_dict=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but got {}'.format(
                type(cfg_dict)))

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
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
        return 'Config (path: {}): {}'.format(self.filename,
                                              self._cfg_dict.__repr__())

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):\
        return iter(self._cfg_dict)
    
    def _to_plain_dict(self, obj=None):
        """
        Recursively convert ConfigDict / dict / list into a pure Python dict/list
        so that yaml.safe_dump doesn’t introduce any Addict-specific tags.
        """
        if obj is None:
            obj = self._cfg_dict

        if isinstance(obj, ConfigDict) or isinstance(obj, dict):
            return {
                key: self._to_plain_dict(val)
                for key, val in obj.items()
            }
        elif isinstance(obj, list):
            return [self._to_plain_dict(val) for val in obj]
        else:
            # primitive (str, int, float, bool, etc.) – yaml can handle it
            return obj

    def to_dict(self):
        """Return the full config as a nested plain `dict`."""
        return self._to_plain_dict()

    def save_config(self, filename):
        """
        Save the current configuration (including any programmatic overrides)
        to `filename` in YAML format.
        """
        # ensure directory exists
        dirname = osp.dirname(filename)
        if dirname and not osp.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        # convert to pure-Python dict
        cfg_dict = self.to_dict()

        # dump as YAML
        with open(filename, 'w') as f:
            yaml.safe_dump(cfg_dict, f, default_flow_style=False)

