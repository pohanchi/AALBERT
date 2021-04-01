



import os
import pathlib
import importlib

import torch
dependencies = ['torch']

search_root = os.path.dirname(__file__)
hubconfs = [str(p) for p in pathlib.Path(search_root).rglob('hubconf.py')]
hubconfs = [os.path.relpath(p, search_root) for p in hubconfs]
hubconfs.remove('hubconf.py')  # remove the root hubconf.py

for hubconf in hubconfs:
    module_name = '.'.join(str(hubconf).split('.')[:-1]).replace('/', '.')
    try:
        _module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        print(f'[hubconf] can not import {module_name}: {str(e)}... Pass.')
        continue

    for variable_name in dir(_module):
        _variable = getattr(_module, variable_name)
        if callable(_variable) and variable_name[0] != '_':
            globals()[variable_name] = _variable