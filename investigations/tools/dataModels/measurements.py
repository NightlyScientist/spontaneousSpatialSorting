import importlib.machinery
import numpy as np
import numba as nb
import types
import os, sys
import os.path
import importlib
from tools.dataAPI.datamodel import DataModel
from tools.dataModels.grainMeasures import Grains
from tools.dataModels.alignmentMeasures import Alignments
from tools.dataModels.geneticMeasures import Genetics
from tools.dataModels.flowMeasures import FlowFields


def import_source_file(fname: str, modname: str) -> "types.ModuleType":
    spec = importlib.util.spec_from_file_location(modname, fname)
    if spec is None:
        raise ImportError(f"Could not load spec for module '{modname}' at: {fname}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try: 
        spec.loader.exec_module(module)
    except FileNotFoundError as e:
        raise ImportError(f"{e.strerror}: {fname}") from e
    return module

class PluginMeta(type):
    def __new__(cls, name, bases, dct):

        possible_paths = dct.get("plugins", None).split("/")
        pluginPath = os.path.join(str(os.getcwd()), "/".join(possible_paths))

        while not os.path.exists(
            os.path.join(str(os.getcwd()), "/".join(possible_paths))
        ):
            possible_paths = possible_paths[1:]
            pluginPath = os.path.join(str(os.getcwd()), "/".join(possible_paths))

        modules = []
        for filename in os.listdir(pluginPath):
            if filename.endswith(".py") and filename != "__init__.py":
                path = os.path.join(pluginPath, filename)
                module = import_source_file(path, filename)
                modules.append(module)

        for module in modules:
            for name in dir(module):
                function = getattr(module, name)
                if isinstance(function, types.FunctionType) or isinstance(
                    function, staticmethod
                ):
                    dct[function.__name__] = function
        return type.__new__(cls, name, bases, dct)


class Base(metaclass=PluginMeta):
    plugins = "investigations/tools/dataModels/measures"

    def check():
        print("Base class")


class Measurements(Base, Grains, Alignments, Genetics, FlowFields):
    plugins = "investigations/tools/dataModels/measures"

    def check():
        print("Measurements class")
