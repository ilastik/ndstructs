from abc import ABC, abstractmethod
import itertools
from collections.abc import Mapping as BaseMapping
import json
import inspect
import re
from typing import Dict, Callable, Any
import numpy as np


def get_constructor_params(klass):
    args_kwargs = sorted([inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD])
    if not inspect.isclass(klass):  # maybe some other function was passed in
        return inspect.signature(klass).parameters.items()

    for kls in klass.__mro__:  # move up heritance hierarchy until some constructor is not *args **kwargs
        params = inspect.signature(kls).parameters
        param_kinds = sorted(p.kind for p in params.values())
        if param_kinds != args_kwargs:
            return params.items()
    raise Exception("Unexpected signature for klass {klass}: {params}")


def hint_to_deserializer(type_hint):
    if re.search(r"^(typing\.)?(List|Iterable|Tuple|Sequence)\[", str(type_hint)):
        item_type = type_hint.__args__[0]
        return lambda value: [from_json_data(item_type, v) for v in value]
    elif re.search(r"^(typing\.)?Union\[", str(type_hint)):
        exceptions = []

        def wrapper(value):
            if value is None and None.__class__ in type_hint.__args__:
                return None
            for item_type in type_hint.__args__:
                if item_type == None.__class__:
                    continue
                try:
                    return from_json_data(item_type, value)
                except Exception as e:
                    exceptions.append(e)
            raise ValueError("Could not deserialize value {value}: {exceptions}")

        return wrapper
    raise NotImplementedError(f"Don't know how to unwrap {type_hint}")


def is_type_hint(obj) -> bool:
    return hasattr(obj, "__origin__")


def from_json_data(cls, data, initOnly: bool = False):
    if is_type_hint(cls):
        deserializer = hint_to_deserializer(cls)
        return deserializer(data)
    if inspect.isclass(cls) and isinstance(data, cls):
        return data
    if (
        not initOnly
        and hasattr(cls, "from_json_data")
        and cls.from_json_data.__qualname__ != "JsonSerializable.from_json_data"
    ):
        return cls.from_json_data(data)
    if isinstance(data, BaseMapping):
        data = data.copy()
        assert "__class__" not in data or data["__class__"] == cls.__name__
        this_params = {}
        for name, parameter in get_constructor_params(cls):
            type_hint = parameter.annotation
            assert type_hint != inspect._empty, f"Missing type hint for param {name}"
            if name not in data:  # might be a missing optional
                continue
            value = data.pop(name)
            if hasattr(type_hint, "from_json_data") and not isinstance(value, type_hint):
                this_params[name] = type_hint.from_json_data(value)
            else:
                this_params[name] = from_json_data(type_hint, value)
        if len(data) > 0:
            print(f"WARNING: Unused arguments when deserializing {cls.__name__}: {data}")
        return cls(**this_params)
    return cls(data)


int_classes = tuple(
    [int]
    + [
        getattr(np, "".join(type_parts))
        for type_parts in itertools.product(["u", ""], ["int"], ["8", "16", "32", "64"])
    ]
)
float_classes = tuple([float] + [np.float, np.float16, np.float32, np.float64, np.float128])


def to_json_data(value, referencer: Callable[[Any], str] = lambda obj: None):
    if isinstance(value, (str, None.__class__)):
        return value
    ref = referencer(value)
    if ref is not None:
        return ref
    if hasattr(value, "to_json_data"):
        return value.to_json_data(referencer=referencer)
    if isinstance(value, int_classes):
        return int(value)
    if isinstance(value, float_classes):
        return float(int)
    if isinstance(value, (tuple, list, np.ndarray)):
        return [to_json_data(v, referencer=referencer) for v in value]
    if isinstance(value, BaseMapping):
        return {k: to_json_data(v, referencer=referencer) for k, v in value.items()}


class JsonSerializable(ABC):
    def to_json_data(self, referencer: Callable[[Any], str] = lambda obj: None):
        out_dict = {"__class__": self.__class__.__name__}
        for name, parameter in inspect.signature(self.__class__).parameters.items():
            out_dict[name] = to_json_data(getattr(self, name), referencer=referencer)
        return out_dict

    @classmethod
    def from_json(cls, data: str):
        return cls.from_json_data(json.loads(data))

    @classmethod
    def from_json_data(cls, data: dict):
        return from_json_data(cls, data)
