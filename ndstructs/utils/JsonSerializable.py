from abc import ABC, abstractmethod
from collections.abc import Mapping
import json
import inspect
import re
from typing import Dict


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


def hint_to_wrapper(type_hint):
    if re.search(r"^(typing\.)?(List|Iterable|Tuple)\[", str(type_hint)):
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


def from_json_data(cls, data):
    if is_type_hint(cls):
        wrapper = hint_to_wrapper(cls)
        return wrapper(data)
    if inspect.isclass(cls) and isinstance(data, cls):
        return data
    if isinstance(data, Mapping):
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


class JsonSerializable(ABC):
    @property
    def json_data(self):
        out_dict = {"__class__": self.__class__.__name__}
        for name, parameter in inspect.signature(self.__class__).parameters.items():
            value = getattr(self, name)
            if isinstance(value, JsonSerializable):
                value = value.json_data
            out_dict[name] = value
        return out_dict

    @classmethod
    def jsonify(cls, json_data: Dict) -> str:
        d = json_data
        out_data = d.copy()
        for k, v in d.items():
            if isinstance(v, JsonSerializable):
                out_data[k] = v.json_data
        return json.dumps(out_data)

    def to_json(self) -> str:
        return JsonSerializable.jsonify(self.json_data)

    @classmethod
    def from_json(cls, data: str):
        return cls.from_json_data(json.loads(data))

    @classmethod
    def from_json_data(cls, data: dict):
        # import pydevd; pydevd.settrace()
        return from_json_data(cls, data)
