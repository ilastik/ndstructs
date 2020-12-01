from abc import ABC, abstractmethod
import itertools
from collections.abc import Mapping as BaseMapping
import json
import inspect
import re
from typing import Dict, Callable, Any, Optional, TypeVar, Type, cast
import numpy as np
import uuid


def get_constructor_params(klass):
    args_kwargs = sorted([inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD])
    if not inspect.isclass(klass):  # maybe some other function was passed in
        return inspect.signature(klass).parameters.items()

    for kls in klass.__mro__:  # move up heritance hierarchy until some constructor is not *args **kwargs
        params = inspect.signature(kls).parameters
        param_kinds = sorted(p.kind for p in params.values())
        if param_kinds != args_kwargs:
            return params.items()
    raise Exception(f"Unexpected signature for klass {klass}: {params}")


Dereferencer = Callable[["JsonReference"], Any]


def hint_to_deserializer(type_hint, dereferencer: Optional[Dereferencer] = None):
    if re.search(r"^(typing\.)?(List|Iterable|Tuple|Sequence)\[", str(type_hint)):
        item_type = type_hint.__args__[0]
        return lambda value: [from_json_data(item_type, v, dereferencer=dereferencer) for v in value]
    elif re.search(r"^(typing\.)?Union\[", str(type_hint)):
        exceptions = []

        def wrapper(value):
            if value is None and None.__class__ in type_hint.__args__:
                return None
            for item_type in type_hint.__args__:
                if item_type == None.__class__:
                    continue
                try:
                    return from_json_data(item_type, value, dereferencer=dereferencer)
                except Exception as e:
                    exceptions.append(e)
            raise ValueError("Could not deserialize value {value}: {exceptions}")

        return wrapper
    raise NotImplementedError(f"Don't know how to unwrap {type_hint}")


def is_type_hint(obj) -> bool:
    return hasattr(obj, "__origin__")


def from_json_data(cls, data, *, dereferencer: Optional[Dereferencer] = None, initOnly: bool = False):
    if isinstance(data, JsonReference):
        return dereferencer(data)
    if is_type_hint(cls):
        deserializer = hint_to_deserializer(cls, dereferencer=dereferencer)
        return deserializer(data)
    if inspect.isclass(cls):
        if isinstance(data, cls):
            return data
        if not isinstance(data, BaseMapping):
            return cls(data)
        if cls.__name__ != "JsonReference" == data.get("__class__"):
            ref = JsonReference.from_json_data(data, dereferencer=None)
            obj = dereferencer(ref)
            return from_json_data(cls, obj, dereferencer=dereferencer)
        if not initOnly and hasattr(cls, "from_json_data"):
            return cls.from_json_data(data, dereferencer=dereferencer)
    data = data.copy()
    assert data.pop("__class__", None) in (cls.__name__, None)
    this_params = {}
    for name, parameter in get_constructor_params(cls):
        type_hint = parameter.annotation
        assert type_hint != inspect._empty, f"Missing type hint for param {name}"
        if name not in data:  # might be a missing optional
            continue
        this_params[name] = from_json_data(type_hint, data.pop(name), dereferencer=dereferencer)
    if len(data) > 0:
        print(f"WARNING: Unused arguments when deserializing {cls.__name__}: {data}")
    return cls(**this_params)


int_classes = tuple(
    [int]
    + [
        getattr(np, "".join(type_parts))
        for type_parts in itertools.product(["u", ""], ["int"], ["8", "16", "32", "64"])
    ]
)
float_classes = tuple([float] + [np.float, np.float16, np.float32, np.float64])

Referencer = Callable[[Any], Optional["JsonReference"]]


def obj_to_json_data(value, *, referencer: Referencer = lambda obj: None, initOnly: bool = False):
    out_dict = {"__class__": value.__class__.__name__}
    ref = referencer(value)
    if ref is not None:
        out_dict["__self__"] = ref.to_json_data()
    for name, parameter in inspect.signature(value.__class__).parameters.items():
        out_dict[name] = to_json_data(getattr(value, name), referencer=referencer)
    return out_dict


def to_json_data(value, *, referencer: Referencer = lambda obj: None, initOnly: bool = False):
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, uuid.UUID):
        return str(value)
    ref = referencer(value)
    if ref is not None:
        return ref.to_json_data()
    if not initOnly and hasattr(value, "to_json_data"):
        return value.to_json_data(referencer=referencer)
    if isinstance(value, int_classes):
        return int(value)
    if isinstance(value, float_classes):
        return float(value)
    if isinstance(value, (tuple, list, np.ndarray)):
        return [to_json_data(v, referencer=referencer) for v in value]
    if isinstance(value, BaseMapping):
        return {k: to_json_data(v, referencer=referencer) for k, v in value.items()}
    return obj_to_json_data(value)


JSO = TypeVar("JSO", bound="JsonSerializable", covariant=True)


class JsonSerializable(ABC):
    def to_json_data(self, referencer: Referencer = lambda obj: None):
        return obj_to_json_data(self, referencer=referencer, initOnly=True)

    def to_json(self, referencer: Referencer = lambda obj: None) -> str:
        return json.dumps(self.to_json_data(referencer=referencer))

    @classmethod
    def from_json(cls: Type[JSO], data: str, dereferencer: Optional[Dereferencer] = None) -> JSO:
        return cls.from_json_data(json.loads(data), dereferencer=dereferencer)

    @classmethod
    def from_json_data(cls: Type[JSO], data: dict, dereferencer: Optional[Dereferencer] = None) -> JSO:
        deserialized = from_json_data(cls, data, dereferencer=dereferencer, initOnly=True)
        return cast(cls, deserialized)


class JsonReference(JsonSerializable):
    def __init__(self, object_id: uuid.UUID):
        self.object_id = object_id

    def to_str(self) -> str:
        return str(self.object_id)

    @classmethod
    def from_str(cls, s: str) -> "JsonReference":
        return cls(uuid.UUID(s))

    @classmethod
    def create(cls):
        return cls(uuid.uuid4())

    def __hash__(self):
        return hash(self.object_id)

    def __eq__(self, other):
        return isinstance(other, JsonReference) and other.object_id == self.object_id
