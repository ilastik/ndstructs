from abc import ABC, abstractmethod
import itertools
from collections.abc import Mapping as BaseMapping
import json
import inspect
import re
from typing import Sequence, List, Mapping, Union, Callable, Any, Optional, TypeVar, Type, cast, Tuple
from typing_extensions import Protocol
import numpy as np
import uuid


JsonLeafValue = Union[int, float, str, bool, None]

JsonObject = Mapping[str, "JsonValue"]

JsonArray = Tuple["JsonValue", ...] #tuples are invariant

JsonValue = Union[JsonLeafValue, JsonArray, JsonObject]

#//////////////////////////////

class IJsonable(Protocol):
    def toJsonValue(self) -> JsonValue:
        ...

JsonableArray = Tuple["JsonableValue", ...]

JsonableMapping = Mapping[str, "JsonableValue"]

JsonableValue = Union[JsonValue, IJsonable, JsonableArray, JsonableMapping]

#///////////////////////////////////////////

def toJsonValue(value: JsonableValue) -> JsonValue:
    if isinstance(value, (int, float, str, bool, type(None))):
        return value

    if isinstance(value, tuple):
        return tuple(toJsonValue(val) for val in value)

    if isinstance(value, BaseMapping):
        return {k: toJsonValue(v) for k, v in value.items()}

    return value.toJsonValue()


#///////////////////////////////////////////


def isJsonLeafValue(value: JsonableValue) -> bool:
    return isinstance(value, (int, float, str, bool, type(None)))

def isJsonableArray(value: JsonableValue) -> bool:
    return isinstance(value, list)

def isIJsonable(value: JsonableValue) -> bool:
    return hasattr(value, "toJsonValue")

def ensureJsonBoolean(value: JsonValue) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean, found {json.dumps(value)}")
    return value

def ensureJsonInt(value: JsonValue) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Expected int, found {json.dumps(value)}")
    return value

def ensureJsonFloat(value: JsonValue) -> float:
    if not isinstance(value, (float, int)):
        raise ValueError(f"Expected float, found {json.dumps(value)}")
    return float(value)

def ensureJsonString(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Expected string, found {json.dumps(value)}")
    return value

def ensureJsonObject(value: JsonValue) -> JsonObject:
    if not isinstance(value, BaseMapping):
        raise ValueError(f"Expected json object, found {json.dumps(value)}")
    return value

def ensureJsonArray(value: JsonValue) -> JsonArray:
    if isinstance(value, list):
        value = tuple(value) # we need tuples for invariant sequences
    if not isinstance(value, tuple):
        raise ValueError(f"Expected json list, found {json.dumps(value)}")
    return value

def ensureJsonIntTripplet(value: JsonValue) -> Tuple[int, int, int]:
    number_array = [ensureJsonInt(element) for element in ensureJsonArray(value)]
    if len(number_array) != 3:
        raise TypeError(f"Expected number tripplet, found this: {json.dumps(value)}")
    return (number_array[0], number_array[1], number_array[2])
