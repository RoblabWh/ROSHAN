"""Thin wrapper around C++ schema-driven observation dict providing named tensor access."""


class ObservationDict:
    __slots__ = ('_schema', '_data')

    def __init__(self, raw_dict):
        self._schema = raw_dict.pop("_schema")  # {group: [(name, dim), ...]}
        self._data = raw_dict

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    @property
    def schema(self):
        return self._schema

    def group_dim(self, group):
        return sum(d for _, d in self._schema[group])
