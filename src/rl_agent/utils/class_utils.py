import typing


class Representable:
    """ Mixin for objects that adds pretty printing of fields """
    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.__dict__)

    def __str__(self):
        return self.__repr__()


class Cloneable:
    """ Mixin for objects that allow to instantiate a copy of them with overriding specific fields """
    def clone_with(self, **kwargs):
        assert all([k in self._get_fields().keys() for k in kwargs.keys()]), \
            "Tried to override non-existing field(s) %s" % [k for k in kwargs.keys() if k not in self._get_fields().keys()]
        return self.__class__.__call__(**{k: kwargs.get(k, v) for k, v in self._get_fields().items()})

    def _get_fields(self):
        return self.__dict__


class NamedTupleBase(typing.NamedTupleMeta):
    """ A NamedTuple extension that can work with multiple-inheritance. See: https://stackoverflow.com/a/50369521 """
    def __new__(cls, typename, bases, ns):
        cls_obj = super().__new__(cls, typename+'_nm_base', bases, ns)
        bases = bases + (cls_obj,)
        return type(typename, bases, {})


class CloneableNamedTuple(Cloneable, metaclass=NamedTupleBase):
    """ A mixin for NamedTuples that adds clone_with functionality """
    def _get_fields(self):
        return self._asdict()
