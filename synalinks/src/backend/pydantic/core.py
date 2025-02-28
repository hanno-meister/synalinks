# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import inspect
import json

import pydantic

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common.json_data_model import JsonDataModel
from synalinks.src.backend.common.stateless_scope import StatelessScope
from synalinks.src.backend.common.symbolic_data_model import SymbolicDataModel
from synalinks.src.backend.common.symbolic_scope import SymbolicScope

IS_THREAD_SAFE = True


class MetaDataModel(type(pydantic.BaseModel)):
    """The metaclass data model.

    This class defines operations at the metaclass level.
    Allowing to use Synalinks Python operators with `DataModel` types.
    """

    def schema(cls):
        """Gets the JSON schema of the data model.

        Returns:
            (dict): The JSON schema.
        """
        return cls.model_json_schema()

    def pretty_schema(cls):
        """Get a pretty version of the JSON schema for display.

        Returns:
            dict: The indented JSON schema.
        """
        return json.dumps(cls.schema(), indent=2)

    def to_symbolic_data_model(cls):
        """Converts the data model to a symbolic data model.

        Returns:
            (SymbolicDataModel): The symbolic data model.
        """
        return SymbolicDataModel(schema=cls.schema())

    def __add__(cls, other):
        from synalinks.src import ops

        return asyncio.get_event_loop().run_until_complete(
            ops.Concat().symbolic_call(cls, other)
        )

    def __radd__(cls, other):
        from synalinks.src import ops

        return asyncio.get_event_loop().run_until_complete(
            ops.Concat().symbolic_call(other, cls)
        )

    def __and__(cls, other):
        from synalinks.src import ops

        return asyncio.get_event_loop().run_until_complete(
            ops.And().symbolic_call(cls, other)
        )

    def __rand__(cls, other):
        from synalinks.src import ops

        return asyncio.get_event_loop().run_until_complete(
            ops.And().symbolic_call(other, cls)
        )

    def __or__(cls, other):
        from synalinks.src import ops

        return asyncio.get_event_loop().run_until_complete(
            ops.Or().symbolic_call(cls, other)
        )

    def __ror__(cls, other):
        from synalinks.src import ops

        return asyncio.get_event_loop().run_until_complete(
            ops.Or().symbolic_call(other, cls)
        )


class DataModel(pydantic.BaseModel, metaclass=MetaDataModel):
    """The backend-dependent data model.

    This data model uses Pydantic to provide, JSON schema inference
    and JSON serialization.

    Examples:

    **Creating a DataModel for structured output**

    ```python
    class AnswerWithReflection(synalinks.DataModel):
        rationale: str
        reflection: str
        answer: str

    language_model = synalinks.LanguageModel("ollama/mistral")

    generator = synalinks.Generator(
        data_model=AnswerWithReflection,
        language_model=language_model,
    )
    ```
    """

    def json(self):
        """Alias for the JSON value of the data model.

        Returns:
            (dict): The JSON value.
        """
        return self.value()

    def value(self):
        """Gets the JSON value of the data model.

        Returns:
            (dict): The JSON value.
        """
        return self.model_dump(mode="json")

    def pretty_json(self):
        """Get a pretty version of the JSON object for display.

        Returns:
            dict: The indented JSON object.
        """
        return json.dumps(self.json(), indent=2)

    def __repr__(self):
        return f"<DataModel value={self.value()}, schema={self.schema()}>"

    def to_json_data_model(self):
        """Converts the data model to a backend-independent data model.

        Returns:
            (JsonDataModel): The backend-independent data model.
        """
        return JsonDataModel(value=self.value(), schema=self.schema())

    def __add__(self, other):
        from synalinks.src import ops

        if any_meta_class(self, other):
            return asyncio.get_event_loop().run_until_complete(
                ops.Concat().symbolic_call(self, other)
            )
        else:
            return asyncio.get_event_loop().run_until_complete(ops.Concat()(self, other))

    def __radd__(self, other):
        from synalinks.src import ops

        if any_meta_class(self, other):
            return asyncio.get_event_loop().run_until_complete(
                ops.Concat().symbolic_call(other, self),
            )
        else:
            return asyncio.get_event_loop().run_until_complete(ops.Concat()(other, self))

    def __and__(self, other):
        from synalinks.src import ops

        if any_meta_class(self, other):
            return asyncio.get_event_loop().run_until_complete(
                ops.Add().symbolic_call(self, other)
            )
        else:
            return asyncio.get_event_loop().run_until_complete(ops.Add()(self, other))

    def __rand__(self, other):
        from synalinks.src import ops

        if any_meta_class(other, self):
            return asyncio.get_event_loop().run_until_complete(
                ops.Add().symbolic_call(other, self)
            )
        else:
            return asyncio.get_event_loop().run_until_complete(ops.Add()(other, self))

    def __or__(self, other):
        from synalinks.src import ops

        if any_meta_class(self, other):
            return asyncio.get_event_loop().run_until_complete(
                ops.Or().symbolic_call(self, other)
            )
        else:
            return asyncio.get_event_loop().run_until_complete(ops.Or()(self, other))

    def __ror__(self, other):
        from synalinks.src import ops

        if any_meta_class(other, self):
            return asyncio.get_event_loop().run_until_complete(
                ops.Or().symbolic_call(other, self)
            )
        else:
            return asyncio.get_event_loop().run_until_complete(ops.Or()(other, self))


def is_data_model(x):
    """Returns whether `x` is a DataModel.

    Args:
        x (any): The object to check.

    Returns:
        bool: True if `x` is a DataModel, False otherwise.
    """
    return isinstance(x, DataModel)


def any_data_model(args=None, kwargs=None):
    """Check if any of the arguments are backend-dependent data models.

    Args:
        args (tuple): Optional. The positional arguments to check.
        kwargs (dict): Optional. The keyword arguments to check.

    Returns:
        bool: True if any of the arguments are meta classes, False otherwise.
    """
    args = args or ()
    kwargs = kwargs or {}
    for x in tree.flatten((args, kwargs)):
        if is_meta_class(x):
            return True
    return False


async def compute_output_spec(fn, *args, **kwargs):
    """Computes the output specification of a function.

    This function wraps the given function call in a stateless and symbolic scope
    to compute the output specification.

    Args:
        fn (callable): The function to compute the output specification for.
        *args: The positional arguments to pass to the function.
        **kwargs: The keyword arguments to pass to the function.

    Returns:
        SymbolicDataModel: The output specification of the function.
    """
    with StatelessScope(), SymbolicScope():
        output_spec = await fn(*args, **kwargs)
    return output_spec


def any_meta_class(args=None, kwargs=None):
    """Check if any of the arguments are meta classes.

    This happen when using a `DataModel` without instanciating it.
    In Synalinks this is used when declaring data models for schema inference.

    Args:
        args (tuple): Optional. The positional arguments to check.
        kwargs (dict): Optional. The keyword arguments to check.

    Returns:
        bool: True if any of the arguments are meta classes, False otherwise.
    """
    args = args or ()
    kwargs = kwargs or {}
    for x in tree.flatten((args, kwargs)):
        if is_meta_class(x):
            return True
    return False


@synalinks_export(
    [
        "synalinks.utils.is_meta_class",
        "synalinks.backend.is_meta_class",
    ]
)
def is_meta_class(x):
    """Returns whether `x` is a meta class.

    A meta class is a python type. This method checks if the data model provided
    if a meta class, allowing to detect if the `DataModel` have been instanciated.
    Meta classes are using in Synalinks when declaring data models for schema inference.

    Args:
        x (any): The object to check.

    Returns:
        bool: True if `x` is a meta class, False otherwise.
    """
    return inspect.isclass(x)
