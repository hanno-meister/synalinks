# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import inspect
import json

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common.json_schema_utils import standardize_schema
from synalinks.src.backend.common.symbolic_data_model import SymbolicDataModel
from synalinks.src.utils.naming import auto_name


@synalinks_export("synalinks.JsonDataModel")
class JsonDataModel:
    """A backend-independent dynamic data model.

    This structure is the one flowing in the pipelines as
    the backend data models are only used for the variable/data model declaration.

    Args:
        schema (dict): The JSON object's schema. If not provided,
            uses the data_model to infer it.
        value (dict): The JSON object's value. If not provided,
            uses the data_model to infer it.
        data_model (DataModel | JsonDataModel): The data_model to use to
            infer the schema and value.
        name (str): Optional. The name of the data model, automatically
            inferred if not provided.

    Examples:

    **Creating a `JsonDataModel` with a DataModel's schema and value:**

    ```python
    class Query(synalinks.DataModel):
        query: str

    value = {"query": "What is the capital of France?"}

    data_model = JsonDataModel(schema=Query.schema(), value=value)
    ```

    **Creating a `JsonDataModel` with a data_model:**

    ```python
    class Query(synalinks.DataModel):
        query: str

    query_instance = Query(query="What is the capital of France?")
    data_model = JsonDataModel(data_model=query_instance)
    ```
    """

    def __init__(
        self,
        schema=None,
        value=None,
        data_model=None,
        name=None,
    ):
        name = name or auto_name(self.__class__.__name__)
        self._name = name
        self._schema = None
        self._value = None

        if not data_model and not schema and not value:
            raise ValueError("Initializing without arguments is not permited.")
        if not schema and not data_model:
            raise ValueError(
                "You should specify at least one argument between data_model or schema"
            )
        if not value and not data_model:
            raise ValueError(
                "You should specify at least one argument between data_model or value"
            )
        if data_model:
            if not schema:
                schema = data_model.schema()
            if not value:
                if inspect.isclass(data_model):
                    raise ValueError(
                        "Couldn't get the JSON data from the data_model, "
                        "the data_model needs to be instanciated."
                    )
                value = data_model.json()

        self._schema = standardize_schema(schema)
        self._value = value

    def to_symbolic_data_model(self):
        """Converts the JsonDataModel to a SymbolicDataModel.

        Returns:
            SymbolicDataModel: The symbolic data model.
        """
        return SymbolicDataModel(schema=self._schema)

    def json(self):
        """Alias for the JSON object's value.

        Returns:
            dict: The current value of the JSON object.
        """
        return self.value()

    def value(self):
        """The current value of the JSON object.

        Returns:
            dict: The current value of the JSON object.
        """
        return self._value

    def schema(self):
        """Gets the schema of the JSON object.

        Returns:
            dict: The JSON schema.
        """
        return self._schema

    def pretty_schema(self):
        """Get a pretty version of the JSON schema for display.

        Returns:
            dict: The indented JSON schema.
        """
        return json.dumps(self.schema(), indent=2)

    def pretty_json(self):
        """Get a pretty version of the JSON object for display.

        Returns:
            dict: The indented JSON object.
        """
        return json.dumps(self.json(), indent=2)

    @property
    def name(self):
        """The name of the Json object."""
        return self._name

    def __add__(self, other):
        """Concatenates this data model with another.

        Args:
            other (JsonDataModel): The other data model to concatenate with.

        Returns:
            JsonDataModel: The concatenated data model.
        """
        from synalinks.src import ops

        return asyncio.get_event_loop().run_until_complete(ops.Concat().call(self, other))

    def __radd__(self, other):
        """Concatenates another data model with this one.

        Args:
            other (JsonDataModel): The other data model to concatenate with.

        Returns:
            JsonDataModel: The concatenated data model.
        """
        from synalinks.src import ops

        return asyncio.get_event_loop().run_until_complete(ops.Concat().call(other, self))

    def factorize(self):
        """Factorizes the data model.

        Returns:
            JsonDataModel: The factorized data model.
        """
        from synalinks.src import ops

        return asyncio.get_event_loop().run_until_complete(ops.Factorize().call(self))

    def in_mask(self, mask=None, recursive=True):
        """Applies a mask to **keep only** specified keys of the data model.

        Args:
            mask (list): The mask to be applied.
            recursive (bool): Optional. Whether to apply the mask recursively.
                Defaults to True.

        Returns:
            JsonDataModel: The data model with the mask applied.
        """
        from synalinks.src import ops

        return asyncio.get_event_loop().run_until_complete(
            ops.InMask(mask=mask, recursive=recursive).call(self)
        )

    def out_mask(self, mask=None, recursive=True):
        """Applies a mask to **remove** specified keys of the data model.

        Args:
            mask (list): The mask to be applied.
            recursive (bool): Optional. Whether to apply the mask recursively.
                Defaults to True.

        Returns:
            JsonDataModel: The data model with the mask applied.
        """
        from synalinks.src import ops

        return asyncio.get_event_loop().run_until_complete(
            ops.OutMask(mask=mask, recursive=recursive).call(self)
        )

    def prefix(self, prefix=None):
        """Add a prefix to **all** the data model fields (non-recursive).

        Args:
            prefix (str): the prefix to add

        Returns:
            JsonDataModel: The data model with the prefix added.
        """
        from synalinks.src import ops

        return asyncio.get_event_loop().run_until_complete(
            ops.Prefix(prefix=prefix).call(self)
        )

    def suffix(self, suffix=None):
        """Add a suffix to **all** the data model fields (non-recursive).

        Args:
            suffix (str): the suffix to add

        Returns:
            JsonDataModel: The data model with the suffix added.
        """
        from synalinks.src import ops

        return asyncio.get_event_loop().run_until_complete(
            ops.Suffix(suffix=suffix).call(self)
        )

    def get(self, key):
        """Get wrapper to make easier to access fields.

        Args:
            key (str): The key to access.
        """
        return self.json().get(key)

    def update(self, kv_dict):
        """Update wrapper to make easier to modify fields.

        Args:
            kv_dict (dict): The key/value dict to update.
        """
        self.json().update(kv_dict)

    def __repr__(self):
        return f"<JsonDataModel schema={self._schema}, value={self._value}>"


def is_json_data_model(x):
    return isinstance(x, JsonDataModel)
