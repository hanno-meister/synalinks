# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import collections
import copy

from synalinks.src.utils.nlp_utils import add_suffix
from synalinks.src.utils.nlp_utils import is_plural
from synalinks.src.utils.nlp_utils import to_plural_without_numerical_suffix
from synalinks.src.utils.nlp_utils import to_singular_without_numerical_suffix


def standardize_schema(schema):
    """Standardize the JSON schema for consistency"""
    if schema.get("title"):
        schema = copy.deepcopy(schema)
        schema.update({"title": "SymbolicDataModel"})
    return schema


def contains_schema(schema1, schema2):
    """Returns True if schema2 properties is a subset of schema1 properties."""
    return schema2.get("properties").items() <= schema1.get("properties").items()


def prefix_schema(schema, prefix):
    """Add a prefix to the schema properties"""
    schema = copy.deepcopy(schema)
    for prop_key, _ in schema.get("properties").items():
        prop_key = f"{prefix}_{prop_key}"
    return schema


def suffix_schema(schema, suffix):
    """Add a suffix to the schema properties"""
    schema = copy.deepcopy(schema)
    for prop_key, _ in schema.get("properties").items():
        prop_key = f"{prop_key}_{suffix}"
    return schema


def is_schema_equal(schema1, schema2):
    """Check if two JSON schemas are equal based on their properties.

    This function compares the properties of two JSON schemas
    to determine if they are equal. It only considers the properties
    and ignores other aspects of the schemas.

    Args:
        schema1 (dict): The first JSON schema to be compared.
        schema2 (dict): The second JSON schema to be compared.

    Returns:
        - (bool): `True` if the schemas are considered equal based on their properties,
        `False` otherwise.
    """
    schema1_properties = schema1.get("properties", {})
    schema2_properties = schema2.get("properties", {})
    if len(schema1_properties) != len(schema2_properties):
        return False
    return schema1_properties == schema2_properties


def is_object(schema):
    """Returns true if the schema is an object's schema"""
    if isinstance(schema, dict):
        if schema.get("type", None) == "object":
            return True
    return False


def is_array(schema):
    """Returns true if the schema is an array's schema"""
    if isinstance(schema, dict):
        if schema.get("type", None) == "array":
            return True
    return False


def concatenate_schema(schema1, schema2):
    """Concatenate two JSON schemas into a single schema.

    This function merges the properties of two JSON schemas into a single schema.
    If there are conflicting property names, it appends a suffix to make them unique.

    Args:
        schema1 (dict): The first JSON schema to be concatenated.
        schema2 (dict): The second JSON schema to be concatenated.

    Returns:
        (dict): A new JSON schema that combines the properties of the input schemas.
    """
    schema1 = copy.deepcopy(schema1)
    schema2 = copy.deepcopy(schema2)
    # Initialize the resulting schema
    result_schema = {
        "additionalProperties": False,
        "$defs": {},
        "properties": {},
        "required": [],
        "title": "SymbolicDataModel",
        "type": "object",
    }

    if schema1.get("$defs") and not schema2.get("$defs"):
        result_schema["$defs"] = schema1.get("$defs")
    if not schema1.get("$defs") and schema2.get("$defs"):
        result_schema["$defs"] = schema2.get("$defs")
    if schema1.get("$defs") and schema2.get("$defs"):
        result_schema["$defs"] = {**schema1.get("$defs"), **schema2.get("$defs")}

    schema1_properties = schema1.get("properties", {})
    schema2_properties = schema2.get("properties", {})

    # Helper function to add a property to the result schema
    def add_property(prop_key, prop_value, suffix=0):
        new_prop_key = prop_key
        while new_prop_key in result_schema["properties"]:
            suffix += 1
            new_prop_key = add_suffix(prop_key, suffix)
            prop_value["title"] = new_prop_key.title().replace("_", " ")
        result_schema["properties"][new_prop_key] = prop_value

        required1 = schema1.get("required")
        required2 = schema2.get("required")
        if required1 and not required2:
            if prop_key in required1:
                result_schema["required"].append(new_prop_key)

        if required2 and not required1:
            if prop_key in required2:
                result_schema["required"].append(new_prop_key)

        if required1 and required2:
            if prop_key in required1 or prop_key in required2:
                result_schema["required"].append(new_prop_key)

    # Add properties from the first schema
    for prop_key, prop_value in schema1_properties.items():
        add_property(prop_key, prop_value)

    # Add properties from the second schema
    for prop_key, prop_value in schema2_properties.items():
        add_property(prop_key, prop_value)

    if len(result_schema.get("$defs")) == 0:
        del result_schema["$defs"]

    # TODO Find an elegant way to deal with that
    # description1 = schema1.get("description")
    # description2 = schema2.get("description")
    # if description1 and not description2:
    #     result_schema.update({"description": description1})
    # elif description2 and not description1:
    #     result_schema.update({"description": description2})
    # elif description1 and description2:
    #     result_schema.update({"description": description1 + "\n" + description2})

    return result_schema


def factorize_schema(schema):
    """Factorize a JSON schema by grouping similar properties into lists.

    This function groups similar properties in a JSON schema into list properties.
        It identifies similar properties based on their base names
        and creates array properties for them. The grouped properties are renamed in
        their plural form.

    Args:
        schema (dict): The input JSON schema to factorize.

    Returns:
        (dict): A factorized JSON schema with grouped properties.
    """
    schema = copy.deepcopy(schema)
    # Initialize the resulting schema
    result_schema = {
        "additionalProperties": False,
        "$defs": {},
        "properties": {},
        "required": [],
        "title": "SymbolicDataModel",
        "type": "object",
    }

    if schema.get("$defs"):
        result_schema["$defs"] = schema.get("$defs")

    schema_properties = schema.get("properties", {})

    for prop_key, prop_value in schema_properties.items():
        # Get the base name
        base_key = to_singular_without_numerical_suffix(prop_key)
        plural_key = to_plural_without_numerical_suffix(base_key)
        # Find all similar properties
        similar_props = [
            p
            for p in schema_properties.keys()
            if to_singular_without_numerical_suffix(p) == base_key and p != prop_key
        ]
        if similar_props and not is_plural(prop_key):
            if plural_key not in result_schema["properties"]:
                # Create an array property
                array_prop = copy.deepcopy(prop_value)
                array_prop["title"] = plural_key.title()
                array_prop["type"] = "array"
                array_prop["items"] = {"type": prop_value["type"]}

                result_schema["properties"][plural_key] = array_prop
                if plural_key not in result_schema["required"]:
                    result_schema["required"].append(plural_key)
        else:
            if not is_plural(prop_key):
                result_schema["properties"][base_key] = prop_value
                if base_key not in result_schema["required"]:
                    result_schema["required"].append(base_key)

    if len(result_schema.get("$defs")) == 0:
        del result_schema["$defs"]

    # TODO Find an elegant way to deal with that
    # description = schema.get("description")
    # if description:
    #     result_schema.update({"description": description})

    return result_schema


def out_mask_schema(schema, mask=None, recursive=True):
    """Mask specific fields of a JSON schema.

    This function looks for properties to mask and removes them.
    It ignores the suffixes that other operations could add.

    Args:
        schema (dict): The input JSON schema to mask.
        mask (list): The base key list to remove.
        recursive (bool): Whether or not to remove
            recursively for nested objects (default True).

    Returns:
        (dict): A masked JSON schema with removed properties.
    """
    result_schema = standardize_schema(copy.deepcopy(schema))

    if not mask:
        return result_schema

    stack_init = [result_schema]

    if recursive:
        if "$defs" in result_schema:
            for obk_name, obj_schema in result_schema["$defs"].items():
                stack_init.append(obj_schema)

    stack = collections.deque(stack_init)

    # Ensure that the mask keys are in singular form
    mask = [to_singular_without_numerical_suffix(k) for k in mask]

    while stack:
        current = stack.pop()
        keys_to_delete = []

        properties = current.get("properties", {})

        for prop_key, prop_value in properties.items():
            base_key = to_singular_without_numerical_suffix(prop_key)
            if base_key in mask:
                keys_to_delete.append(prop_key)

            if recursive:
                if is_object(prop_value):
                    stack.append(prop_value)
                elif is_array(prop_value):
                    stack.append(prop_value["items"])

        for key in keys_to_delete:
            del properties[key]

        if "required" in current:
            temp_required = []
            for req in current.get("required"):
                if req not in keys_to_delete:
                    temp_required.append(req)
            current["required"] = temp_required

    # Clean up defs if no link found after masking
    new_defs = {}
    if "$defs" in result_schema:
        for obj_key, obj_schema in result_schema["$defs"].items():
            if str(result_schema).find(f"#/$defs/{obj_key}") > 0:
                new_defs[obj_key] = obj_schema
        result_schema["$defs"] = new_defs

    return result_schema


def in_mask_schema(schema, mask=None, recursive=True):
    """Keep specific fields of a JSON schema.

    This function looks for properties to keep and removes all others.
    It ignores the suffixes that other operations could add.

    Args:
        - schema (dict): The input JSON schema to mask.
        - mask (list): The base key list to keep.
        - recursive (bool): Whether or not to keep
            recursively for nested objects (default True).

    Returns:
        - (dict): A masked JSON schema with only the specified properties.
    """
    if not mask:
        return {
            "additionalProperties": False,
            "properties": {},
            "title": "SymbolicDataModel",
            "type": "object",
        }

    result_schema = standardize_schema(copy.deepcopy(schema))

    stack_init = [result_schema]

    if recursive:
        if "$defs" in result_schema:
            for _, obj_schema in result_schema["$defs"].items():
                stack_init.append(obj_schema)

    stack = collections.deque(stack_init)

    # Ensure that the mask keys are in singular form
    mask = [to_singular_without_numerical_suffix(k) for k in mask]

    while stack:
        current = stack.pop()
        keys_to_keep = []

        properties = current.get("properties", {})

        for prop_key, prop_value in properties.items():
            base_key = to_singular_without_numerical_suffix(prop_key)

            if base_key in mask:
                keys_to_keep.append(prop_key)

            if recursive:
                if is_object(prop_value):
                    stack.append(prop_value)
                elif is_array(prop_value):
                    stack.append(prop_value["items"])

        keys_to_delete = set(properties.keys()) - set(keys_to_keep)
        for key in keys_to_delete:
            del properties[key]
        if "required" in current:
            required = []
            for req in current["required"]:
                if req in keys_to_keep:
                    required.append(req)
            if required:
                current["required"] = required
            else:
                del current["required"]

    # Clean up defs if no link found after masking
    new_defs = {}
    if "$defs" in result_schema:
        for obj_key, obj_schema in result_schema["$defs"].items():
            if str(result_schema).find(f"#/$defs/{obj_key}") > 0:
                new_defs[obj_key] = obj_schema
        result_schema["$defs"] = new_defs

    return result_schema
