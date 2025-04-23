# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)
import copy


def dynamic_enum(schema, prop_to_update, labels, parent_schema=None, description=None):
    """Update a schema with dynamic Enum string.

    Args:
        schema (dict): The schema to update.
        prop_to_update (str): The property to update.
        labels (list): The list of labels (strings).
        parent_schema (dict, optional): An optional parent schema to use as the base.
        description (str, optional): An optional description for the enum.

    Returns:
        dict: The updated schema with the enum applied to the specified property.
    """
    schema = copy.deepcopy(schema)
    if parent_schema:
        parent_schema = copy.deepcopy(parent_schema)
    title = prop_to_update.title().replace("_", " ")
    if not schema.get("$defs"):
        schema = {"$defs": {}, **schema}

    if description:
        enum_definition = {
            "enum": labels,
            "description": description,
            "title": title,
            "type": "string",
        }
    else:
        enum_definition = {
            "enum": labels,
            "title": title,
            "type": "string",
        }

    if parent_schema:
        parent_schema["$defs"].update({title: enum_definition})
    else:
        schema["$defs"].update({title: enum_definition})

    schema.setdefault("properties", {}).update(
        {prop_to_update: {"$ref": f"#/$defs/{title}"}}
    )

    return parent_schema if parent_schema else schema


# TODO

# def dynamic_union(schema, prop_to_update, schemas, parent_schema=None):
#     """Updates the specified property in the schema to be a union of multiple schemas.

#     Args:
#         schema (dict): The original schema to be updated.
#         prop_to_update (str): The property key to update with the union of schemas.
#         schemas (list): A list of schemas to be combined into the union.
#         parent_schema (dict, optional): An optional parent schema to use as the base.

#     Returns:
#         dict: The updated schema with the union applied to the specified property.
#     """
#     target_schema = parent_schema if parent_schema else schema

#     if not target_schema.get("$defs"):
#         target_schema["$defs"] = {}

#     target_schema["$defs"].update(schemas)

#     schema.setdefault("properties", {})[prop_to_update] = {
#         "anyOf": [{"$ref": f"#/$defs/{k}"} for k in schemas.keys()]
#     }

#     return target_schema


# def dynamic_optional(schema):
#     """Modifies the schema to make all properties optional by allowing them to be null,
#     but only if they are not already optional.

#     Args:
#         schema (dict): The schema to update.

#     Returns:
#         dict: The updated schema with all properties made optional.
#     """
#     for prop_key, prop_value in schema.get("properties", {}).items():
#         if not any(
#             isinstance(clause, dict) and clause.get("type") == "null"
#             for clause in prop_value.get("anyOf", [])
#         ):
#             schema["properties"][prop_key] = {
#                 "anyOf": [prop_value, {"type": "null"}]
#             }
#     return schema


# def dynamic_description(schema, description):
#     """Adds a description to the schema.

#     Args:
#         schema (dict): The schema to update.
#         description (str): The description to add to the schema.

#     Returns:
#         dict: The updated schema with the description added.
#     """
#     schema["description"] = description
#     return schema
