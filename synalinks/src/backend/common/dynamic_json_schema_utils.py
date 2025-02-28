# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)


def dynamic_enum(schema, prop_to_update, labels):
    """Update a schema with dynamic Enum string

    Args:
        schema (dict): The schema to update.
        prop_to_update (str): The property to update.
        labels (list): The list of labels (strings).
    """
    title = prop_to_update.title().replace("_", " ")
    if not schema.get("$defs"):
        schema = {"$defs": {}, **schema}
    schema.update(
        {
            "$defs": {
                title: {"enum": labels},
                "title": title,
                "type": "string",
            },
        }
    )
    schema.get("properties").update(
        {
            prop_to_update: {"$ref": f"#/$defs/{title}"},
        }
    )
    return schema
