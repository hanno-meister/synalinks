import synalinks

class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )