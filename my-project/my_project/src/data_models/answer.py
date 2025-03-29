import synalinks

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(
        description="The correct answer",
    )