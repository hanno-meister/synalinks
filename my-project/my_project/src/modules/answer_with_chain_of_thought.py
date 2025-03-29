import synalinks
from my_project.src.data_models import Answer

class AnswerWithChainOfThought(synalinks.Program):
    """Answer step by step.
    
    Args:
        language_model (LanguageModel): The language model to use.
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        language_model=None,
        name=None,
        description=None,
        trainable=None,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.language_model = language_model

    async def build(self, inputs):
        outputs = await synalinks.ChainOfThought(
            data_model=Answer,
            language_model=self.language_model,
        )(inputs)
        
        super().__init__(
            inputs=inputs,
            outputs=outputs,
        )