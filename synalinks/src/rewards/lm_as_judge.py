# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import ops
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules import Generator
from synalinks.src.programs import Program
from synalinks.src.rewards.reward_wrappers import ProgramAsJudge
from synalinks.src.saving import serialization_lib


class RewardWithCritique(DataModel):
    thinking: str = Field(
        description="The step by step thinking to critique",
    )
    critique: str = Field(
        description="The critique of the provided inputs",
    )
    reward: float = Field(
        description="The reward value between [0.0, 1.0]",
        ge=0.0,
        le=1.0,
    )


class LMAsJudgeProgram(Program):
    """Evaluate the output of a program using a `LanguageModel`.

    Args:
        language_model (LanguageModel): The language model to use.
        prompt_template (str): The default jinja2 prompt template
            to use (see `Generator`).
        examples (list): The default examples to use in the prompt
            (see `Generator`).
        hints (list): The default hints to use (see `Generator`).
        name (str): Optional. The name of the program.
        description (str): Optional. The description of the program.
        trainable (bool): Whether the program's variables should be trainable.
    """

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        hints=None,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.generator = Generator(
            data_model=RewardWithCritique,
            language_model=language_model,
            prompt_template=prompt_template,
            examples=examples,
            hints=hints,
            name=self.name + "_generator",
        )
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.examples = examples
        self.hints = hints

    async def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("The inputs should be a list or tuple.")
        if len(inputs) != 2:
            raise ValueError("The inputs of the program should have a length of 2.")
        y_true = inputs[0]
        y_pred = inputs[1]
        if y_true:
            y_true = await ops.prefix(
                y_true,
                prefix="gold",
                name="gold_y_true",
            )
            return await self.generator(
                await ops.concat(
                    y_true,
                    y_pred,
                    name="y_true_with_y_pred",
                )
            )
        else:
            return await self.generator(y_pred)

    def get_config(self):
        config = {
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "hints": self.hints,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model
            )
        }
        return {**language_model_config, **config}

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(language_model=language_model, **config)


class LMAsJudge(ProgramAsJudge):
    """Evaluate the output of a program using a `LanguageModel`.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.LMAsJudge()
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```

    Args:
        language_model (LanguageModel): The language model to use.
        prompt_template (str): The default jinja2 prompt template
            to use (see `Generator`).
        examples (list): The default examples to use in the prompt
            (see `Generator`).
        hints (list): The default hints to use (see `Generator`).
        name (str): Optional. string name of the reward instance.
        in_mask (list): Optional. list of keys to keep to compute the reward.
        out_mask (list): Optional. list of keys to remove to compute the reward.
    """

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        hints=None,
        name="lm_as_judge",
        in_mask=None,
        out_mask=None,
    ):
        program = LMAsJudgeProgram(
            language_model=language_model,
            prompt_template=prompt_template,
            examples=examples,
            hints=hints,
        )
        super().__init__(
            program=program,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
        )
