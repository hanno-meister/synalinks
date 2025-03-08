# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)


from synalinks.src.modules.core.action import Action
from synalinks.src.modules.core.branch import Branch
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.merging.logical_or import Or
from synalinks.src.programs.program import Program
from synalinks.src.utils.tool_utils import Tool


def get_decision_question():
    """The default question used for decision-making"""
    return "Choose the next function to use based on its name."


def get_hints():
    """The default hints for decision-making"""
    return [
        "Always reflect on your previous actions to know what to do.",
        "As soon as you know the answer, or the task is finished, choose `finish`.",
    ]


class ReACTAgent(Program):
    """ReACT agent as a directed acyclic graph that choose at each step
        the function to use.

    More information [here](https://arxiv.org/abs/2210.03629)

    The difference with DSPy or AdalFlow implementation is that each node in the DAG
    is a separate module with its own trainable variables, yielding better optimization
    (specific for each step). Which makes it more memory intensive, but since ReACT are
    anyway limited to a small set of tools/functions, its ok.

    **Note:** Each function **MUST** return a JSON object dict and be asynchrounous

    Example:

    ```python

    async def main():

        class Query(DataModel):
            query: str

        class FinalAnswer(DataModel):
            answer: float

        async def calculate(expression: str):
            \"""Calculate the result of a mathematical expression.

            Args:
                expression (str): The mathematical expression to calculate, such as
                    '2 + 2'. The expression can contain numbers, operators (+, -, *, /),
                    parentheses, and spaces.
            \"""
            if not all(char in "0123456789+-*/(). " for char in expression):
                return {
                    "result": None,
                    "log": "Error: invalid characters in expression",
                }
            try:
                # Evaluate the mathematical expression safely
                result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
                return {
                    "result": result,
                    "log": "Successfully executed",
                }
            except Exception as e:
                return {
                    "result": None,
                    "log": f"Error: {e}",
                }

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

        x0 = Input(data_model=Query)
        x1 = await ReACTAgent(
            data_model=FinalAnswer,
            language_model=language_model,
            functions=[calculate],
            max_iterations=3,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    Args:
        schema (dict): The JSON schema to use for the final answer.
            If not provided, it will use the `output_data_model` argument.
        data_model (DataModel | JsonDataModel | SymbolicDataModel): Optional.
            The data model to use for the final answer.
            If None provided, the Agent will return a ChatMessage-like data model.
        functions (list): A list of Python functions for the agent to choose from.
        question (str): Optional. The question to branch at each step.
        language_model (LanguageModel): The language model to use, if provided
            it will ignore `decision_language_model` and `action_language_model` argument.
        decision_language_model (LanguageModel): The language model used for
            decision-making.
        action_language_model (LanguageModel): The language model used for actions.
        prompt_template (str): Optional. The jinja2 prompt template to use
            (See `Generator`).
        examples (list): A default list of examples for decision-making (See `Decision`).
        hints (list): A default list of hints for decision-making (See `Decision`).
        use_inputs_schema (bool): Optional. Whether or not use the inputs schema in
            the decision prompt (Default to False) (see `Decision`).
        use_outputs_schema (bool): Optional. Whether or not use the outputs schema in
            the decision prompt (Default to False) (see `Decision`).
        max_iterations (int): The maximum number of steps to perform.
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        schema=None,
        data_model=None,
        functions=None,
        question=None,
        language_model=None,
        decision_language_model=None,
        action_language_model=None,
        prompt_template=None,
        examples=None,
        hints=None,
        use_inputs_schema=False,
        use_outputs_schema=False,
        max_iterations=5,
        name=None,
        description=None,
        trainable=True,
    ):
        if not schema and data_model:
            schema = data_model.get_schema()
        self.schema = schema

        if language_model:
            self.decision_language_model = language_model
            self.action_language_model = language_model
        else:
            self.decision_language_model = decision_language_model
            self.action_language_model = action_language_model

        self.prompt_template = prompt_template

        if examples:
            examples = []
        self.examples = examples

        if not hints:
            hints = get_hints()
        self.hints = hints

        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema

        assert max_iterations > 1
        self.max_iterations = max_iterations

        if not question:
            question = get_decision_question()
        self.question = question

        self.labels = []
        self.functions = functions
        for fn in self.functions:
            self.labels.append(Tool(fn).name())

        self.labels.append("finish")

        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )

    async def build(self, inputs):
        current_steps = [inputs]
        next_steps = []
        finish_branches = []
        for i in range(self.max_iterations):
            if i < self.max_iterations - 1:
                for step in current_steps:
                    actions = [
                        Action(
                            fn=fn,
                            language_model=self.action_language_model,
                            prompt_template=self.prompt_template,
                            use_inputs_schema=self.use_inputs_schema,
                            use_outputs_schema=self.use_outputs_schema,
                        )
                        for fn in self.functions
                    ]
                    actions.append(
                        Generator(
                            schema=self.schema,
                            language_model=self.action_language_model,
                            prompt_template=self.prompt_template,
                            use_inputs_schema=self.use_inputs_schema,
                            use_outputs_schema=self.use_outputs_schema,
                        )
                    )
                    branches = await Branch(
                        question=self.question,
                        labels=self.labels,
                        branches=actions,
                        language_model=self.decision_language_model,
                        prompt_template=self.prompt_template,
                        examples=self.examples,
                        hints=self.hints,
                        use_inputs_schema=self.use_inputs_schema,
                        use_outputs_schema=self.use_outputs_schema,
                        return_decision=False,
                    )(step)
                    next_steps.extend([step & branch for branch in branches[:-1]])
                    finish_branches.append(branches[-1])
                current_steps = next_steps
                next_steps = []
            else:
                for step in current_steps:
                    last_step = await Generator(
                        schema=self.schema,
                        language_model=self.action_language_model,
                        prompt_template=self.prompt_template,
                    )(step)
                    finish_branches.append(last_step)

        final = await Or()(finish_branches)

        super().__init__(
            inputs=inputs,
            outputs=final,
            name=self.name,
            description=self.description,
            trainable=self.trainable,
        )
