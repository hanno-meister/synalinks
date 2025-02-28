import marimo

__generated_with = "0.11.9"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import synalinks

    synalinks.backend.clear_session()
    return mo, synalinks


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Implementing custom modules & programs via subclassing

        This tutorial, is for more advanced users, it will cover how to 
        create custom modules/programs via subclassing.

        In this tutorial, we will cover the following themes:

        - The `Module` class
        - The `add_variable()` method
        - Trainable and non-trainable variables
        - The `build()` method
        - The `compute_output_spec()` method
        - The training argument in `call()`
        - Making sure your module/program can be serialized

        ---

        One of the main abstraction of Synalinks is the `Module` class.
        A `Module` encapsulate both a state (the module's variables) and 
        a transformation from inputs to outputs (the `call()` method).

        For this tutorial, we are going to make a simple neuro-symbolic component
        called `BacktrackingOfThought`. This component is an adaptation of the 
        famous backtracking algorithm, used a lot in symbolic planning/reasoning, 
        combined with chain of thought, nowadays most used technique to enhance
        the LMs predicitons. 

        The principle is straitforward, the component will have to "think" then 
        we will critique at runtime the thinking and aggregate it to 
        the current chain of thinking only if it is above the given threshold. 
        This mechanism will allow the system to discard bad thinking to resume 
        at the previsous step. Additionally we will add a stop condition.

        This algorithm a simplified version of the popular `TreeOfThought` that
        instead of being a tree strucutre, is only a sequential chain of thinking.
        """
    )
    return


@app.cell
def _(synalinks):
    from synalinks import ops

    class Thinking(synalinks.DataModel):
        thinking: str = synalinks.Field(
            description="Your step by step thinking process"
        )

    class CritiqueWithReward(synalinks.DataModel):
        critique: str = synalinks.Field(
            description="The step by step critique"
        )
        reward: float = synalinks.Field(
            description="The reward corresponding to the critique between [0.0, 1.0]",
            le=1.0,
            ge=0.0,
        )

    class BacktrackingOfThought(synalinks.Module):
        def __init__(
            self,
            schema=None,
            data_model=None,
            language_model=None,
            backtracking_threshold=0.5,
            stop_threshold=0.8,
            max_iterations=5,
            critique_program=None,
            prompt_template=None,
            examples=None,
            hints=None,
            use_inputs_schema=False,
            use_outputs_schema=False,
            name=None,
            description=None,
            trainable=None,
        ):
            super().__init__(
                name=name,
                description=description,
                trainable=trainable,
            )
            if not schema and data_model:
                schema = data_model.schema()
            self.schema = schema
            self.language_model = language_model
            self.backtracking_threshold = backtracking_threshold
            self.stop_threshold = stop_threshold
            self.max_iterations = max_iterations
            self.critique_program = critique_program
            self.prompt_template= prompt_template
            self.examples = examples
            self.hints = hints
            self.use_inputs_schema = use_inputs_schema
            self.use_outputs_schema = use_outputs_schema
            if not self.critique_program:
                # If no critique program is provided
                # We compute the reward in the thinking step
                thinking_data_model = \
                    Thinking \
                    + synalinks.SymbolicDataModel(
                        schema=self.schema
                    ) + CritiqueWithReward
            else:
                thinking_data_model = \
                    Thinking \
                    + synalinks.SymbolicDataModel(
                        schema=self.schema
                    )
            # This is for generating the intermediary steps
            self.thinking = synalinks.Generator(
                data_model=thinking_data_model,
                language_model=self.language_model,
                prompt_template=self.prompt_template,
                examples=self.examples,
                hints=self.hints,
                use_inputs_schema=self.use_inputs_schema,
                use_outputs_schema=self.use_outputs_schema,
                name=self.name+"_thinking_generator",
            )
            # This is going to be the final generator
            self.generator = synalinks.Generator(
                schema=self.schema,
                language_model=self.language_model,
                prompt_template=self.prompt_template,
                examples=self.examples,
                hints=self.hints,
                use_inputs_schema=self.use_inputs_schema,
                use_outputs_schema=self.use_outputs_schema,
                name=self.name+"_generator",
            )

        async def call(self, inputs, training=False):
            if not inputs:
                # This is to allow logical flows
                # (don't run the module if no inputs provided)
                return None
            for i in self.max_iterations:
                thinking = await self.thinking(inputs)
                reward = 0.0
                if self.critique_program:
                    critique = await self.critique_program(thinking)
                    reward = critique.get("reward")
                else:
                    reward = thinking.get("reward")
                if reward > self.backtracking_threshold:
                    if reward > self.stop_threshold:
                        break
                    inputs = await ops.concat(
                        inputs,
                        thinking,
                        name=self.name+f"_thinking_{i}"
                    )
            return await self.generator(inputs)

        async def compute_output_spec(self, _, training=False):
            return synalinks.SymbolicDataModel(self.schema)

        def get_config(self):
            config = {
                "schema": self.schema,
                "backtracking_threshold": self.backtracking_threshold,
                "stop_threshold": self.stop_threshold,
                "max_iterations": self.max_iterations,
                "prompt_template": self.prompt_template,
                "examples": self.examples,
                "hints": self.hints,
                "use_inputs_schema": self.use_inputs_schema,
                "use_outputs_schema": self.use_outputs_schema,
                "name": self.name,
                "description": self.description,
                "trainable": self.trainable,
            }        
            language_model_config = {
                "language_model": synalinks.saving.serialize_synalinks_object(
                    self.language_model,
                )
            }
            if self.critique_program:
                critique_program_config = {
                    "critique_program": synalinks.saving.serialize_synalinks_object(
                        self.critique_program,
                    )
                }
            else:
                critique_program_config = {
                    "critique_program": None,
                }
            return {**config, **language_model_config, **critique_program_config}

        @classmethod
        def from_config(cls, config):
            language_model = synalinks.saving.deserialize_synalinks_object(
                config.pop("language_model")
            )
            if config.get("critique_program"):
                critique_program = synalinks.saving.deserialize_synalinks_object(
                    config.pop("critique_program")
                )
            else:
                critique_program = None
            return cls(
                language_model=language_model,
                critique_program=critique_program,
                **config,
            )
    return BacktrackingOfThought, CritiqueWithReward, Thinking, ops


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The `__init__()` function

        First, let's explain the `__init__()` function. When implementing modules that
        use a `Generator`, you want to externalize the generator's parameters 
        (`prompt_template`, `hints`, `examples`, `use_inputs_schema`, `use_outputs_schema`)
        to give maximum flexibility to your module when possible.
        Then, you have to include the default arguments of a module (`name`, `description`, `trainable`)
        that will be provided to the `super().__init__()`. 
        Although the name and description are inferred automatically it is a good practice to
        let the user personalize them. The `trainable` argument, will indicate if the module 
        is frozen or not, meaning that their variables could be updated by the optimizer, 
        by default, a module should be trainable. 

        And finally, you can add any relevant information, weither for the initialization of 
        the variables, or a config parameter like here.

        To add a variable to the module, you have to use the `add_variables` function,
        this function can only be used in the `__init__()` or in the `build()` function.
        The build function is usefull to create variables, or initialize your module/program 
        based on the actual inputs schema, that is not known at this stage,
        because each module should be as general as possible.

        ### How to know when using a `Variable`?

        As a rule of thumb, the variables should be anything that evolve over time during
        inference/training. These variables could be updated by the module itself, or by 
        the optimizer if you have an optimizer designed for that. They will be serialized
        when you save your program so you can recover the state of your program by loading
        a JSON file. In this example, the variables are encapsulated in the `Generator`.

        ### The `call()` function

        The `call()` function is the core of the `Module` class. It defines the computation 
        performed at every call of the module.
        This function takes `inputs` and an optional `training` argument, which indicates
        whether the module is in training mode or not.

        In the `BacktrackingOfThought` module, the `call()` function implements the 
        backtracking logic:

        - It iterates up to `max_iterations` times.
        - In each iteration, it generates a "thinking" step using the `thinking` generator.
        - It then critiques the generated thinking using either a provided critique program or 
            a reward value embedded in the thinking step.
        - If the reward exceeds the `backtracking_threshold`, the thinking step is concatenated 
            with the inputs for the next iteration.
        - If the reward exceeds the `stop_threshold`, the iteration stops early.
        - Finally, the `generator` produces the final output based on the accumulated inputs.

        ### The `compute_output_spec()` function

        The `compute_output_spec()` function is responsible for defining the output data model
        of the module/program. It allows the system to understand the structure of the data
        produced by this module.

        In this example, `compute_output_spec()` returns a `SymbolicDataModel` based on the module's 
        schema, indicating the expected structure of the output data.

        As a rule of thumb, if you access a data model field (using `get()`) you will have to 
        implement it otherwise, Synalinks will infer the output spec by running the call 
        function with symbolic data models. If you have any doubt, do not implement it and the system will
        raise an error if you needs to.

        ### Serialization and Deserialization

        To ensure that your module can be saved and loaded correctly, you need to implement serialization
        and deserialization methods. This is crucial for saving the state of your module, including 
        any trainable variables, and restoring it later.

        - The `get_config()` method should return a dictionary containing all the information needed 
            to recreate the module. This includes the module's configuration and any serialized 
            sub-components like the language model or critique program.
        - The `from_config()` class method should be able to reconstruct the module from the 
            configuration dictionary returned by `get_config()`.

        ### Conclusion

        By following these guidelines, you can create custom modules in Synalinks that are flexible, 
        reusable, and can be integrated into larger programs. The `BacktrackingOfThought` module 
        demonstrates how to combine symbolic reasoning with language model predictions to enhance 
        the decision-making process.

        ---

        This concludes the tutorial on implementing custom modules and programs via subclassing in Synalinks.
        You should now have a solid understanding of how to create and integrate custom components into
        your neuro-symbolic programs.
        """
    )
    return


if __name__ == "__main__":
    app.run()
