# First Steps

First, install Synalinks, the easiest way is using pip:

```shell
pip install synalinks
```

Or uv (recommended):

```shell
uv pip install synalinks
```

**Note for Windows users**: Use the [Windows Linux Sub-System](https://learn.microsoft.com/en-us/windows/wsl/install) (WLS).

If you want to install it from source (for contributors), then do:

```shell
git clone https://github.com/SynaLinks/synalinks
cd synalinks
./shell/uv.sh # Install uv
./shell/install.sh # Create the virtual env and install Synalinks
```

After this, open a python file or notebook and check the install:

```python
import synalinks
print(synalinks.__version__)
```

or use `uv run synalinks --version`

Now create a new project using the following command:

```shell
uv run synalinks init
```

This will setup a template project ready to work on.

Synalinks use a global context to ensure that each variable/module
have a unique name. Clear it at the beginning of your scripts to 
ensure naming reproductability.

```python
# Clear the global context
synalinks.clear_session()
```

Addtionally, you can install Ollama [here](https://ollama.com/) to run
Language Models (LMs) locally, which is very useful to development.

## Prompting

You will notice that there is no traditional prompting involved in 
Synalinks, everything is described as data models in and out.
However we use a prompt template, that will tell the system how to 
construct the prompt automatically.

The prompt template is a jinja2 template that describe how to render 
the examples, instructions and how to convert them into chat messages:

### Default Prompt template

::: synalinks.src.modules.core.generator.default_prompt_template

The template use the XML tags `<system>...</system>`, `<user>...</user>` and
`<assistant>...</assistant>` to know how to convert the prompt template 
into chat messages. You can modify at any time the default template used by using the 
`prompt_template` argument in Synalinks modules. You can notice also, 
that we send the inputs's and output's JSON schema to instruct the LMs
how to answer, you can enable/disable that behavior by using `use_inputs_schema`
and `use_outputs_schema` in Synalinks modules. Synalinks use **constrained
structured output** ensuring that the LMs answer respect the data models
specification (the JSON schema), and is ready to parse, so in theory
we don't need it, except if you use it to provide additional information
to the LMs. You can find more information in the 
[`Generator`](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/) documentation.

## Data Models

To provide additional information to the LMs, you can use the data models
`Field`. You can notice that Synalinks use Pydantic as default data backend.
Allowing Synalinks to be compatible out-of-the-box with constrained structured output, FastAPI and FastMCP.

```python
class AnswerWithThinking(synalinks.DataModel):
    thinking: str = synalinks.Field(
        description="Your step by step thinking",
    )
    answer: str = synalinks.Field(
        description="The correct answer",
    )
```

## Conclusion
        
Usually that will be enough to instruct the LMs, you don't need to modify
the prompt template. Just by adding additional descriptions to the data
models fields you can instruct your system to behave as you want. 
If the system needs general instructions about how to behave, you can 
use the `instructions` argument in Synalinks modules that will be formatted as 
presented in the prompt template.

### Key Takeaways

- **Ease of Integration**: Synalinks seamlessly integrates with existing
    Python projects, making it easy to incorporate advanced language 
    model capabilities without extensive modifications.
- **Structured Outputs**: By using data models and JSON schemas combined with
    *constrained structed output*, Synalinks ensures that the LMs responses are structured and ready for parsing, reducing the need for additional post-processing.
- **Customizable Prompts**: The prompt templates in Synalinks are highly
    customizable, allowing you to tailor the instructions provided to
    the LMs based on your specific use case. 
- **Compatibility**: Synalinks use Pydantic as the default data backend
    ensures compatibility with structured output, FastAPI and FastMCP.
