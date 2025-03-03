# Modules API

Modules are the basic building blocks of programs in Synalinks. A Module consists of data model-in & data model-out computation function (the module's `call()` method) and some state (held in `Variable`).

A module instance is callable, much like a function:

``` py
import synalinks
import asyncio

async def main():
    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )

    class AnswerWithThinking(synalinks.DataModel):
        thinking: str = synalinks.Field(
            description="Your step by step thinking",
        )
        answer: str = synalinks.Field(
            description="The correct answer",
        )

    language_model = LanguageModel("ollama_chat/deepseek-r1")

    generator = synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )

    inputs = Query(query="What is the capital of France?")
    outputs = await generator(inputs)


if __name__ == "__main__":
    asyncio.run(main())
```

## Modules API overview

- [Base Module class](Base Module class.md)

---

### Core Modules

- [Input Module](Core Modules/Input module.md)
- [Generator Module](Core Modules/Generator module.md)
- [Decision Module](Core Modules/Decision module.md)
- [Action Module](Core Modules/Action module.md)
- [Branch Module](Core Modules/Branch module.md)

---

### Merging Modules

- [Concat Module](Merging Modules/Concat module.md)
- [Logical And](Merging Modules/And module.md)
- [Logical Or](Merging Modules/Or module.md)

---

### Agents Modules

- [ReACT Agent Module](Agents Modules/ReACT Agent module.md)