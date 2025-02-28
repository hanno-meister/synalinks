# Modules API

Modules are the basic building blocks of programs in Synalinks. A Module consists of data model-in & data model-out computation function (the module's `call()` method) and some state (held in `Variable`).

A module instance is callable, much like a function:

``` py
import synalinks

class Query(synalinks.DataModel):
    query: str

class ChainOfThought(synalinks.DataModel):
    thinking: str
    answer: str

language_model = LanguageModel("ollama_chat/deepseek-r1")

generator = synalinks.Generator(
    data_model=ChainOfThought,
    language_model=language_model,
)

inputs = Query(query="What is the capital of France?")
outputs = generator(inputs)
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
- [Logical And](Merging Modules/Logical And module.md)
- [Logical Or](Merging Modules/Logical Or module.md)

---

### Agents Modules

- [ReACT Agent Module](Agents Modules/ReACT Agent module.md)