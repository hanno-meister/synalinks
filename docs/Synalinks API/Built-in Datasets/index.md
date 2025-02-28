# Built-in Datasets

The `synalinks.datasets` module provide a few datasets that can be used to debugging, evaluation or to create code examples.

These datasets are leaked in nowadays LMs training data, which is a big concern in todays ML community, so they won't give you much information about the reasoning abilities of the underlying models. But they are still usefull as baseline to compare neuro-symbolic methods or when using small language models.

---

- [GSM8K dataset](GSM8K.md): A dataset of 8.5K high quality linguistically diverse grade school math word problems. Usefull to evaluate reasoning capabilities.

- [ARC-AGI dataset](ARC-AGI.md): A dataset of 400 different tasks about general artificial intelligence, as a program synthesis benchmark. Usefull to evaluate general reasoning abilities and program synthesis applications.