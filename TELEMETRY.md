# Usage telemetry

In order to enhance Synalinks and provide you with the best software possible we collect usage information with [Sentry](https://sentry.io/welcome/).

Here is the list of what we collect, and the reason behind:

- *Modules exceptions*: To solve errors before they can go into production
- *Language & embedding models exceptions*: The monitor integrations stability
- *Knowledge bases exceptions*: To monitor integrations stability
- *Uncatched exceptions*: To enhance your experience by providing you with better error messages

You can disable the telemetry by setting the following evironment variable `export SYNALINKS_TELEMETRY=false` in your `~/.bashrc` (or equivalent).

OR you can use `synalinks.disable_telemetry()` at the beginning of your scripts

```python
import synalinks

synalinks.disable_telemetry()
```

**Note**: We might collect more information in the future, but we will **NEVER** collect private, personal or confidential (like your prompts/data models) informations.

If you have specific feedbacks or feature request we invite you to open an [issue](https://github.com/SynaLinks/synalinks/issues).