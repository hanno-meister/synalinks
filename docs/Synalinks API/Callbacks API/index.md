# Callbacks API

A callback is an object that can perform various actions at multiple stages of the program's training. For example, at the start or end of an epoch, before or after a single batch, etc.

## How to use Callbacks

You can pass a list of callbacks to the `.fit()` method of a program.

```python
callbacks = [
    synalinks.callbacks.CSVLogger(filename="training_log.csv"),
    synalinks.callbacks.ProgramCheckpoint(
        filepath="program.{epoch:02d}-{val_loss:.2f}.json"
    ),
]

program.fit(
    x=x_train,
    y=y_train,
    epochs=10,
    callbacks=callbacks,
)
```

## Callbacks Overview

- [Base Callback class](Base Callback class.md)
- [CSVLogger callback](CSVLogger.md)
- [ProgramCheckPoint callback](ProgramCheckpoint.md)