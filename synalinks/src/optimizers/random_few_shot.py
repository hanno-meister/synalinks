# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.optimizers.optimizer import Optimizer


class FewShotOptimizedVariable(DataModel):
    examples: List[
        Tuple[
            Dict[str, Any],
            Dict[str, Any],
            Optional[float],
        ]
    ] = []
    predictions: List[
        Tuple[
            Dict[str, Any],
            Dict[str, Any],
            Optional[float],
        ],
    ] = []


@synalinks_export("synalinks.optimizers.RandomFewShot")
class RandomFewShot(Optimizer):
    """Select randomly the best examples to populate the LM's prompt to make it
        learn using Few Shot Learning.

    Example:

    ```python
    import synalinks
    import asyncio

    async def main():
        # ... your program definition

        program.compile(
            reward=synalinks.rewards.ExactMatch(),
            optimizer=synalinks.optimizers.RandomFewShot(
                k=3,
                k_best=10,
            ),
        )

        history = await program.fit(...)
    ```

    References:
        - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

    Args:
        k (int): The number of examples to select (default 3) among the best predictions.
        k_best (int): The max number of best predictions to select from (default 10).

    Returns:
        (Optimizer): The created optimizer.
    """

    def __init__(
        self,
        k=3,
        k_best=10,
        name=None,
        description=None,
    ):
        super().__init__(
            name=name,
            description=description,
            data_model=FewShotOptimizedVariable,
        )
        self.k = k
        self.k_best = k_best

    def build(self, variables):
        self.built = True

    async def optimize(self, trainable_variable, reward=None):
        """Perform a backprop/optimization on a single variable."""
        # Reward backpropagation
        predictions = trainable_variable.get_json().get("predictions")
        predictions = backpropagate_reward_to_predictions(predictions, reward)
        trainable_variable.get_json().update({"predictions": predictions})
        # Get the k best predictions (sorted by reward)
        sorted_predictions = sorted(
            predictions,
            key=lambda x: x[2] if x[2] is not None else float("-inf"),
            reverse=True,
        )
        top_k_predictions = sorted_predictions[: self.k_best]
        if len(top_k_predictions) > self.k:
            selected_predictions = random.sample(top_k_predictions, self.k)
        else:
            selected_predictions = top_k_predictions
        trainable_variable.get_json().update({"examples": selected_predictions})

    async def finalize(self, trainable_variable):
        """Finalize the optimization of a single variable (cleanup/scaling etc.)."""
        trainable_variable.get_json().update({"predictions": []})

    def get_config(self):
        return {
            "k": self.k,
            "k_best": self.k_best,
            "name": self.name,
            "description": self.description,
        }


def backpropagate_reward_to_predictions(predictions, reward):
    assigned_prediction = []
    for p in predictions:
        if p[2] is None:
            p = list(p)
            p[2] = reward
            p = tuple(p)
        assigned_prediction.append(p)
    return assigned_prediction
