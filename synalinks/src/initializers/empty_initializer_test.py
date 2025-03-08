# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from typing import List

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.initializers.empty_initializer import Empty


class EmptyInitializerTest(testing.TestCase):
    def test_empty_initializer(self):
        class Hints(DataModel):
            hints: List[str] = []

        initializer = Empty(data_model=Hints)
        empty_data_model = initializer()
        self.assertEqual(initializer.get_schema(), Hints.get_schema())
        self.assertEqual(empty_data_model, Hints().get_json())

    def test_empty_initializer_from_config(self):
        class Hints(DataModel):
            hints: List[str] = []

        initializer = Empty(data_model=Hints)
        config = initializer.get_config()
        initializer = Empty.from_config(config)
        empty_data_model = initializer()
        self.assertEqual(initializer.get_schema(), Hints.get_schema())
        self.assertEqual(empty_data_model, Hints().get_json())
