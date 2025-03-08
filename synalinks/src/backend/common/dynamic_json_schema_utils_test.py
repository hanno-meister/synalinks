# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from enum import Enum

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import is_schema_equal
from synalinks.src.backend.common.dynamic_json_schema_utils import dynamic_enum


class DynamicEnumTest(testing.TestCase):
    def test_basic_dynamic_enum(self):
        class DecisionAnswer(DataModel):
            thinking: str
            choice: str

        class Choice(str, Enum):
            easy = "easy"
            difficult = "difficult"
            unknown = "unknown"

        class Decision(DataModel):
            thinking: str
            choice: Choice

        labels = ["easy", "difficult", "unkown"]

        schema = dynamic_enum(DecisionAnswer.get_schema(), "choice", labels)

        self.assertTrue(is_schema_equal(Decision.get_schema(), schema))
