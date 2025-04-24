# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.modules.core.none import NoneModule
from synalinks.src.modules.core.input_module import Input
from synalinks.src.programs.program import Program


class NoneModuleTest(testing.TestCase):
    async def test_single_inputs(self):
        class Query(DataModel):
            query: str

        inputs = Input(data_model=Query)
        outputs = await NoneModule()(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
        )

        result = await program(Query(query="a"))
        
        self.assertEqual(result, None)

    async def test_tuple_inputs(self):
        class Query(DataModel):
            query: str

        inputs = (
            Input(data_model=Query),
            Input(data_model=Query),
            Input(data_model=Query),
        )
        outputs = await NoneModule()(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
        )

        result = await program(
            (
                Query(query="a"),
                Query(query="b"),
                Query(query="c"),
            )
        )

        self.assertEqual(result[0], None)
        self.assertEqual(result[1], None)
        self.assertEqual(result[2], None)

    async def test_list_inputs(self):
        class Query(DataModel):
            query: str

        inputs = [
            Input(data_model=Query),
            Input(data_model=Query),
            Input(data_model=Query),
        ]
        outputs = await NoneModule()(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
        )

        result = await program(
            [
                Query(query="a"),
                Query(query="b"),
                Query(query="c"),
            ]
        )

        self.assertEqual(result[0], None)
        self.assertEqual(result[1], None)
        self.assertEqual(result[2], None)

    async def test_dict_inputs(self):
        class Query(DataModel):
            query: str

        inputs = {
            "a": Input(data_model=Query),
            "b": Input(data_model=Query),
            "c": Input(data_model=Query),
        }
        outputs = await NoneModule()(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
        )

        result = await program(
            {
                "a": Query(query="a"),
                "b": Query(query="b"),
                "c": Query(query="c"),
            }
        )

        expected_json_a = {
            "query": "a",
        }
        expected_json_b = {
            "query": "b",
        }
        expected_json_c = {
            "query": "c",
        }

        self.assertEqual(result["a"], None)
        self.assertEqual(result["b"], None)
        self.assertEqual(result["c"], None)
