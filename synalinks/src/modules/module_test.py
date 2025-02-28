# Modified from: keras/src/layers/layer_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import backend
from synalinks.src import modules
from synalinks.src import testing


class ModuleTest(testing.TestCase):
    async def test_compute_output_spec(self):
        class Query(backend.DataModel):
            query: str

        # Case: single output
        class TestModule(modules.Module):
            async def call(self, x):
                assert False  # Should never be called.

            async def compute_output_spec(self, input_schema):
                return input_schema

        module = TestModule()
        self.assertEqual(
            await module.compute_output_spec(
                backend.SymbolicDataModel(data_model=Query)
            ).schema(),
            backend.standardize_schema(Query.schema()),
        )

        # Case: tuple output
        class TestModule(modules.Module):
            async def call(self, x):
                assert False  # Should never be called.

            async def compute_output_spec(self, input_schema):
                return (input_schema, input_schema)

        module = TestModule()
        out = await module.compute_output_spec(
            backend.SymbolicDataModel(data_model=Query)
        )
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].schema(), backend.standardize_schema(Query.schema()))
        self.assertEqual(out[1].schema(), backend.standardize_schema(Query.schema()))

        # Case: list output
        class TestModule(modules.Module):
            async def call(self, x):
                assert False  # Should never be called.

            async def compute_output_spec(self, input_schema):
                return [input_schema, input_schema]

        module = TestModule()
        out = await module.compute_output_spec(
            backend.SymbolicDataModel(data_model=Query)
        )

        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].schema(), backend.standardize_schema(Query.schema()))
        self.assertEqual(out[1].schema(), backend.standardize_schema(Query.schema()))

        # Case: dict output
        class TestModule(modules.Module):
            async def call(self, x):
                assert False  # Should never be called.

            async def compute_output_spec(self, input_schema):
                return {"1": input_schema, "2": input_schema}

        module = TestModule()
        out = await module.compute_output_spec(
            backend.SymbolicDataModel(data_model=Query)
        )

        self.assertIsInstance(out, dict)
        self.assertEqual(len(out), 2)
        self.assertEqual(out["1"].schema(), backend.standardize_schema(Query.schema()))
        self.assertEqual(out["2"].schema(), backend.standardize_schema(Query.schema()))
