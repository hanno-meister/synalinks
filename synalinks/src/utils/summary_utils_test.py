from synalinks.src import testing
from synalinks.src.testing import test_utils
from synalinks.src.backend import DataModel, Field
from synalinks.src.programs import Program
from synalinks.src.modules import Input
from synalinks.src.modules import Generator
from synalinks.src.language_models import LanguageModel


class SummaryUtilsTest(testing.TestCase):
    async def test_return_string(self):
        
        class Query(DataModel):
            query: str = Field(
                description="The user query",
            )

        class AnswerWithThinking(DataModel):
            thinking: str = Field(
                description="Your step by step thinking process",
            )
            answer: float = Field(
                description="The correct numerical answer",
            )

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

        x0 = Input(data_model=Query)
        x1 = await Generator(
            data_model=AnswerWithThinking,
            language_model=language_model,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="chain_of_thought",
            description="Usefull to answer in a step by step manner.",
        )
        
        program.summary(return_string=True)