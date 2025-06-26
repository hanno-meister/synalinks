# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)Add commentMore actions

import asyncio
import time
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Input
from synalinks.src.modules.agents.react_agent import ParallelReACTAgent
from synalinks.src.programs import Program


class ReACTAgentTest(testing.TestCase):
    @patch("litellm.completion")
    async def test_basic_flow_with_parallel_actions(self, mock_completion):
        # Shared log to capture timestamps from both functions
        execution_log = []

        class Query(DataModel):
            query: str

        class FinalAnswer(DataModel):
            answer_sum: float
            answer_wc: int

        async def calculate(expression: str):
            """
            Calculate a basic math expression (supports +, −, *, /).

            Args:
                expression (str): The expression to evaluate, e.g. "12 + 15".

            Returns:
                dict: {"result": float | None, "log": str}
            """
            start_time = time.time()
            execution_log.append({"function": "calculate", "event": "start", "timestamp": start_time})

            await asyncio.sleep(0.1)

            if not all(char in "0123456789+-*/(). " for char in expression):
                end_time = time.time()
                execution_log.append({"function": "calculate", "event": "end", "timestamp": end_time})
                return {
                    "result": None,
                    "log": "Error: invalid characters in expression",
                }
            try:
                # Evaluate the mathematical expression safely
                result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
                end_time = time.time()
                execution_log.append({"function": "calculate", "event": "end", "timestamp": end_time})
                return {
                    "result": result,
                    "log": "Successfully executed",
                }
            except Exception as e:
                end_time = time.time()
                execution_log.append({"function": "calculate", "event": "end", "timestamp": end_time})
                return {
                    "result": None,
                    "log": f"Error: {e}",
                }

        async def word_count(text: str):
            """
            Count the number of whitespace-separated words in *text*.

            Args:
                text (str): The string to analyse. Words are separated by whitespace.

            Returns:
                dict: {"result": int | None, "log": str}
            """
            start_time = time.time()
            execution_log.append({"function": "word_count", "event": "start", "timestamp": start_time})

            await asyncio.sleep(0.1)

            if not isinstance(text, str) or not text.strip():
                end_time = time.time()
                execution_log.append({"function": "word_count", "event": "end", "timestamp": end_time})
                return {
                    "result": None,
                    "log": "Error: Expression must be a non-empty string"
                }
            try:
                word_count_result = len(text.split())
                end_time = time.time()
                execution_log.append({"function": "word_count", "event": "end", "timestamp": end_time})
                
                return {
                    "result": word_count_result,
                    "log": "Successfully executed"
                }
            except Exception as e:
                end_time = time.time()
                execution_log.append({"function": "word_count", "event": "end", "timestamp": end_time})
                return {
                    "result": None,
                    "log": f"Error: {e}",
                }

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

        decision_continue = (
            """{
                "thinking": "I'll need to run some tools before I can answer.", 
                "choice": "continue"
            }"""
        )

        decision_tools = (
            """{
                "thinking": "I should add 12 + 15 and count the words in 'hello parallel agent'.", 
                "choices": ["calculate", "word_count"]
            }"""
        )

        inference_response_calc = """{"expression": "12 + 15"}"""
        inference_response_wc = """{"text": "hello parallel agent"}"""

        decision_continue_1 = (
            """{
                "thinking": "Now I know the answer so I finished, so I select `finish`.", 
                "choice": "finish"
            }"""
        )

        final_answer = """{"answer_sum": 27.0, "answer_wc": 3}"""

        mock_responses = [
            {"choices": [{"message": {"content": decision_continue}}]},
            {"choices": [{"message": {"content": decision_tools}}]},
            {"choices": [{"message": {"content": inference_response_calc}}]},
            {"choices": [{"message": {"content": inference_response_wc}}]},
            {"choices": [{"message": {"content": decision_continue_1}}]},
            {"choices": [{"message": {"content": final_answer}}]},
        ]
        mock_completion.side_effect = mock_responses

        x0 = Input(data_model=Query)
        x1 = await ParallelReACTAgent(
            data_model=FinalAnswer,
            language_model=language_model,
            functions=[calculate, word_count],
            max_iterations=3,
            return_inputs_with_trajectory=True
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
        )

        result = await program(
            Query(
                query=(
                    "You have a basket with 12 apples. "
                    "Your friend gives you 15 more apples. "
                    "How many apples in total? "
                    "And how many words are in 'hello parallel agent'?"
                )
            )
        )
        
        # Verify the results
        self.assertEqual(result.get("answer_sum"), 27.0)
        self.assertEqual(result.get("answer_wc"), 3)

        # Verify parallel execution by checking timestamp overlap
        self._verify_parallel_execution(execution_log)

    def _verify_parallel_execution(self, execution_log):
        """
        Verify that the functions executed in parallel by checking timestamp overlap.
        """
        # Find start and end times for each function
        calculate_times = {}
        word_count_times = {}
        
        for entry in execution_log:
            if entry["function"] == "calculate":
                calculate_times[entry["event"]] = entry["timestamp"]
            elif entry["function"] == "word_count":
                word_count_times[entry["event"]] = entry["timestamp"]
        
        # Check for time overlap (parallel execution)
        calc_start = calculate_times["start"]
        calc_end = calculate_times["end"]
        wc_start = word_count_times["start"]
        wc_end = word_count_times["end"]
        
        # Functions are running in parallel if their execution windows overlap
        # This means: max(start1, start2) < min(end1, end2)
        overlap_start = max(calc_start, wc_start)
        overlap_end = min(calc_end, wc_end)
        
        self.assertLess(
            overlap_start, 
            overlap_end, 
            f"Functions should execute in parallel. "
            f"calculate: {calc_start:.3f}-{calc_end:.3f}, "
            f"word_count: {wc_start:.3f}-{wc_end:.3f}"
        )
