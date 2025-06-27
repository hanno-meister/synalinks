# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)Add commentMore actions

import asyncio
import time
from unittest.mock import patch
import sys

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

        # Mocks for first round
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

        # Mocks for second round
        decision_continue_2 = (
            """{
                "thinking": "I got some results, but let me run the tools again to double-check.", 
                "choice": "continue"
            }"""
        )

        decision_tools_2 = (
            """{
                "thinking": "Let me run both tools again to verify the results.", 
                "choices": ["calculate", "word_count"]
            }"""
        )

        decision_finish = (
            """{
                "thinking": "Now I know the answer so I finished, so I select `finish`.", 
                "choice": "finish"
            }"""
    )

        final_answer = """{"answer_sum": 27.0, "answer_wc": 3}"""

        mock_responses = [
            # First round
            {"choices": [{"message": {"content": decision_continue}}]},
            {"choices": [{"message": {"content": decision_tools}}]},
            {"choices": [{"message": {"content": inference_response_calc}}]},
            {"choices": [{"message": {"content": inference_response_wc}}]},
            
            # Second round
            {"choices": [{"message": {"content": decision_continue_2}}]},
            {"choices": [{"message": {"content": decision_tools_2}}]},
            {"choices": [{"message": {"content": inference_response_calc}}]},
            {"choices": [{"message": {"content": inference_response_wc}}]},
            
            # Final
            {"choices": [{"message": {"content": decision_finish}}]},
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

        # Verify parallel execution by checking timestamp overlap
        self._verify_parallel_execution(execution_log)

    def _verify_parallel_execution(self, execution_log):
        """
        Verify that the functions executed in parallel in both iterations.
        """
        # Group execution events by round
        rounds = []
        current_round = {}
        
        for entry in execution_log:
            func_name = entry["function"]
            event = entry["event"]
            timestamp = entry["timestamp"]
            
            if func_name not in current_round:
                current_round[func_name] = {}
            
            current_round[func_name][event] = timestamp
            
            # When both functions have completed, we've finished a round
            if (len(current_round) == 2 and 
                all("end" in times for times in current_round.values())):
                rounds.append(current_round)
                current_round = {}
        
        sys.stdout.write(f"Found {len(rounds)} execution rounds\n")
        
        # Verify parallel execution for each round
        for i, round_data in enumerate(rounds):
            sys.stdout.write(f"\nRound {i+1} timing analysis:\n")
            
            calc_start = round_data["calculate"]["start"]
            calc_end = round_data["calculate"]["end"]
            wc_start = round_data["word_count"]["start"]
            wc_end = round_data["word_count"]["end"]
            
            sys.stdout.write(f"  calculate: {calc_start:.3f}-{calc_end:.3f} (duration: {calc_end-calc_start:.3f}s)\n")
            sys.stdout.write(f"  word_count: {wc_start:.3f}-{wc_end:.3f} (duration: {wc_end-wc_start:.3f}s)\n")
            
            # Check for time overlap (parallel execution)
            overlap_start = max(calc_start, wc_start)
            overlap_end = min(calc_end, wc_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            sys.stdout.write(f"  overlap: {overlap_start:.3f}-{overlap_end:.3f} (duration: {overlap_duration:.3f}s)\n")
            
            self.assertLess(
                overlap_start, 
                overlap_end, 
                f"Functions should execute in parallel in round {i+1}. "
                f"calculate: {calc_start:.3f}-{calc_end:.3f}, "
                f"word_count: {wc_start:.3f}-{wc_end:.3f}"
            )
        
        # Verify we had the expected number of rounds
        self.assertEqual(len(rounds), 2, "Expected exactly 2 rounds of parallel execution")
