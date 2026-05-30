from __future__ import annotations

from dataflow.operators.eval import MetaScorer
from dataflow.serving.APILLMServing_request import APILLMServing_request

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import LLMAsSelector


def selector():
    return {
        "input": "/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl",
        "output": "data/output_llm.jsonl",
        "selector": CustomOmegaConfig.of(
            LLMAsSelector,
            k=100000,
            text_key="text",
            scorer=CustomOmegaConfig.of(
                MetaScorer,
                llm_serving=CustomOmegaConfig.of(
                    APILLMServing_request,
                    api_url="https://api.openai.com/v1/chat/completions",
                    key_name_of_api_key="DF_API_KEY",
                    model_name="gpt-4o",
                    max_workers=10,
                ),
                dimensions=[
                    {
                        "dimension_name": "Mathematical Correctness",
                        "description": "Evaluate whether the mathematical reasoning and final answer are correct.",
                        "example_list": [
                            {"text": "The derivative of x^2 is 2x by the power rule.", "score": "5"},
                            {"text": "The derivative of x^2 is x by dividing the exponent.", "score": "1"},
                        ],
                    },
                    {
                        "dimension_name": "Solution Clarity",
                        "description": "Assess how clearly the solution steps are explained and organized.",
                        "example_list": [
                            {"text": "Step 1: Factor the quadratic. Step 2: Apply zero product property. Step 3: Solve each factor.", "score": "5"},
                            {"text": "Just do the math and you get x=3.", "score": "2"},
                        ],
                    },
                    {
                        "dimension_name": "Educational Value",
                        "description": "Determine whether the problem and solution provide meaningful learning potential for mathematics.",
                        "example_list": [
                            {"text": "This problem demonstrates the application of the chain rule in multivariable calculus.", "score": "5"},
                            {"text": "What is 1+1?", "score": "1"},
                        ],
                    },
                ],
            ),
        ),
    }
