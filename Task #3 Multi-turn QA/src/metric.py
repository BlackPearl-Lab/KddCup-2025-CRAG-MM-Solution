import json
import time
# from openai import OpenAI
from p_tqdm import p_map
from rich.console import Console
from typing import Optional, Dict, List, Any, Callable

console = Console()
MAX_API_RETRIES = 3
DEFAULT_EVAL_MODEL = "gpt-4o-mini"

def aggregate_evaluation_results(
    results: Dict[str, Any], eval_results: List[Dict[str, Any]]
) -> None:
    """
    Aggregate individual evaluation results into the final results dictionary.

    Args:
        results: The results dictionary to update
        eval_results: List of individual evaluation results
    """
    # Aggregate results
    for result in eval_results:
        results["total"] += 1
        results["correct_exact"] += result["is_exact_match"]
        results["correct"] += result["is_correct"]
        results["miss"] += result["is_miss"]
        results["examples"].append(result)

    # Calculate metrics
    n = results["total"]
    if n > 0:
        results["exact_accuracy"] = results["correct_exact"] / n
        results["accuracy"] = results["correct"] / n
        results["missing"] = results["miss"] / n
        results["hallucination"] = (n - results["correct"] - results["miss"]) / n
        # results["score"] = (2 * results["correct"] + results["miss"]) / n - 1
        results["score"] = results["accuracy"] - results["hallucination"]
    else:
        results["exact_accuracy"] = 0
        results["accuracy"] = 0
        results["missing"] = 0
        results["hallucination"] = 0
        results["score"] = 0

def attempt_api_call(client, model_name, messages, max_retries=MAX_API_RETRIES):
    """
    Attempt a structured API call with retries upon encountering specific errors.

    Args:
        client: The API client to use
        model_name: The model to query
        messages: List of message objects for the conversation
        max_retries: Maximum number of retry attempts

    Returns:
        Dictionary with accuracy and raw response, or None if all attempts fail
    """
    for attempt in range(max_retries):
        try:
            # # Use completion.create instead of parse to avoid using the EvaluationResult class in worker processes
            # response = client.chat.completions.create(
            #     model=model_name,
            #     messages=messages,
            #     response_format={"type": "json_object"},
            # )
            import openai
            # API_KEY = "1760530884454920218"
            API_KEY = "1914617170185932860"
            BASE_URL = "https://aigc.sankuai.com/v1/openai/native"

            openai.api_key = API_KEY
            openai.api_base = BASE_URL

            response = openai.ChatCompletion.create(
                model=model_name,
                user="zhangzijian14",
                messages=messages,
                response_format={"type": "json_object"},
            )

            # Parse the JSON content manually
            content = response.choices[0].message.content
            time.sleep(2)
            import random
            sleep_time = random.randint(1, 5)
            time.sleep(sleep_time)
            # print(content)
            try:
                result_json = json.loads(content)
                accuracy = result_json.get("accuracy", False)
                # Return both the parsed result and raw JSON for debugging
                return {"accuracy": accuracy, "raw": content}
            except json.JSONDecodeError:
                console.print(
                    f"[yellow]Failed to parse JSON from response: {content}[/yellow]"
                )
                if attempt == max_retries - 1:
                    return {"accuracy": False, "raw": content}
        except Exception as e:
            time.sleep(10)
            console.print(
                f"[yellow]API call failed on attempt {attempt + 1}, retrying: {str(e)}[/yellow]"
            )
            if attempt == max_retries - 1:
                console.print(
                    f"[red]Failed after {max_retries} attempts: {str(e)}[/red]"
                )
    return None

def get_system_message() -> str:
    """Returns the system message for the evaluator."""
    return (
        "You are an expert evaluator for question answering systems. "
        "Your task is to determine if a prediction correctly answers a question based on the ground truth.\n\n"
        "Rules:\n"
        "1. The prediction is correct if it captures all the key information from the ground truth.\n"
        "2. The prediction is correct even if phrased differently as long as the meaning is the same.\n"
        "3. The prediction is incorrect if it contains incorrect information or is missing essential details.\n"
        "Output a JSON object with a single field 'accuracy' whose value is true or false."
    )

def evaluate_response(
    examples, eval_model_name
) -> dict:
    agent_response = str(examples["agent_response"])
    ground_truth = str(examples["ground_truth"])
    query = str(examples["query"])
    
    # Initial evaluation
    is_idk = "i don't know" in agent_response.lower()
    is_exact_match = agent_response.strip().lower() == ground_truth.strip().lower()
    is_semantically_correct = False
    api_response = None

    # Determine correctness
    is_correct = is_exact_match  # Start with exact match

    # If not an exact match and we have an evaluation model, use semantic evaluation
    # print('****** step-3 ******', is_exact_match, eval_model_name)
    if not is_idk and not is_exact_match and eval_model_name:
        # Create a new OpenAI client inside this function
        # local_openai_client = OpenAI(
        #     api_key = "1760530884454920218",
        #     base_url = "https://aigc.sankuai.com/v1/openai/native"
        # )
        local_openai_client = "mock"

        # Prepare API call - same format regardless of IDK status
        messages = [
            {"role": "system", "content": get_system_message()},
            {
                "role": "user",
                "content": f"Question: {query}\nGround truth: {ground_truth}\nPrediction: {agent_response}\n",

            },
        ]

        # Make the API call
        api_response = attempt_api_call(local_openai_client, eval_model_name, messages)

        if api_response:
            is_semantically_correct = api_response["accuracy"]
            # Only update is_correct if it's not an IDK response
            is_correct = is_semantically_correct
    if is_exact_match:
        # Exact matches are always semantically correct
        is_semantically_correct = True

    # Return a dictionary with evaluation results
    return {
        "agent_response": agent_response,
        "ground_truth": ground_truth,
        "query": query,
        "is_exact_match": is_exact_match,
        "is_correct": is_correct,
        "is_miss": is_idk,
        "is_semantically_correct": is_semantically_correct,
        "api_response": api_response,
    }

def parallel_evaluate(examples):
    return evaluate_response(examples, DEFAULT_EVAL_MODEL)

def single_turn_evaluate(examples):
    results = {
        "correct_exact": 0,
        "correct": 0,
        "miss": 0,
        "total": 0,
        "examples": [],
        "metadata": {
            "eval_model": DEFAULT_EVAL_MODEL,
        },
    }

    eval_results = p_map(
        parallel_evaluate,
        examples,
        desc="Evaluating responses",
        num_cpus=1,
        disable=False,
    )
    aggregate_evaluation_results(results, eval_results)


    return results