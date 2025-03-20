from datetime import datetime
import json
import os
import openai
import random
import numpy as np
import dotenv

dotenv.load_dotenv()
# ---------------------------------------------------------------------
# 1. Setup and configuration
# ---------------------------------------------------------------------

# Range of digits to test
MIN_DIGITS = 1
MAX_DIGITS = 20

# Number of times to query the model for each pair of digit lengths
NUM_TRIALS = 5

# Name of the OpenAI model to use
MODEL_NAME = "gpt-3.5-turbo"

client = openai.OpenAI(
    api_key=os.environ["ARCADE_API_KEY"],
    base_url="http://api.arcade.dev/v1",
)

# Define the checkpoint file path
CHECKPOINT_FILE = "checkpoint.json"


# ---------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------


def get_random_number(num_digits: int) -> int:
    """
    Generates a random integer with exactly `num_digits` digits.
    For example, if num_digits=3, returns something in [100, 999].
    """
    if num_digits < 1:
        raise ValueError("num_digits must be >= 1")
    lower_bound = 10 ** (num_digits - 1)
    upper_bound = (10**num_digits) - 1
    return random.randint(lower_bound, upper_bound)


def query_model_for_product(num1: int, num2: int) -> str:
    """
    Queries the language model with the specified prompt and
    returns the raw response text.
    """
    prompt = (
        f"Calculate the product of {num1} and {num2}.\n"
        "Please provide the final answer in the format: Final Answer: [result no commas]"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # For maximum determinism
        tools=[
            "Math.Multiply",
            "Math.Add",
            "Math.Subtract",
            "Math.Divide",
            "Math.Sqrt",
            "Math.SumList",
            "Math.SumRange",
            "Math.Mod",
            "Math.Power",
            "Math.AbsVal",
            "Math.Log",
            "Math.Avg",
            "Math.Factorial",
            "Math.Ceil",
            "Math.Floor",
            "Math.RoundNum",
            "Math.Gcd",
            "Math.Lcm",
            "Math.Median",
            "Math.DegToRad",
            "Math.RadToDeg",
        ],
        tool_choice="generate",
    )

    text = response.choices[0].message.content
    return text


def parse_final_answer(response_text: str) -> str:
    """
    Parses the model's response to extract the number after "Final Answer:".
    If parsing fails, returns None.
    """
    marker = "Final Answer:"
    if marker in response_text:
        after_marker = response_text.split(marker, 1)[1].strip()
        possible_answer = after_marker.split()[0]
        possible_answer = possible_answer.strip(",.")
        return possible_answer
    else:
        return None


def write_results_to_file(accuracy_matrix):
    results_dict = {}
    for i in range(MIN_DIGITS, MAX_DIGITS + 1):
        results_dict[str(i)] = {}
        for j in range(MIN_DIGITS, MAX_DIGITS + 1):
            results_dict[str(i)][str(j)] = accuracy_matrix[i - 1, j - 1]

    output_dict = {
        "accuracy_matrix": results_dict,
        "max_digits": MAX_DIGITS,
        "min_digits": MIN_DIGITS,
        "model": MODEL_NAME,
        "num_trials": NUM_TRIALS,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }
    filename = (
        f"results_{MODEL_NAME}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    )
    with open(filename, "w") as json_file:
        json.dump(output_dict, json_file, indent=4)
    print(f"Final results written to {filename}")


def save_checkpoint(data):
    """Write the checkpoint data to disk atomically."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------------------
# 3. Load checkpoint (if any) and Main logic: gather data
# ---------------------------------------------------------------------

# Load checkpoint if it exists; otherwise, create an empty progress dict.
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        checkpoint = json.load(f)
    print("Resuming from existing checkpoint.")
else:
    checkpoint = {"progress": {}}
    print("No checkpoint found; starting fresh.")

# We will store accuracy results in a 2D NumPy array.
accuracy_matrix = np.zeros((MAX_DIGITS, MAX_DIGITS))

for i in range(MIN_DIGITS, MAX_DIGITS + 1):
    # Ensure progress structure exists for digit count i.
    if str(i) not in checkpoint["progress"]:
        checkpoint["progress"][str(i)] = {}

    for j in range(MIN_DIGITS, MAX_DIGITS + 1):
        pair_key = str(j)
        # Initialize progress for this digit pair if not present.
        if pair_key in checkpoint["progress"][str(i)]:
            completed_trials = checkpoint["progress"][str(i)][pair_key].get(
                "completed_trials", 0
            )
            correct_count = checkpoint["progress"][str(i)][pair_key].get(
                "correct_count", 0
            )
        else:
            completed_trials = 0
            correct_count = 0
            checkpoint["progress"][str(i)][pair_key] = {
                "completed_trials": 0,
                "correct_count": 0,
            }

        # Process remaining trials for this (i, j) pair.
        for trial in range(completed_trials, NUM_TRIALS):
            n1 = get_random_number(i)
            n2 = get_random_number(j)
            print(f"Trial {trial + 1} of {NUM_TRIALS} for {i}x{j}: {n1} x {n2}")

            true_product = n1 * n2
            response_text = query_model_for_product(n1, n2)
            parsed_answer = parse_final_answer(response_text)

            if parsed_answer is not None:
                try:
                    model_product = int(parsed_answer)
                    if model_product == true_product:
                        correct_count += 1
                    else:
                        print(
                            f"Model product {model_product} does not match true product {true_product}"
                        )
                except ValueError:
                    print(f"Could not parse {parsed_answer} as an integer")
            else:
                print("No valid answer parsed from the model response.")

            # Update the progress for this pair.
            completed_trials += 1
            checkpoint["progress"][str(i)][pair_key]["completed_trials"] = (
                completed_trials
            )
            checkpoint["progress"][str(i)][pair_key]["correct_count"] = correct_count

            # Write checkpoint after each trial.
            save_checkpoint(checkpoint)

        # Calculate accuracy for this digit pair and update our accuracy matrix.
        accuracy = (correct_count / NUM_TRIALS) * 100
        accuracy_matrix[i - 1, j - 1] = accuracy

# ---------------------------------------------------------------------
# 4. Store final results to disk
# ---------------------------------------------------------------------

write_results_to_file(accuracy_matrix)
