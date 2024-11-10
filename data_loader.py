from openai import OpenAI

import langchain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from typing import List, Tuple
import os
import json

# constants
data_folder = './data'

# Create an LLM instance with the API key
#TODO: replace with Llama
llm = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=os.environ.get('OPENAI_API_KEY'), max_tokens=3000)

def load_data(task="training"):

    assert task in ["training", "evaluation"], \
        f"Invalid task: {task}. Valid inputs are 'training' or 'evaluation'"

    challenges = {}
    solutions = {}

    dataset_path = os.path.join(data_folder, task)

    for data in os.listdir(dataset_path):
        key_name = os.path.splitext(data)[0]

        task_sample = os.path.join(dataset_path, data)

        with open(task_sample) as f:
            task_sample_data = json.load(f)

            challenges[key_name] = {}

            if 'train' in task_sample_data:
                challenges[key_name]['train'] = [
                    {'input': item['input'], 'output': item['output']}
                    for item in task_sample_data['train']
                ]

            if 'test' in task_sample_data:
                challenges[key_name]['test'] = [
                    {'input': item['input']}
                    for item in task_sample_data['test']
                ]

            solutions[key_name] = task_sample_data['test'][0]['output']

    return challenges, solutions


# TODO: use this as starting point and optimize from here and instruct Llama maybe in different ways
def json_task_to_string(challenge_tasks: dict, task_id: str, test_input_index: int) -> str:
    task_json = challenge_tasks[task_id]

    final_output = ""

    train_tasks = task_json['train']
    test_task = task_json['test']

    final_output = "Training Examples\n"

    for i, task in enumerate(train_tasks):
        final_output += f"Example {i + 1}: Input\n["
        for row in task['input']:
            final_output += f"\n{str(row)},"

        final_output += "]\n\n"
        final_output += f"Example {i + 1}: Output\n["

        for row in task['output']:
            final_output += f"\n{str(row)},"

        final_output += "]\n\n"

    final_output += "Test\n["
    for row in test_task[test_input_index]['input']:
        final_output += f"\n{str(row)}"

    final_output += "]\n\nGive me your answer to the Test Input:"

    return final_output

# LangChain output parsing for getting json output
class ARCPrediction(BaseModel):
    prediction: List[List] = Field(..., description="A prediction for a task")


# TODO: add researched prompt engineering methods to help the model get better results
def prompt_eng_method():
    prompt_eng_response = """
                            Your job is to solve the following puzzle. Below, you see multiple input and output pairs
                            as part of the train set. You need to pick up on a pattern from these train input-output pairs.
                            After you have gotten the pattern, you will need to look at the test input, apply the pattern
                            that you have learned, and only return to me, in JSON format, a list containing lists with the output. 
                          """
    return prompt_eng_response


def get_task_prediction(challenge_tasks, task_id, test_input_index) -> List[List]:

    task_string = json_task_to_string(challenge_tasks, task_id, test_input_index)
    
    parser = JsonOutputParser(pydantic_object=ARCPrediction)

    prompt_eng_response = prompt_eng_method() 
    
    prompt = PromptTemplate(
        template=prompt_eng_response +
                    "{format_instructions}\n{task_string}\n",
        input_variables=["task_string"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    ############################# TODO: switch implementation to Llama from GPT #############################

    # wrap chain with LCEL
    chain = prompt | llm | parser
    
    # get prediction from LLM with API call
    output = chain.invoke({"task_string": task_string})

    # get the prediction key. If not there, then get output
    if isinstance(output, dict):
        prediction = output.get('prediction', output)
    else:
        prediction = output

    # check if a list of lists of ints isn't returned. Retries if not correct format
    if not all(isinstance(sublist, list) and all(isinstance(item, int) for item in sublist) for sublist in prediction):
        print("Warning: Output must be a list of lists of integers.")
        print (f"Errored Output: {prediction}")
        raise ValueError("Output must be a list of lists of integers.")
    
    # get shape of prediction
    num_rows = len(prediction)
    num_cols = len(prediction[0]) if num_rows > 0 else 0
    print(f"    Prediction Shape: {num_rows}x{num_cols}\n")
    
    return prediction


def run_model(challenges, NUM_ATTEMPTS=2, RETRY_ATTEMPTS=3, NUM_TASKS=None):

    # submission JSON
    submission = {}

    for i, task_id in enumerate(challenges):
        
        task_attempts = []

        for t, pair in enumerate(challenges[task_id]['test']):
            print(f"Starting task #{i + 1} ({task_id}), pair #{t+1}")

            # dictionary stores attempts for the current test pair
            pair_attempts = {}  

            # run through each prediction attempt
            for attempt in range(1, NUM_ATTEMPTS + 1):
                attempt_key = f"attempt_{attempt}"
                pair_attempts[attempt_key] = [] # Init your attempt

                # get a prediction, retry if failure
                for retry in range(RETRY_ATTEMPTS):
                    try:
                        print(f"    Predicting attempt #{attempt}, retry #{retry + 1}")
                        prediction = get_task_prediction(challenge_tasks=challenges,
                                                         task_id=task_id,
                                                         test_input_index=t)
                        
                        pair_attempts[attempt_key] = prediction
                        break
                    except Exception as e:
                        print(f"Retrying: {e}")
                        if retry == RETRY_ATTEMPTS - 1:
                            pair_attempts[attempt_key] = None  # assign None if all retries fail

            # after attempt, append to task attempts
            task_attempts.append(pair_attempts)

        # append task attempts to the submission
        submission[task_id] = task_attempts

        # if needed, to stop after a select number of tasks
        if NUM_TASKS is not None and i + 1 == NUM_TASKS:
            break

    return submission


if __name__=="__main__":

    # get the challenges and solutions
    # of the according set (train or eval)
    challenges, solutions = load_data(task="training")

    # TODO: Add Llama Fine Tuning with unsloth to train the model

    # only 1 JSON task is given to the model and 10 predictions are attempted
    # TODO: optimize so that attempts use previous attempts results to help maybe?
    submission = run_model(challenges, NUM_TASKS=1, NUM_ATTEMPTS=10)