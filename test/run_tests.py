import os
import numpy as np
import json
from pathlib import Path
from dotenv import load_dotenv
import anthropic
import matplotlib.pyplot as plt
from openai import OpenAI
from datetime import datetime
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
import time
import google.api_core.exceptions

# Load environment variables from .env file
load_dotenv()

safety_settings = None

def load_evaluation_files(data_dir):
    #print(f"Loading evaluation files from {data_dir}")
    files = list(Path(data_dir).glob("*.json"))
    #print(f"Found {len(files)} files")
    return files

def format_user_message(train_data, test_input):
    user_message = json.dumps({
        "train": train_data,
        "test": [{"input": test_input, "output": ["..."]}]
    })
    #print(f"Formatted user message: {user_message[:100]}...")  # Print first 100 chars for brevity
    return user_message + "\n\nGet test output as JSON array. Nothing else, no intro, no summary, just JSON text code block."

def send_request_to_google_(client, userMessage, model="gemini-1.5-pro-002", temperature=0):
    #print(f"Sending request to LLM with message: {userMessage[:100]}...")  # Print first 100 chars for brevity

    responses = client.generate_content(
        [userMessage],
        generation_config={
            "max_output_tokens": 8192,
            "temperature": temperature,
            "top_p": 0.95,
        },
        safety_settings=safety_settings,
        stream=True,
    )

    content = ''.join([response.text for response in responses])
    print(f"Received response: {content[:50]}...")  # Print first 50 chars for brevity
    return content

def send_request_to_google(client, userMessage, model="gemini-1.5-pro-002", temperature=0):
    max_retries = 6
    backoff_factor = 2
    initial_delay = 1  # in seconds

    for attempt in range(max_retries):
        try:
            responses = client.generate_content(
                [userMessage],
                generation_config={
                    "max_output_tokens": 8192,
                    "temperature": temperature,
                    "top_p": 0.95,
                },
                safety_settings=safety_settings,
                stream=True,
            )

            content = ''.join([response.text for response in responses])
            print(f"Received response: {content[:50]}...")  # Print first 50 chars for brevity
            return content

        except google.api_core.exceptions.ResourceExhausted as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (backoff_factor ** attempt)
                print(f"Quota exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Quota exceeded.")
                raise

def send_request_to_anthropic(client, userMessage, model="claude-3-5-sonnet-20240620", temperature=0):
    #print(f"Sending request to LLM with message: {userMessage[:100]}...")  # Print first 100 chars for brevity
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": userMessage
            }
        ]
    )
    print(f"Received response: {response.content[0].text[:50]}...")  # Print first 50 chars for brevity
    return response.content[0].text

def send_request_to_openai(client, userMessage, model="gpt-4o", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": userMessage
            }
        ],
        temperature=temperature,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    message_content = response.choices[0].message.content
    print(f"Received response: {message_content[:50]}...")  # Print first 50 chars for brevity
    if model == "deepseek-reasoner":
        #print(response.choices[0].message.reasoning_content)
        reasoning_content = response.choices[0].message.reasoning_content
        log_file="deepseek_reasoning_log.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as file:
            file.write(f"[{timestamp}] {reasoning_content}\n")
    return message_content

def send_request_to_deepseek(client, userMessage, model="deepseek-reasoner", temperature=0):
    return send_request_to_openai(client, userMessage, model, temperature)

def get_depth(nested_list):
    if not isinstance(nested_list, list):
        return 0
    if not nested_list:
        return 1
    return 1 + max(get_depth(item) for item in nested_list)

def flatten(nested_list, target_depth = 3):
    current_depth = get_depth(nested_list)
    while current_depth > target_depth:
        nested_list = [item for sublist in nested_list for item in sublist]
        current_depth = get_depth(nested_list)
    return nested_list

def extract_and_parse_json_block(text):
    if not text:
        return []

    stack = []
    start_index = None
    results = []
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if char == '"' and not escaped:
            in_string = not in_string
        elif char == '{' and not in_string:
            stack.append(char)
            if len(stack) == 1:
                start_index = index
        elif char == '[' and not in_string:
            stack.append(char)
            if len(stack) == 1 and start_index is None:
                start_index = index
        elif char == '}' and not in_string and stack and stack[-1] == '{':
            stack.pop()
            if not stack:
                end_index = index + 1
                json_block = text[start_index:end_index]
                try:
                    results.append(json.loads(json_block))
                    start_index = None
                except json.JSONDecodeError:
                    continue
        elif char == ']' and not in_string and stack and stack[-1] == '[':
            stack.pop()
            if not stack:
                end_index = index + 1
                json_block = text[start_index:end_index]
                try:
                    results.append(json.loads(json_block))
                    start_index = None
                except json.JSONDecodeError:
                    continue
        escaped = (char == '\\') and not escaped

    return flatten(results)

def compare_outputs(predicted_output, true_output):
    comparison_result = predicted_output == true_output
    print(f"Comparison result: {comparison_result}")
    return comparison_result

def log_results(log_dir, file_name, raw_response, true_output, comparison_flag, error_flag):
    log_file = os.path.join(log_dir, f"{file_name}.log")
    #print(f"Logging results to {log_file}")
    with open(log_file, "w") as f:
        json.dump({"response": raw_response, "true_output": true_output, "comparison_flag": comparison_flag, "error_flag": error_flag}, f)
    #print(f"Results logged for {file_name}")

def create_and_save_plot(test_input, true_output, predicted_output, file_name):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    test_input = np.array(test_input)
    true_output = np.array(true_output)

    axes[0].imshow(test_input, aspect='equal', cmap='viridis')
    axes[0].set_title('Test Input')
    axes[0].set_xticks(np.arange(-0.5, test_input.shape[1], 1))
    axes[0].set_yticks(np.arange(-0.5, test_input.shape[0], 1))
    axes[0].grid(True, color='white', linestyle='-', linewidth=0.5)
    axes[0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    axes[1].imshow(true_output, aspect='equal', cmap='viridis')
    axes[1].set_title('True Output')
    axes[1].set_xticks(np.arange(-0.5, true_output.shape[1], 1))
    axes[1].set_yticks(np.arange(-0.5, true_output.shape[0], 1))
    axes[1].grid(True, color='white', linestyle='-', linewidth=0.5)
    axes[1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    if predicted_output is not None:
        try:
            predicted_output = np.array(predicted_output) if predicted_output else None
            axes[2].imshow(predicted_output, aspect='equal', cmap='viridis')
            axes[2].set_title('Predicted Output')
            axes[2].set_xticks(np.arange(-0.5, predicted_output.shape[1], 1))
            axes[2].set_yticks(np.arange(-0.5, predicted_output.shape[0], 1))
            axes[2].grid(True, color='white', linestyle='-', linewidth=0.5)
            axes[2].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        except (ValueError, TypeError):
            print(f"Prediction Image Output Error {file_name}")
            axes[2].set_title('Predicted Output')
            axes[2].text(0.5, 0.5, "Invalid Image Data", horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)
    
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    #print(f"Plot saved to {file_name}")

def create_and_save_examples_plot(train_data, file_name):
    num_examples = len(train_data)
    fig, axes = plt.subplots(num_examples, 2, figsize=(10, 5 * num_examples))
    
    for i, data in enumerate(train_data):
        input_data = np.array(data["input"])
        output_data = np.array(data["output"])
        
        axes[i, 0].imshow(input_data, aspect='equal', cmap='viridis')
        axes[i, 0].set_title(f'Train Input {i+1}')
        axes[i, 0].set_xticks(np.arange(-0.5, input_data.shape[1], 1))
        axes[i, 0].set_yticks(np.arange(-0.5, input_data.shape[0], 1))
        axes[i, 0].grid(True, color='white', linestyle='-', linewidth=0.5)
        axes[i, 0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

        axes[i, 1].imshow(output_data, aspect='equal', cmap='viridis')
        axes[i, 1].set_title(f'Train Output {i+1}')
        axes[i, 1].set_xticks(np.arange(-0.5, output_data.shape[1], 1))
        axes[i, 1].set_yticks(np.arange(-0.5, output_data.shape[0], 1))
        axes[i, 1].grid(True, color='white', linestyle='-', linewidth=0.5)
        axes[i, 1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    print(f"Examples plot saved to {file_name}")

def main(llmClientName=None):
    data_dir = "../data/evaluation"
    log_dir = "./log_"+llmClientName if llmClientName else "./log"
    img_dir = "./img_"+llmClientName if llmClientName else "./img"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    print(f"Log directory created at {log_dir}")
    
    if llmClientName == "openai":
        llm_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), 
            organization=os.getenv("OPENAI_ORGANIZATION_KEY"), 
            project=os.getenv("OPENAI_PROJECT_KEY")
        )
    elif llmClientName == "deepseek":
        llm_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    elif llmClientName == "google":
        vertexai.init(project="gen-lang-client-0064249622", location="us-central1")
        llm_client = GenerativeModel(
            "gemini-1.5-pro-002",
        )
        safety_settings = [
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
        ]
    else:
        llm_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    evaluation_files = load_evaluation_files(data_dir)

    r = 250
    for file_path in evaluation_files[r:400]:
        r += 1
        print("--------------------------------------------------")
        print(f"Processing file #{r}: {file_path}")
        with open(file_path, "r") as file:
            data = json.load(file)
            train_data = data["train"]
            test_data = data["test"]
            print(f"Loaded train and test data from {file_path}")
            
            examples_img_file_name = os.path.join(img_dir, f"{file_path.stem}_examples.png")
            if not os.path.exists(examples_img_file_name):
                create_and_save_examples_plot(train_data, examples_img_file_name)

            for test_item in test_data[:1]:
                test_input = test_item["input"]
                true_output = test_item["output"]
                log_file = os.path.join(log_dir, f"{file_path.stem}.log")
                
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        log_data = json.load(f)
                        raw_response = log_data.get("response", None)
                        parsed_responses = extract_and_parse_json_block(raw_response)
                        json_response = parsed_responses[0] if parsed_responses else None
                else:
                    
                    userMessage = format_user_message(train_data, test_input)
                    if llmClientName == "openai":
                        # gpt-4o
                        raw_response = send_request_to_openai(llm_client, userMessage, "gpt-4o", 0.0)
                    elif llmClientName == "deepseek":
                        # deepseek-reasoner
                        # 22.10.2025: temperature 1.0 which is supposed to be for data analysis
                        # gives much worae answers than 0.0, dropping the success rate to 6%
                        raw_response = send_request_to_deepseek(llm_client, userMessage, "deepseek-chat", 0.0)
                    elif llmClientName == "google":
                        # gemini-1.5-pro-002, gemini-2.0-flash-exp
                        raw_response = send_request_to_google(llm_client, userMessage, "gemini-1.5-pro-002", 0.0)
                    else:
                        # claude-3-5-sonnet-20240620"
                        raw_response = send_request_to_anthropic(llm_client, userMessage, "claude-3-5-sonnet-20241022", 0.0)

                    error_flag = False
                    try:
                        parsed_responses = extract_and_parse_json_block(raw_response)
                        json_response = parsed_responses[0] if parsed_responses else None
                        comparison_flag = compare_outputs(json_response, true_output)
                    except (json.JSONDecodeError, TypeError):
                        print(f"JSON decode error for response: {raw_response}")
                        json_response = None
                        comparison_flag = False
                        error_flag = True

                    log_results(log_dir, file_path.stem, raw_response, true_output, comparison_flag, error_flag)

                img_file_name = os.path.join(img_dir, f"{file_path.stem}.png")
                if not os.path.exists(img_file_name):
                    create_and_save_plot(test_input, true_output, json_response, img_file_name)

if __name__ == "__main__":
    llmClientName = 'google'  # None for Anthropic Claude 3.5 Sonnet, "deepseek" for v3, or "openai" for GPT-4o
    main(llmClientName)
