# ARC-AGI-Public data test

## Claude 3.5 Sonnet (06/2024, 10/2024), Open AI GPT-4o, DeepSeek v3/r1 comparison

See the following report for the background information: [ARC-AGI-Claude-3-5-Sonnet-and-GPT-4o-comparison report](https://www.notion.so/mesokosmos/ARC-AGI-Claude-3-5-Sonnet-and-GPT-4o-comparison-1d90907ca1784832a0acc315882b1dc2)

This repository contains the necessary scripts and instructions to compare the performance of Claude 3.5 and GPT-4o in direct prompting approach to public evaluation data in ARC-AGI benchmark.

## Prerequisites

To reproduce the results, you will need:

- An account with Anthropic.
- A minimum balance of $10 in your Anthropic account.
- An API key from Anthropic, which should be set in the `.env` file.

## Installation

1. **Install Required Modules**:
   - Ensure all necessary Python modules are installed by running the following command:
     ```
     pip install -r requirements.txt
     ```

## Running the Tests

1. **Execute the Test Script**:
   - Run the following command to execute all tests for `Claude 3.5 Sonnet`. This process will take approximately 30 minutes to run all tests:
     ```
     python run_tests.py
     ```

You may alter runnig only a limited test files by adjusting list index from run_tests.py:226:
    ```
    evaluation_files -> evaluation_files[:1]
    ```

## Analyzing Results

1. **Generate Analysis Plots**:
   - After running the tests, execute the `analyze.py` script to view the results in graphical format. Ensure that any required libraries for viewing plots are installed:
     ```
     python analyze.py
     ```

  - You may also plot all correctly predicted patterns in a single multiplot grid:
    ```
    python plot_correct.py
    ```

## Configuration for `GPT-4o`

- If you want to conduct tests with GPT-4o, you need to modify the `llmClientName` parameter in the `main` function of both `run_tests.py` and `analyze.py` scripts:
  - Change `llmClientName = None` to `llmClientName = "openai"`.

You also need balance (likely a bit more than with the Claude) in the account and API KEY (optionally organization and project keys) from OpenAI to be set in the `.env` file.

## Configuration for `DeepSeek v3`

- If you want to conduct tests with DeepSeek v3 or r1, you need to modify the `llmClientName` parameter in the `main` function of both `run_tests.py` and `analyze.py` scripts:
  - Change `llmClientName = None` to `llmClientName = "deepseek"`.
  - Change model name in `return send_request_to_openai(client, userMessage, "deepseek-chat")` or `"deepseek-reasoner"`

You also need balance (likely a bit less than with the Claude) in the account and API KEY (optionally organization and project keys) from DeepSeek to be set in the `.env` file.

## Configuration for `Gemini`

pip install --upgrade google-cloud-aiplatform
brew install --cask google-cloud-sdk
gcloud auth application-default login

***

I have written an essay about AGI in Finnish that you can read from: [AGI – Yleistekoäly: Totta vai tarua, nyt vai huomenna?](https://mesokosmos.com/2024/06/21/agi-yleistekoaly-totta-vai-tarua-nyt-vai-huomenna/)
