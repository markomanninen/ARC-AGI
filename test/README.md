# ARC-AGI-Claude-3-5-Sonnet-and-GPT-4o-comparison Report

See: [ARC-AGI-Claude-3-5-Sonnet-and-GPT-4o-comparison report](https://www.notion.so/mesokosmos/ARC-AGI-Claude-3-5-Sonnet-and-GPT-4o-comparison-1d90907ca1784832a0acc315882b1dc2)

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

1. **Execute the Test Scripts**:
   - Run the following command to execute all tests. This process will take approximately 30 minutes:
     ```
     python run_tests.py
     ```

## Analyzing Results

1. **Generate Analysis Plots**:
   - After running the tests, execute the `analyze.py` script to view the results in graphical format. Ensure that any required libraries for viewing plots are installed:
     ```
     python analyze.py
     ```

## Configuration for GPT-4o

- If you want to conduct tests with GPT-4o, you need to modify the `llmClientName` parameter in the `main` function of both `run_tests.py` and `analyze.py` scripts:
  - Change `llmClientName = None` to `llmClientName = "openai"`.

You also need balance (likely a bit more than with the Claude) in the account and API KEY from OpenAI to be set in the `.env` file.

## Additional Notes

- Verify that you have all environment-specific settings configured correctly, such as Python version and any necessary environment variables.
- For detailed visualization of plots, ensure that all dependencies are properly installed and configured.
