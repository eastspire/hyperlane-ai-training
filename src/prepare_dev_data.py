import os
import json

# --- Configuration ---
OUTPUT_FILE = "./training_data.jsonl"

# A small set of dummy QA pairs to ensure the training pipeline can run quickly.
# This avoids the slow process of loading a large model to generate data.
DUMMY_QA_PAIRS = [
    {
        "question": "What is Hyperlane?",
        "answer": "Hyperlane is a permissionless interoperability layer that allows you to connect dApps across different blockchain networks.",
    },
    {
        "question": "How does Hyperlane work?",
        "answer": "Hyperlane uses a system of smart contracts and a decentralized network of validators to securely pass messages between chains.",
    },
    {
        "question": "What is permissionless interoperability?",
        "answer": "It means that any developer can deploy Hyperlane to any chain without needing to ask for permission, enabling seamless cross-chain communication.",
    },
    {
        "question": "What are the core components of Hyperlane?",
        "answer": "The core components include the Mailbox smart contract for sending/receiving messages, and a network of validators and relayers for security and message transport.",
    },
]


def create_fast_dev_data():
    """
    Generates a small, static training file for development purposes.
    This is much faster than generating data with a real model.
    """
    print("---" + " Starting fast dev data preparation ---")
    print(f"This will create a dummy '{OUTPUT_FILE}' to speed up the dev pipeline.")

    # The format required by the trainer
    training_prompt_format = "Question: {question}\nAnswer: {answer}"

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for pair in DUMMY_QA_PAIRS:
                # Create the text field content
                text_content = training_prompt_format.format(
                    question=pair["question"], answer=pair["answer"]
                )
                # Create the JSON object for the line
                json_line = {"text": text_content}
                # Write the JSON object to the file, followed by a newline
                f.write(json.dumps(json_line) + "\n")

        print(
            f"\n\033[92mSuccessfully created dummy training data at: {os.path.abspath(OUTPUT_FILE)}\033[0m"
        )
        print("The next step (train_dev) will use this small file.")

    except IOError as e:
        print(f"--- ERROR: Could not write to file {OUTPUT_FILE} ---")
        print(e)
        # Exit with an error code if we can't create the data
        import sys

        sys.exit(1)


if __name__ == "__main__":
    create_fast_dev_data()
