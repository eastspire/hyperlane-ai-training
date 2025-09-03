from huggingface_hub import HfApi, HfFolder
import os

# --- Configuration ---
HUGGING_FACE_REPO_ID = "hyperlane-dev/hyperlane-ai-training"
MODEL_DIR = "Qwen3-4B-Instruct-2507"


# --- Main Script ---
def upload_model_to_hub():
    """
    Uploads the contents of the MODEL_DIR to the specified Hugging Face Hub repository.
    """
    api = HfApi()
    token = HfFolder.get_token()

    if token is None:
        print(
            "Hugging Face token not found. Please log in using 'huggingface-cli login' or 'hf auth login'."
        )
        return

    # Get user info to check who is logged in
    try:
        user_info = api.whoami(token=token)
        username = user_info.get("name")
        print(f"Logged in as: {username}")
    except Exception as e:
        print(f"Error getting user info: {e}")
        print("Please ensure your token is valid.")
        return

    repo_owner = HUGGING_FACE_REPO_ID.split("/")[0]
    if username != repo_owner:
        print(
            f"Warning: You are logged in as '{username}', but the repository owner is '{repo_owner}'."
        )
        print(
            "Please make sure you have the necessary permissions to upload to this repository."
        )

    print(
        f"Preparing to upload the contents of '{MODEL_DIR}' to '{HUGGING_FACE_REPO_ID}'..."
    )

    # Create the repository on the Hub (if it doesn't exist). It will be public.
    try:
        api.create_repo(
            repo_id=HUGGING_FACE_REPO_ID, repo_type="model", exist_ok=True, token=token
        )
        print(f"Repository '{HUGGING_FACE_REPO_ID}' created or already exists.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Upload the folder
    try:
        api.upload_folder(
            folder_path=MODEL_DIR,
            repo_id=HUGGING_FACE_REPO_ID,
            repo_type="model",
            token=token,
            commit_message=f"Upload fine-tuned model and GGUF file from training session.",
        )
        print(
            f"Successfully uploaded the contents of '{MODEL_DIR}' to '{HUGGING_FACE_REPO_ID}'."
        )
    except Exception as e:
        print(f"Error uploading folder: {e}")


if __name__ == "__main__":
    upload_model_to_hub()
