from pathlib import Path
from urllib.request import urlretrieve
import hashlib

URL = (
    "https://huggingface.co/datasets/bitext/"
    "Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)

OUTPUT_PATH = Path("data/raw/bitext_customer_support.csv")

EXPECTED_SHA256 = (
    "6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434"
)


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset...")
    urlretrieve(URL, OUTPUT_PATH)

    file_hash = calculate_sha256(OUTPUT_PATH)

    if file_hash != EXPECTED_SHA256:
        OUTPUT_PATH.unlink(missing_ok=True)
        raise RuntimeError("Dataset checksum verification failed.")

    print(f"Dataset saved to: {OUTPUT_PATH}")
    print("SHA256 verified successfully.")


if __name__ == "__main__":
    main()