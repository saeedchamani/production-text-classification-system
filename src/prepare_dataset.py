import re

import pandas as pd


INPUT_PATH = "data/raw/bitext_customer_support.csv"
OUTPUT_PATH = "data/processed/cleaned_support_messages.csv"


def clean_text(text: str) -> str:
    text = text.strip()
    text = text.lower()

    # Replace placeholders such as {{Order Number}}
    text = re.sub(r"\{\{.*?\}\}", " placeholder ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df = pd.read_csv(INPUT_PATH)

print("Original shape:", df.shape)

# Keep only useful columns
df = df[["instruction", "category", "intent"]].copy()

# Rename input text column
df = df.rename(columns={"instruction": "text"})

# Remove missing values
df = df.dropna(subset=["text", "category", "intent"])

# Clean text
df["text"] = df["text"].astype(str).apply(clean_text)

# Remove empty texts
df = df[df["text"].str.len() > 0]

# Remove exact duplicates
df = df.drop_duplicates()

# Find texts that have conflicting intent labels
conflicting_texts = (
    df.groupby("text")["intent"]
    .nunique()
)

conflicting_texts = conflicting_texts[
    conflicting_texts > 1
].index

print("Conflicting texts:", len(conflicting_texts))

# Remove ambiguous duplicated texts
df = df[~df["text"].isin(conflicting_texts)]

# Reset indexes
df = df.reset_index(drop=True)

# Save cleaned dataset
df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8",
)

print("Cleaned shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nIntent distribution:")
print(df["intent"].value_counts())

print("\nSample:")
print(df.head())