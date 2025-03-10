import nltk
import aiofiles
import asyncio
from nltk.tokenize import wordpunct_tokenize

nltk.download("punkt")

ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,.!?")

def is_invalid(sentence: str) -> bool:
    return any(char not in ALLOWED_CHARS for char in sentence)

def tokenize_sentence(sentence: str) -> list:
    return ["<SOS>"] + wordpunct_tokenize(sentence) + ["<EOS>"]

def process_line(eng_sentence: str, nld_sentence: str):
    if is_invalid(eng_sentence) or is_invalid(nld_sentence):
        return None

    return tokenize_sentence(eng_sentence), tokenize_sentence(nld_sentence)

def process_files(eng_file: str, nld_file: str, output_file: str):
    with open(eng_file, mode="r", encoding="utf-8") as eng_f, \
         open(nld_file, mode="r", encoding="utf-8") as nld_f, \
         open(output_file, mode="a", encoding="utf-8", newline="") as csv_file:

        for eng_line, nld_line in zip(eng_f, nld_f):
            eng_line = eng_line.strip()
            nld_line = nld_line.strip()

            result = process_line(eng_line, nld_line)

            if result is None:
                continue

            eng_tokens, nld_tokens = result
            csv_file.write(f'"{" ".join(eng_tokens)}","{" ".join(nld_tokens)}"\n')

def process_additional_file(additional_file: str, output_file: str):
    with open(additional_file, mode="r", encoding="utf-8") as add_f, \
         open(output_file, mode="a", encoding="utf-8", newline="") as csv_file:

        for line in add_f:
            columns = line.split("\t")

            eng_sentence = columns[1].strip()
            nld_sentence = columns[3].strip()

            result = process_line(eng_sentence, nld_sentence)

            if result is None:
                continue

            eng_tokens, nld_tokens = result
            csv_file.write(f'"{" ".join(eng_tokens)}","{" ".join(nld_tokens)}"\n')

# Run the process
async def main():
    main_folder = "/home/ocmkieboom/Downloads/en-nl.txt(1)/en-nl.txt(1)/"
    eng_file = main_folder + "CCMatrix.en-nl.en" # Path to your English file
    nld_file = main_folder + "CCMatrix.en-nl.nl" # Path to your Dutch file

    additional_file = "./data-sets/Zinparen in Engels-Nederlands - 2024-06-07.tsv"

    output_file = "./data-sets/filtered_and_tokenized_sentences(2).csv" # Path to output CSV file

    process_files(eng_file, nld_file, output_file)
    process_additional_file(additional_file, output_file)

# Entry point for running the async process
if __name__ == "__main__":
    asyncio.run(main())
