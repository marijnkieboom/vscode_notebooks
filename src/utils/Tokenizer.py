import asyncio
import aiofiles
import nltk
from nltk.tokenize import word_tokenize

# Download the tokenizer models from nltk
nltk.download("punkt")
nltk.download("punkt_tab")

FORBIDDEN_CHARS = ['"']


async def tokenize_sentence(sentence: str) -> list:
    ''' Tokenize a single sentence '''
    return ["<SOS>"] + word_tokenize(sentence) + ["<EOS>"]


async def process_line(line: str):
    columns = line.strip().split("\t")
    eng_sentence = columns[1]
    nld_sentence = columns[3]

    if any(char in eng_sentence or char in nld_sentence for char in FORBIDDEN_CHARS):
        return None  # Skip this sentence if it contains certain characters

    eng_tokens = await tokenize_sentence(eng_sentence)
    nld_tokens = await tokenize_sentence(nld_sentence)

    return eng_tokens, nld_tokens


async def process_file(input_file: str, output_file: str):
    async with aiofiles.open(input_file, mode="r", encoding="utf-8") as tsv_file:
        reader = await tsv_file.readlines()

    async with aiofiles.open(
        output_file, mode="w", encoding="utf-8", newline=""
    ) as csv_file:
        await csv_file.write("ENG_TOKENS,NLD_TOKENS\n")  # Write headers

        # Process each line asynchronously
        tasks = []
        for line in reader:
            tasks.append(process_line(line))

        # Gather all results asynchronously
        results = await asyncio.gather(*tasks)

        for result in results:
            if result is None:
                continue

            eng_tokens, nld_tokens = result
            await csv_file.write(f'"{' '.join(eng_tokens)}","{' '.join(nld_tokens)}"\n')


def main(input_file: str, output_file: str):
    asyncio.run(process_file(input_file, output_file))


if __name__ == "__main__":
    input_file = "./data-sets/Zinparen in Engels-Nederlands - 2024-06-07.tsv"
    output_file = "./data-sets/Zinparen in Engels-Nederlands - 2024-10-21.csv"
    main(input_file, output_file)
