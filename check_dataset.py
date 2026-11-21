import json


files = [
    "dataset/train.jsonl",
    "dataset/validation.jsonl",
    "dataset/test.jsonl"
]


for filename in files:

    print(f"\nChecking {filename}...")

    count = 0

    try:
        with open(filename, "r", encoding="utf-8") as file:

            for line_number, line in enumerate(file, start=1):

                if not line.strip():
                    continue

                json.loads(line)

                count += 1

        print(f"Valid JSONL")
        print(f"Examples: {count}")

    except json.JSONDecodeError as e:

        print(f"Invalid JSON on line {line_number}")
        print(e)

    except FileNotFoundError:

        print("File not found.")