import json
import re
from tqdm import tqdm

def format_data(system, prompt, answer):
    return {"messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}]
                }
            ]
        }

def convert_camel_case_to_snake_case(word):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', word).lower()

def format_json_string(json_str):
    return f"```json\n{json_str}\n```"

def prepare_data_for_llama(df):

    formatted_dataset = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        system = row["instruction"]
        prompt = row["input"]
        answer = row["output"]
        # data_list = json.loads(data_str)
        # data_list = [{k: v for d in data_list for k, v in d.items()}]
        # answer = format_json_string(data_list[0])
        formatted_dataset.append(format_data(system, prompt, answer))

    return formatted_dataset
