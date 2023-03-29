import json
import os

import pandas as pd

if __name__ == "__main__":
    root_path = "./diffusion"

    with open("./diffusion/train/metadata.jsonl") as f:
        train_data = {
            "filepath": [],
            "title": [],
        }
        for line in f:
            item = json.loads(line)
            train_data["filepath"].append(
                os.path.join("./diffusion/train/", item["file_name"])
            )
            train_data["title"].append(item["text"])

        train_df = pd.DataFrame.from_dict(train_data)

    with open("./diffusion/validation/metadata.jsonl") as f:
        validation_data = {
            "filepath": [],
            "title": [],
        }
        for line in f:
            item = json.loads(line)
            validation_data["filepath"].append(
                os.path.join("./diffusion/validation/", item["file_name"])
            )
            validation_data["title"].append(item["text"])

        valid_df = pd.DataFrame.from_dict(validation_data)

    pd.DataFrame.from_dict(train_df).to_csv(
        os.path.join(root_path, "train_coca.csv"), index=False, sep="\t"
    )

    pd.DataFrame.from_dict(valid_df).to_csv(
        os.path.join(root_path, "valid_coca.csv"), index=False, sep="\t"
    )
