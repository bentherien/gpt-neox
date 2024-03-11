# Quick and dirty script to test merges for CL. Requires having converted NeoX models to HF format.
# Adapted from https://huggingface.co/blog/mlabonne/merge-models
# Requires mergekit. Remove unnecessary keys in mergekit/mergekit/_data/architectures/gpt-neox.json if running into issues.
# Presupposes you already logged into the Huggingface hub with the cli.
import yaml
from huggingface_hub import ModelCard, ModelCardData, HfApi
from jinja2 import Template
import subprocess, os

hf_username = "adami1"
model_size = "410M"
base_models = ["pile_300B"]
# merging one at a time
models_to_merge_in = ["slimp_300B", "slimp_300B_from_pile_replay5", "german_200B", "german_200B_from_pile_replay25"]
densities = [0.05, 0.25, 0.50, 0.75, 0.95]
assert all(0 <= density <= 1 for density in densities), "Densities must be < 1."

if model_size == "410M":
    model_paths = {
        "pile_300B": "btherien/JOB-3150994_410M_it-132366_tr-pile-train_scratch",
        "slimp_300B": "btherien/JOB-3160197_410M_it-132366_tr-slim-pajama-300B_scratch",
        "slimp_300B_from_pile_replay5": "btherien/Model_-410M_It_-132366_Tr_-slim-pajama-300B-replay5_finetune",
        "german_200B": "btherien/JOB-3312386_410M_it-86245_tr-german-only_scratch",
        "german_200B_from_pile_replay25": "btherien/JOB-3312838_410M_it-86245_tr-german-replay-25_scratch",
    }
elif model_size == "10B":
    model_paths = {
        "pile_300B": "",
        "slimp_300B": "",
        "slimp_300B_from_pile_replay5": "",
    }
else:
    raise ValueError(f"{model_size} must be 410M or 10B")

# Generate configs automatically
for base_model in base_models:
    for model_to_merge_in in models_to_merge_in:
        base_model_path = model_paths[base_model]
        model_to_merge_in_path = model_paths[model_to_merge_in]
        for density in densities:
            yaml_config = f"""models:
  - model: {base_model_path}
    # no parameters necessary for base model
  - model: {model_to_merge_in_path}
    parameters:
      density: {density}
      weight: 1.0
merge_method: ties
base_model: {base_model_path}
parameters:
  normalize: true
dtype: float16"""
            MODEL_NAME = model_size + "_TIES-merge_" + model_to_merge_in + "_into_" + base_model + "_density-" + str(density)
            config_path = MODEL_NAME + '/mergekit_config.yml'

            if not os.path.exists(MODEL_NAME):
                os.makedirs(MODEL_NAME)
            with open(config_path, 'w', encoding="utf-8") as f:
                f.write(yaml_config)

            mergekit_command = f"mergekit-yaml {config_path} {MODEL_NAME} --copy-tokenizer --lazy-unpickle"
            print("Running command: ", mergekit_command)
            result = subprocess.run(mergekit_command, shell=True, capture_output=True, text=True)
            print(f"Done merging. Preparing to upload model {MODEL_NAME}...")

            # model card template
            template_text = r"""---
License: apache-2.0
tags:
- merge
- mergekit
- lazymergekit
{%- for model in models %}
- {{ model }}
{%- endfor %}
---

# {{ model_name }}

{{ model_name }} is a merge of the following models using [mergekit](https://github.com/cg123/mergekit):

{%- for model in models %}
* [{{ model }}](https://huggingface.co/{{ model }})
{%- endfor %}

## 🧩 Configuration

\```yaml
{{- yaml_config -}}
\```"""
            jinja_template = Template(template_text.strip())

            # Get list of models from config
            data = yaml.safe_load(yaml_config)
            if "models" in data:
                models = [data["models"][i]["model"] for i in range(len(data["models"])) if "parameters" in data["models"][i]]
            elif "parameters" in data:
                models = [data["slices"][0]["sources"][i]["model"] for i in range(len(data["slices"][0]["sources"]))]
            elif "slices" in data:
                models = [data["slices"][i]["sources"][0]["model"] for i in range(len(data["slices"]))]
            else:
                raise Exception("No models or slices found in yaml config")

            # Fill the template
            content = jinja_template.render(
                model_name=MODEL_NAME,
                models=models,
                yaml_config=yaml_config,
                username=hf_username,
            )

            # Save the model card
            card = ModelCard(content)
            card.save(MODEL_NAME + '/README.md')

            # Defined in the secrets tab in Google Colab
            api = HfApi()#token=userdata.get("HF_TOKEN"))

            try:
                api.create_repo(
                    repo_id=f"{hf_username}/{MODEL_NAME}",
                    repo_type="model"
                )
            except:
                print("WARNING: Could not create the repo. This is a problem unless the repo already exists.")

            api.upload_folder(
                repo_id=f"{hf_username}/{MODEL_NAME}",
                folder_path=MODEL_NAME,
            )
