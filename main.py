from ebl import collect_samples, generate_rules, consolidate_rules
from pathlib import Path
from termcolor import cprint
import json
from collections import defaultdict
from tqdm import tqdm
from typing import Tuple
import yaml
import random

def process_one_query(folder_path, model_name='gpt-4o-2024-08-06', api_key=None):
    path = Path(folder_path)/"step_wise_eval_results.json"
    data = collect_samples(path)
    data = generate_rules(data, model_name=model_name, api_key=api_key)
    data = consolidate_rules(data, fuse=True, model_name=model_name, api_key=api_key)
    return data

def substitute_descriptions(ebl_data, yaml_data):
    ebl_data = ebl_data.copy()
    mcp_servers = yaml_data["mcp_servers"]
    mcp_servers_keys = list(mcp_servers.keys())
    mcp_server1 = mcp_servers[mcp_servers_keys[0]]
    tools = mcp_server1["tools"]
    for tool in tools:
        tool_name = tool["tool_name"]
        if tool_name in ebl_data:
            tool["description"] = ebl_data[tool_name]
            ebl_data.pop(tool_name)
    return ebl_data, yaml_data


def load_mcp_yaml(mcp_tool_path):
    path = Path(mcp_tool_path)
    # load the yaml file into a dictionary "category" --> "tool_name" --> yaml
    cate_dict = defaultdict[Tuple[str, str], dict](dict)
    for category_folder in (progress_bar := tqdm(list(path.iterdir()), desc="Loading MCP YAML files")):
        # tqdm with category as progress bar
        for file in category_folder.iterdir():
            # check if the file is a yaml file
            if not str(file).endswith(".yaml"):
                continue
            with open(file, "r") as f:
                yaml_data = yaml.safe_load(f)
            tool_name = list(yaml_data["mcp_servers"].keys())[0]
            cate_name = yaml_data["mcp_servers"][tool_name]["category"]
            cate_dict[(cate_name, tool_name)] = file
    return cate_dict

def list_eval_results_folders(eval_results_dir):
    path = Path(eval_results_dir)
    eval_results_list = []
    for folder in path.iterdir():
        # if is dir, then it is a api provider
        if folder.is_dir():
            eval_results_list.append(folder)
    return eval_results_list

def is_valid_eval_result(eval_result_path):
    # check if file "evaluation_statistics.json" exists
    if not (Path(eval_result_path) / "evaluation_statistics.json").exists():
        return False
    # if summary->total_api_calls is 0, then it is not a valid eval result
    with open(Path(eval_result_path) / "evaluation_statistics.json", "r") as f:
        eval_results = json.load(f)
    if eval_results["summary"]["total_api_calls"] == 0:
        return False
    return True

def find_corresponding_yaml_file(eval_result_path):
    # read "run_parameters.json"
    with open(Path(eval_result_path) / "run_parameters.json", "r") as f:
        run_parameters = json.load(f)
    # use the dataset to find the corresponding query file
    queries_path = Path(run_parameters["dataset"])
    # read the query file
    with open(queries_path, "r") as f:
        queries = json.load(f)
    # find the corresponding query
    query1 = queries[0]
    api1 = query1["api_list"][0]
    category_name = api1["category_name"]
    tool_name = api1["tool_name"]
    return (category_name, tool_name)
    

def process_dir(yaml_folder, output_folder, eval_results_dir, model_name='gpt-4o-2024-08-06', api_key=None):
    # 1. read all yaml files
    # 2. read the eval results folder
    # 2.1 for each query, find the corresponding yaml file
    # 2.1.1 use the (category_name, tool_name) to find the corresponding yaml file
    # 2.2 if the yaml file is already in the output folder, skip it
    # 2.3 if the yaml file is not in the output folder, run the D2 pipeline


    # read yaml
    all_mcp_yaml = load_mcp_yaml(yaml_folder)
    # read eval results folders
    eval_results_folders = list_eval_results_folders(eval_results_dir)
    random.shuffle(eval_results_folders)
    for eval_results_folder in eval_results_folders:
        if is_valid_eval_result(eval_results_folder):
            (category_name, tool_name) = find_corresponding_yaml_file(eval_results_folder)
            if (category_name, tool_name) in all_mcp_yaml:
                yaml_file = all_mcp_yaml[(category_name, tool_name)]
                output_file = output_folder / category_name / yaml_file.name
                if output_file.exists():
                    cprint(f"Skipping {yaml_file.name} because it already exists.", 'yellow')
                    continue
                ebl_data = process_one_query(eval_results_folder, model_name=model_name, api_key=api_key)
                with open(yaml_file, "r") as f:
                    yaml_data = yaml.safe_load(f)
                ebl_data_remaining, yaml_data = substitute_descriptions(ebl_data, yaml_data)
                if len(ebl_data_remaining) > 0:
                    cprint(f"Some tools are not found: {ebl_data_remaining.keys()}", 'yellow')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w") as f:
                    yaml.dump(yaml_data, f, 
                     default_flow_style=False, 
                     indent=2, 
                     sort_keys=False,
                     allow_unicode=True)

                cprint(f"Processed {output_file}.", 'green')



if __name__ == '__main__':
    """
    python main.py --yaml_folder /home/sagemaker-user/FunctionWrapper/description_improvement/results/StableToolBench_D1/ --eval_results_dir /home/sagemaker-user/FunctionWrapper/experiments/20251113_051305/ --output_dir /home/sagemaker-user/FunctionWrapper/description_improvement/results/StableToolBench_D1_ebl/
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_folder", type=str)
    parser.add_argument("--eval_results_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_name", type=str, default="openai:gpt-4.1-2025-04-14",
                       help="Model name for LLM, e.g. 'openai:gpt-4.1-2025-04-14' or 'vllm:Qwen/Qwen2.5-7B-Instruct' (overrides config)")
    parser.add_argument("--openai_api_key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    args = parser.parse_args()

    yaml_folder = Path(args.yaml_folder)
    eval_results_dir = Path(args.eval_results_dir)
    output_dir = Path(args.output_dir)

    model_name = args.model_name.split(":", 1)[1] if ":" in args.model_name else args.model_name

    process_dir(yaml_folder, output_dir, eval_results_dir, model_name=model_name, api_key=args.openai_api_key)