import json
from collections import defaultdict
from llms import LlmWithCache
from utils import json_loads
from tqdm import tqdm
import pdb, os


def collect_samples(path):
    results = json.load(open(path))
    descriptions = dict()
    for result in results:
        for step in result['main_results']['step_wise_results']:
            name = step['scenario']['selected_api_name']
            description = step['scenario']['selected_description']
            if name in descriptions:
                assert(descriptions[name] == description)
            descriptions[name] = description

    print('%i tools'%len(descriptions))

    missing_descriptions = set()
    
    n_run = 0
    samples = []
    for result in results:
        if set(result['main_results']['golden_log_entries'].keys()) == set([str(step['subtask_id']) for step in result['main_results']['step_wise_results']]):
            pass
        else:
            continue
        for step in result['main_results']['step_wise_results']:
            gold = result['main_results']['golden_log_entries'][str(step['subtask_id'])]

            for i_run, run in enumerate(step['runs']):
                if 'HTTP 401' in str(run.get('parameter_quality_evaluation')):
                    continue

                _id = '%s/subtask%i/run%i'%(result['query_id'], step['subtask_id'], i_run)
                assert(gold['subtask_input'] == run['subtask_input'])
                
                n_run += 1
                if run['exact_match_accuracy'] and run['api_success']:
                    continue

                if run['selected_api'] not in descriptions:
                    missing_descriptions.add(run['selected_api'] )
                    continue
                if gold['expected_golden_api'] not in descriptions:
                    missing_descriptions.add(gold['expected_golden_api'])
                    continue

                sample = {
                    'id': _id,
                    'query': result['query_data']['query'],
                    'subtask': run['subtask_input'],
                    'predicted': {
                        'tool': {
                            'name': run['selected_api'],
                            'parameters': run['api_parameters'],
                        },
                        'output': run['subtask_output'],
                    },
                    'ground_truth': {
                        'tool': {
                            'name': gold['expected_golden_api'],
                        },
                    },
                    'api_success': run['api_success'],
                    'parameter_quality_evaluation': run['parameter_quality_evaluation'],
                }
                #pdb.set_trace()
                samples.append(sample)

    return {
        'descriptions': descriptions,
        'samples': samples,
    }
    # json.dump({
    #     'descriptions': descriptions,
    #     'samples': samples,
    # }, open(path + '.ebl_samples.json', 'w'), indent=2, ensure_ascii=False)

    # print('missing descripts %i, %i samples from %i runs'%(len(missing_descriptions), len(samples), n_run))
    # print(missing_descriptions)


def make_prompt_generate(sample, descriptions):

    tool_name_correct = sample['predicted']['tool']['name'] == sample['ground_truth']['tool']['name']
    prompt = [
        'You are an AI agent designer.',
        'The following is an example where the AI agent makes a mistake when using tools (APIs).',
    ]
    if tool_name_correct:
        prompt += [
            'The agent selected the correct tool but constructed incorrect parameters.'
        ]
    else:
        prompt += [
            'The agent selected the wrong tool for the task.'
        ]

    prompt += [
        'Your task is to propose a rule that can be added to tool description so this AI agent can avoid making similar mistakes in the future.',
        '',
        '# Query:',
        sample['query'],'',
        '## Subtask:',
        sample['subtask'],
    ]

    if tool_name_correct:
        prompt += [
            '',
            '# Tool Information',
            '',
            '## Selected Tool (Correct)',
            f"Tool Name: {sample['predicted']['tool']['name']}",
            '',
            '## Tool Description',
            '<tool_description>',
            descriptions[sample['predicted']['tool']['name']],
            '</tool_description>',
            '',
            '## Incorrect Parameters',
            json.dumps(sample['predicted']['tool']['parameters'], indent=2),
            '',
            '## Parameters Evaluation',
            json.dumps(sample['parameter_quality_evaluation'], indent=2),
        ]
    else:
        prompt += [
            '',
            '# Tool Used by AI Agent (Incorrect)',
            f"## Tool Name: {sample['predicted']['tool']['name']}",
            '## Tool Description',
            '<tool_description>',
            descriptions[sample['predicted']['tool']['name']],
            '</tool_description>',
            '',
            '# Ground-Truth Tool (Correct)',
            f"## Tool Name: {sample['ground_truth']['tool']['name']}",
            '## Tool Description',
            '<tool_description>',
            descriptions[sample['ground_truth']['tool']['name']],
            '</tool_description>',
        ]

    prompt += [
        '',
        '# Instructions',
        '1. Analyze the error above in detail.',
        '2. Explain why the AI agent made this mistake.',
    ]
    if tool_name_correct:
        names = (sample['ground_truth']['tool']['name'],)
        prompt += [
            '3. Propose a rule that can be appended to the description of tool %s so the agent avoid such mistakes in the future.'%sample['ground_truth']['tool']['name'],
            '4. Provide your rule in the following JSON format: {"rule": "..."}',
        ]
    else:
        names = (sample['ground_truth']['tool']['name'], sample['predicted']['tool']['name'])
        prompt += [
            '3. Propose rules, one for each tool, that can be appended to the description of tool %s and %s, so the agent avoid such mistakes in the future.'%names,
            '4. Provide your rule in the following JSON format: {"rule_for_%s": "...", "rule_for_%s": "..."}'%names,
        ]
    prompt += [
        '5. The length of the rule should be no more than 30 words.'
    ]

    prompt = '\n'.join(prompt)
    #pdb.set_trace()
    return prompt, names


def generate_rules(upstream, model_name='gpt-4o-2024-08-06', api_key=None):
    llm = LlmWithCache(model=model_name, api_key=api_key)
    d_rules = defaultdict(list)
    n = 0
    n_sample = 0

    data = upstream

    for sample in tqdm(data['samples']):
        n_sample += 1
        prompt, names = make_prompt_generate(sample, data['descriptions'])
        messages = [{'role': 'user', 'content': prompt}]
        resp = llm.call(messages)
        if not resp:
            continue
        parsed = json_loads(resp['content'])
        if not parsed:
            continue
        if len(names) == 1:
            rule = {
                'rule': parsed['rule'],
                'err_type': 'args',
                'sample': sample['id'],
            }
            d_rules[names[0]].append(rule)
            n += 1
        else:
            for name in names:
                rule = {
                    'rule': parsed['rule_for_%s'%name],
                    'err_type': 'select',
                    'sample': sample['id'],
                }
                d_rules[name].append(rule)
                n += 1

        # if n_sample % 10 == 0:
        #     print('%i rules from %i samples for %i tools'%(n, n_sample, len(d_rules)))
        #     json.dump({
        #         'descriptions': data['descriptions'],
        #         'rules': d_rules
        #         }, open(path + '.rules.json', 'w'), indent=2, ensure_ascii=False)
        
    print('%i rules from %i samples for %i tools'%(n, n_sample, len(d_rules)))
    return {
        'descriptions': data['descriptions'],
        'rules': d_rules
        }
    # json.dump({
    #     'descriptions': data['descriptions'],
    #     'rules': d_rules
    #     }, open(path + '.rules.json', 'w'), indent=2, ensure_ascii=False)


def make_prompt_consolidate(rules, tool_name, description):
    prompt = [
        'You are a designer of tools for AI agents.',
        f'Your task is to improve the functionality of the tool: {tool_name}, by enriching its documentation.',
        '',
        '# Current Tool Description',
        '<tool_description>',
        description,
        '</tool_description>',
        '',
        '# Rules to Incorporate',
        'Below are additional rules distilled from failure traces of AI agents using this tool, based on the current description above:',
    ] + ['- ' + rule for rule in rules] + [
        '',
        'Some of these rules may be redundant or overlapping.',
        'Please consolidate them into a concise list of non-overlapping rules that capture the most important insights — especially those not already covered in the original description.',
        '',
        '## Instructions',
        '1. Start with a brief analysis of what new constraints or insights the rules provide.',
        '2. Identify content that is missing from the original description and is important to include.',
        '3. Then, produce a final consolidated list of rules that should be *added* to the tool’s existing description to improve future use.',
        '',
        'Return your output in the following JSON format:',
        '{"consolidated_list": ["rule 1", "rule 2", "..."]}',
    ]
    return '\n'.join(prompt)


def make_prompt_fuse():
    return '\n'.join([
        'Merge the rules into the description to create a revised version, ensuring it incorporates key information from both the original description and the new rules.',
        'End your response with the following JSON format: {"revised_description": "..."}',
    ])


def consolidate_rules(upstream, fuse=False, model_name='gpt-4o-2024-08-06', api_key=None):
    data = upstream
    new_descriptions = dict()
    llm = LlmWithCache(model=model_name, api_key=api_key)
    for tool_name in tqdm(data['rules']):
        rules = list(set([x['rule'] for x in data['rules'][tool_name]]))
        description = data['descriptions'][tool_name]
        prompt = make_prompt_consolidate(rules, tool_name, description)
        messages = [{'role': 'user', 'content': prompt}]
        resp = llm.call(messages)
        if fuse:
            messages += [
                {'role': 'assistant', 'content': resp['content']},
                {'role': 'user', 'content': make_prompt_fuse()},
            ]
            resp = llm.call(messages)
            parsed = json_loads(resp['content'], parse_with_key='revised_description')
            if not parsed:
                continue
                pdb.set_trace()
            new_descriptions[tool_name] = parsed['revised_description']
        else:
            parsed = json_loads(resp['content'])
            new_descriptions[tool_name] = description + '\n\nImportant Rules:\n' + '\n'.join(['- ' + rule for rule in parsed['consolidated_list']])
    
    return new_descriptions
    # fld = path + '.consolidated'
    # if fuse:
    #     fld += '.fused'
    # os.makedirs(fld, exist_ok=True)
    # for tool_name in new_descriptions:
    #     with open(f'{fld}/{tool_name}.txt', 'w') as f:
    #         f.write(new_descriptions[tool_name])



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,)
    parser.add_argument("--path", type=str,)
    parser.add_argument("--fuse", action='store_true')
    args = parser.parse_args()

    if args.task == 'samples':
        collect_samples(args.path)
    elif args.task == 'generate':
        generate_rules(args.path)
    elif args.task == 'consolidate':
        consolidate_rules(args.path, fuse=args.fuse)