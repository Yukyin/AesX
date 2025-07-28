import json
import numpy
from datasets import load_dataset
from normal_utils import *
import re
gpqa_system_prompt = "You are a very intelligent assistant, who follows instructions directly."
LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

def load_gpqa():
    train = read_json_file('../Data/GPQA/gpqa_qa.json')

    return train

def load_aime():
    train = read_json_file('../Data/MATH/train_data.json')
    test = read_json_file('../Data/MATH/test_data.json')

    return train, test

def base_prompt(sample):
    prompt = f"What is the correct answer to this question: {sample['question']}"
    prompt += f"\n\nChoices:\n(A) {sample['choices'][0]}\n(B) {sample['choices'][1]}\n(C) {sample['choices'][2]}\n(D) {sample['choices'][3]}"
    return prompt

def base_prompt_math(sample):
    prompt = f"What is the solution to this question: {sample['problem']}"
    return prompt

def zero_shot_chain_of_thought_prompt_math(sample,system_prompt,client,model_name="gpt-4o-mini"):
    prompt = base_prompt_math(sample)
    prompt += "\nLet's think step by step: "
    completion = query_gpt(prompt, system_prompt,client,model_name)
    cot_reasoning = completion.choices[0].message.content
    prompt += f"{cot_reasoning}\n\nBased on the above, the actual answer of the question is {sample['solution']}. Please judge the correctness of the solution based on the answer. Answer in the format \"The solution is (correct/incorrect)\"."
    return prompt

def zero_shot_chain_of_thought_prompt(sample,system_prompt,client,model_name="gpt-4o-mini"):
    prompt = base_prompt(sample)
    prompt += "\nLet's think step by step: "
    completion = query_gpt(prompt, system_prompt,client,model_name)
    cot_reasoning = completion.choices[0].message.content
    prompt += f"{cot_reasoning}\n\nBased on the above, what is the single, most likely answer choice? Answer in the format \"The correct answer is (insert answer here)\"."
    return prompt

def query_gpt(prompt,system_prompt,client,model="gpt-4o-mini"):
    completion = client.chat.completions.create(model=model, messages=[ 
    {"role": "system", "content": system_prompt}, 
    {"role": "user", "content": prompt} 
    ] 
    )

    return completion


def parse_sampled_answer(answer):
    patterns = [r'answer is \((.)\)', r'Answer: \((.)\)', r'answer: \((.)\)', r'answer \((.)\)', r'\((.)\)']
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match and match.group(1) in LETTER_TO_INDEX:
            return match.group(1)
    return None





