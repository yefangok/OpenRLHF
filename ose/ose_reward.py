import json
import torch
from typing import Dict
import re
import importlib.util
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor

def extract_code(processed_str):
    regex_str = r'```(?:ObjectSense|objectSense|ose|objectsense)\n(.*?)\n```'
    ose_match = re.search(regex_str,processed_str,re.DOTALL)
    if ose_match:
        ose_code = ose_match.group(1)
    else:
        ose_code = ""
    return ose_code

def validate_response_structure(ose_code: str,solution_str:str) -> bool:
    validation_passed = True

    if re.search(r"\bself:",ose_code,flags=re.DOTALL):
        validation_passed = False
    elif re.search(r"^\s*?endclass\s*?$",ose_code,flags=re.MULTILINE | re.IGNORECASE):
        validation_passed = False

    eft_pos = solution_str.find("<|endoftext|>")
    ime_pos = solution_str.find("<|im_end|>")
    if eft_pos!=-1 and eft_pos < ime_pos:
        validation_passed = False
    return validation_passed

def run_code(ose_code):
    spec = importlib.util.spec_from_file_location("ose_validator", "/home/clouder/ose_code_model_data_preprocess/ose_validator.py")
    reward_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward_module)
    ret = reward_module.run_ose_code3(ose_code)
    return ret


def compute_score(prompt:str, solution_str: str, ose_test) :
    # Validate response structure
    ose_code = extract_code(solution_str)
    format_correct = validate_response_structure(ose_code,solution_str)
    if format_correct:
        format_score = 0.1
        ose_solution = ose_code + '\n' + ose_test
        ret = run_code(ose_solution)
        if ret["is_success"]:
            answer_score = 1.5
            grammar_score = 0.4
        else:
            answer_score = 0
            error_msg = ret["error_msg"]
            grammar_error = re.search(r": Vim.*?:E\d+:",error_msg,flags=re.DOTALL)
            test_error = re.search(r"function RunTests line \d+: Expected .*? but got",error_msg,flags=re.DOTALL)
            if test_error and not grammar_error:
                grammar_score = 0.4
            else:
                grammar_score = 0
    else:
        ret = {}
        format_score = 0
        grammar_score = 0
        answer_score = 0

    # # 0.1 + 0.4 + 1.5 - 1
    total_score = format_score + grammar_score + answer_score - 1

    # 0.1 + 1.9 - 1
    # total_score =  grammar_score + answer_score - 1

    report_str = (
        f"  Format validation: {'PASS' if format_correct else 'FAIL'}\n"
        f"  Format score: {format_score}\n"
        f"{prompt}\n"
        f"{'-'*80}\n"
        f"{solution_str}\n"
        f"{'-'*80}\n"
        f"{ose_test}\n"
        f"{'-'*80}\n"
        f"{ret}\n"
        f"{'  Final Score '.center(80, '-')}\n"
        f"  Format: {grammar_score}\n"
        f"  Answer: {answer_score}\n"
        f"  Total: {total_score}\n"
        f"{'='*80}\n"
    )
    print(report_str)
    return total_score


def reward_func(queries, prompts, labels):
    solutions = [qr[len(prompt):] for qr,prompt in zip(queries, prompts)]

    # scores = map(compute_score, prompts, solutions, labels)

    with ThreadPoolExecutor(max_workers=8) as executor:
        scores = executor.map(compute_score, prompts, solutions, labels)

    return torch.tensor(list(scores), dtype=torch.float32)