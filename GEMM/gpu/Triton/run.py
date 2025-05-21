# %%
import dataclasses
import multiprocessing
import re
import time
import os
import sys
import math
from pathlib import Path
from typing import Any, Optional
import yaml

import torch.cuda

from utils import set_seed
try:
    from task import TestSpec
except ImportError:
    TestSpec = dict

from reference import check_implementation, generate_input




@dataclasses.dataclass
class TestCase:
    args: dict
    spec: str
    memory_usage: Optional[float] = None
    FLOPs: Optional[int] = None

def _combine(a: int, b: int) -> int:
    # combine two integers into one:
    # we need this to generate a secret seed based on the test-level seed and
    # the global secret seed.
    # the test-level seeds are public knowledge, and typically relatively small numbers,
    # so we need to make sure they don't provide any useful info for the full seed.
    # This Cantor construction ensures that if the secret seed is a large number,
    # then so is the overall seed.
    return int(a + (a+b)*(a+b+1)//2)


def get_test_cases(file_name: str, data_type: str, seed: Optional[int]) -> list[TestCase]:
    try:
        with open(file_name, 'r') as file:
            data = yaml.safe_load(file)
    except Exception as E:
        print(f"Could not open test file`{file_name}`: {E}", file=sys.stderr)
        exit(113)

    tests_data = data.get(data_type, None)
    if tests_data is None:
        print(f"Could not find test data for type `{data_type}` in file `{file_name}`", file=sys.stderr)
        exit(113)

    tests = []
    for data in tests_data:
        memory_usage = (data['m'] * data['n'] + data['k'] * data['n'] + data['k'] * data['m']) * 2 / 1024 / 1024
        FLOPs = (data['m'] * data['n'] * data['k']) * 2
        tests.append(TestCase(spec=str(data), args=data, memory_usage=memory_usage, FLOPs=FLOPs))

    if seed is not None:
        for test in tests:
            if "seed" in test.args:
                test.args["seed"] = _combine(test.args["seed"], seed)

    return tests



@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float

def calculate_stats(durations: list[int]):
    """
    Calculate statistical data from a list of durations.

    @param durations: A list of durations in nanoseconds.
    @return: A Stats object containing the number of runs, mean, standard deviation, error, best, and worst durations.
    """
    runs = len(durations)
    total = sum(durations)
    best = min(durations)
    worst = max(durations)

    avg = total / runs
    variance = sum(map(lambda x: (x - avg)**2, durations))
    std = math.sqrt(variance / (runs - 1))
    err = std / math.sqrt(runs)

    return Stats(runs=runs, mean=avg, std=std, err=err, best=float(best),
                 worst=float(worst))

def _clone_data(data):
    """
    Recursively goes through data and clones all tensors.
    """
    if isinstance(data, tuple):
        return tuple(_clone_data(x) for x in data)
    elif isinstance(data, list):
        return [_clone_data(x) for x in data]
    elif isinstance(data, dict):
        return {k: _clone_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.clone()
    else:
        return data
        
def wrap_check_implementation(data, submission_output):
    # Old version returned just a single string, new version
    # returns (bool, str); this function ensures compatibility with old
    # problem definitions.
    result = check_implementation(data, submission_output)
    if isinstance(result, tuple):
        return result
    else:
        return not bool(result), result


# %%
def _run_single_test(test: TestCase):
    """
    Runs a single test case. Do not call directly
    """
    from triton_v0 import custom_kernel

    data = generate_input(**test.args)
    start_time = time.time()
    torch.cuda.synchronize()
    submission_output =  custom_kernel(_clone_data(data))
    torch.cuda.synchronize()
    end_time = time.time()
    duration = float((end_time - start_time) * 1e3)  # convert to nanoseconds
    good, message = wrap_check_implementation(data, submission_output)
    return good, message, duration

def run_single_test(pool: multiprocessing.Pool, test: TestCase):
    """
    Runs a single test in another process.
    """
    return pool.apply(_run_single_test, (test,))
    


# %%

seed = 42
set_seed(seed or 42)
tests_data = get_test_cases('./task.yml', 'tests', seed)

# import multiprocessing
# mp_context = multiprocessing.get_context('spawn')
# with mp_context.Pool(1) as pool:
#     for idx, test in enumerate(tests_data):
#         good, message = run_single_test(pool, test)
# import time


for idx, test in enumerate(tests_data):
    print(f"test.{idx}.name", test.spec)
    good, message, duration = _run_single_test(test)
    if not good:
        print(f"test.{idx}.status", "fail")
        print(f"test.{idx}.error", message)
        passed = False
    else:
        print(f"test.{idx}.status", "pass")
        print(f"test.{idx}.duration {duration:.4f}ms")
        print(f"test.{idx}.TFLOPS {test.FLOPs/duration*1e-9:.4f}")
        if message:
            print(f"test.{idx}.message", f"{message}")



