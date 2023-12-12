import os
import random
import numpy as np
import torch
import pprint
import subprocess
import sys
from IPython.display import display, HTML, Image

pp = pprint.PrettyPrinter(indent=4)

def seeding(seed):
    '''Function to set seed value.'''
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    '''
    # https://stackoverflow.com/a/58965640
    If your model does not change and your input sizes remain the same - then you may benefit from setting 
    torch.backends.cudnn.benchmark = True.
    However, if your model changes: for instance, if you have layers that are only "activated" when certain conditions are met, 
    or you have layers inside a loop that can be iterated a different number of times, 
    then setting torch.backends.cudnn.benchmark = True might stall your execution.
    '''
    torch.backends.cudnn.benchmark = True 
    # torch.backends.cudnn.benchmark = False


def pprint_objects(*arg):
    '''Prints large and nested objects clearly.'''
    pp.pprint(arg)

def create_directory_if_not_exists(path):
    '''Creates a directory or multiple nested directories if they don't exist.'''
    if not os.path.exists(path):
        os.makedirs(path)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins, elapsed_secs = divmod(elapsed_time, 60)
    return elapsed_mins, elapsed_secs

def excute_cmd(command, log=False):
    '''
    Execute a command and check for success.

    Args:
        command ('str'): Command to execute.
        log ('bool'): Boolean for printing the result
    
    Returns:
        result ('str'): Output of the command if successful.
    '''
    # excute the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

    # Check the return code to see if the command was successful
    if result.returncode == 0:
        if log: 
            print(result.stdout)
            return None

        return result.stdout
    else:
        print(f"Command failed with an error: {command}")
        print(result.stderr)
        return result.stderr
    
def execute_cmd_realtime(command, log=True):
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            if log:
                sys.stdout.write(output)
                sys.stdout.flush()
            else:
                display(HTML(output.strip()))

    rc = process.poll()
    return rc


def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
