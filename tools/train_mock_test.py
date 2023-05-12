import argparse
import time
from clearml import Task
import os
import time
from clearml.task import TaskInstance
from torch.distributed import run
import sys

def get_args():
    parser = argparse.ArgumentParser('Test')
    parser.add_argument('gpus', type=int, help='number of gpus to use')
    parser.add_argument('--remote', action="store_true", help='run remotely on clearml-agent')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    task:TaskInstance = Task.init(project_name="Debug", task_name="Test")
    task.set_base_docker(docker_image="pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel")

    args = get_args()

    if args.remote:
        task.execute_remotely(queue_name="default")

    # torch.distributed only sets MASTER_ADDR when `torch.distributed.run` is called
    if os.environ.get("MASTER_ADDR") is None:
        # This block runs once and spawns two worker threads.
        # Here we can, for instance, create the clearml task and do any
        # setup we need to

        os.environ["OMP_NUM_THREADS"] = "1"
        
        # Use the same argparser as torch.distributed.run for conveniency
        distributed_args = run.parse_args(sys.argv)
        distributed_args.nproc_per_node = args.gpus
        run.run(distributed_args)
    else:
        # This will only run on the worker subprocesses.
        # Each worker runs the main function separately
        # and then the main worker syncs their results
        print(f"=== STARTED WORKER {os.environ.get('RANK')}")
        print(f"{args} = {args}")
        # Gets the task if running through clearML
        time.sleep(2)
        print(f"=== FINISHED WORKER {os.environ.get('RANK')}")
