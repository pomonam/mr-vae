import itertools


class ConfigIterator:

    def __init__(self, conf):
        self.conf = conf

    def __iter__(self):
        return itertools.product(*[self.conf[key] for key in self.conf])


def chunks(lst, n):
    n = max(1, n)
    return (lst[i:i + n] for i in range(0, len(lst), n))


def generate_job_strings(config, command_template="python train.py "):
    jobs = []
    for setting in ConfigIterator(config):
        command = command_template
        for i, k in enumerate(config):
            command += "--{} {} ".format(k, setting[i])
        command += "\n"
        jobs.append(command)
    # Remove newline for the last line
    jobs[-1] = jobs[-1].strip()
    return jobs


def generate_sh_file(file_name, num_jobs, mem=8, qos="normal"):
    lines = []
    lines += "#!/bin/bash\n"
    lines += "#SBATCH -N 1\n"
    lines += "#SBATCH -J test\n"
    lines += "#SBATCH --gres=gpu:1\n"
    lines += "#SBATCH --mem={}GB\n".format(mem)
    lines += "#SBATCH --partition=t4v1,p100,t4v2,rtx6000\n"
    lines += "#SBATCH --qos={}\n".format(qos)
    lines += "#SBATCH --export=ALL\n"
    lines += "#SBATCH --array=0-{0}%{0}\n".format(num_jobs)
    lines += "#SBATCH --output=temp/array-%A_%a.out\n"
    lines += "#SBATCH -c 4\n"
    lines += "\n"

    lines += ". $HOME/envs/hi_env\n"
    lines += "export PYTHONPATH=$HOME/codes/hyper-influence:$PYTHONPATH\n"
    lines += "\n"

    lines += "IFS=$'\\n' read -d '' -r -a lines < {}\n".format(file_name)
    lines += "cd ..\n"
    lines += "\n"

    lines += "echo ${lines[SLURM_ARRAY_TASK_ID]} --checkpoint_dir /checkpoint/${USER}/${SLURM_JOB_ID}\n"
    lines += "eval ${lines[SLURM_ARRAY_TASK_ID]} --checkpoint_dir /checkpoint/${USER}/${SLURM_JOB_ID}"

    with open("run_" + file_name + ".sh", "w") as f:
        f.writelines(lines)
