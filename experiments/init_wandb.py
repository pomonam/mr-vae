import os
import pathlib

from dotenv import find_dotenv
from dotenv import load_dotenv
import wandb

log_dir = pathlib.Path(__file__).parents[1] / "logs"
load_dotenv(find_dotenv())


def init_api():
    api = wandb.Api()
    return api


def init_wandb(checkpoint_dir, project_name, run_name=None, config=None):
    if checkpoint_dir is None:
        wandb.init(project=project_name,
                   name=run_name,
                   config=config,
                   dir=log_dir)
    else:
        # If the run_id was previously saved, resume from there
        wandb_id_file_path = pathlib.Path(
            os.path.join(checkpoint_dir, "_wandb_runid.txt"))
        if wandb_id_file_path.exists():
            resume_id = wandb_id_file_path.read_text()
            wandb.init(project=project_name,
                       name=run_name,
                       resume=resume_id,
                       config=config,
                       dir=log_dir)
        else:
            # If the run_id doesn't exist, then create a new run and write the run id the file
            run = wandb.init(project=project_name,
                             name=run_name,
                             config=config,
                             dir=log_dir)
            wandb_id_file_path.write_text(str(run.id))

    wandb_config = wandb.config
    if config is not None:
        # update the current passed in config with the wandb_config
        config.update(wandb_config)

    return config
