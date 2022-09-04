import pathlib
import os
import wandb


def init_api() -> wandb.Api:
  api = wandb.Api()
  return api


def init_wandb(checkpoint_dir: str,
               project_name: str,
               run_name: str = None,
               config: dict = None) -> None:
  # Put your wandb API key here
  os.environ["WANDB_API_KEY"] = "56048bb80728a171e201df652b4687a74f23c3e7"

  if checkpoint_dir is None:
    wandb.init(project=project_name, name=run_name, config=config)
  else:
    # If the run_id was previously saved, resume from there
    wandb_id_file_path = pathlib.Path(
        os.path.join(checkpoint_dir, "_wandb_runid.txt"))
    if wandb_id_file_path.exists():
      resume_id = wandb_id_file_path.read_text()
      wandb.init(
          project=project_name,
          name=run_name,
          resume=resume_id,
          config=config)
    else:
      # If the run_id doesn't exist, then create a new run and write the id the file
      run = wandb.init(
          project=project_name, name=run_name, config=config)
      wandb_id_file_path.write_text(str(run.id))

  wandb_config = wandb.config
  if config is not None:
    config.update(wandb_config)

  return config
