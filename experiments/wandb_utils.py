"""Thin wrappers around `wandb` for training scripts.

The training scripts in this repo log metrics with Weights & Biases. To use
your own account, set the `WANDB_API_KEY` environment variable before running.
To run without uploading anything to W&B (e.g. for a quick local test), set
`WANDB_MODE=disabled`.
"""
import os
import pathlib
from typing import Optional

import wandb


def init_api() -> wandb.Api:
  return wandb.Api()


def init_wandb(checkpoint_dir: Optional[str],
               project_name: str,
               run_name: Optional[str] = None,
               config: Optional[dict] = None,
               return_id: bool = False) -> Optional[str]:
  """Initialize a W&B run, resuming from a saved run id if one exists.

  If `WANDB_MODE` is set to "disabled", wandb calls become no-ops and no
  API key is required.
  """
  if checkpoint_dir is None:
    run = wandb.init(project=project_name, name=run_name, config=config)
    project_id = str(run.id) if run is not None else ""
  else:
    wandb_id_file_path = pathlib.Path(
        os.path.join(checkpoint_dir, "_wandb_runid.txt"))
    if wandb_id_file_path.exists():
      resume_id = wandb_id_file_path.read_text()
      wandb.init(
          project=project_name, name=run_name, resume=resume_id, config=config)
      project_id = resume_id
    else:
      run = wandb.init(project=project_name, name=run_name, config=config)
      project_id = str(run.id) if run is not None else ""
      if project_id:
        wandb_id_file_path.write_text(project_id)

  wandb_config = wandb.config
  if config is not None:
    config.update(wandb_config)

  if return_id:
    return project_id
  return None
