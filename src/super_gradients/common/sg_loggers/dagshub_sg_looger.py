import os
from pathlib import Path
from typing import Union, Optional, Any

import torch

from super_gradients.common.registry.registry import register_sg_logger
from super_gradients.common.abstractions.abstract_logger import get_logger

from super_gradients.common.sg_loggers.base_sg_logger import BaseSGLogger
from super_gradients.common.environment.ddp_utils import multi_process_safe

logger = get_logger(__name__)

try:
    import dagshub
    from dagshub.upload import Repo
except (ModuleNotFoundError, ImportError, NameError) as dagshub_import_err:
    _import_dagshub_error = dagshub_import_err

try:
    import mlflow
except (ModuleNotFoundError, ImportError, NameError) as mlflow_import_err:
    _import_mlflow_error = mlflow_import_err

@register_sg_logger("dagshub_sg_logger")
class DagsHubSGLogger(BaseSGLogger):
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        storage_location: str,
        resumed: bool,
        training_params: dict,
        checkpoints_dir_path: str,
        tb_files_user_prompt: bool = False,
        launch_tensorboard: bool = False,
        tensorboard_port: int = None,
        save_checkpoints_remote: bool = True,
        save_tensorboard_remote: bool = True,
        save_logs_remote: bool = True,
        monitor_system: bool = None,
    ):
        """

        :param experiment_name:         Name used for logging and loading purposes
        :param storage_location:        If set to 's3' (i.e. s3://my-bucket) saves the Checkpoints in AWS S3 otherwise saves the Checkpoints Locally
        :param resumed:                 If true, then old tensorboard files will **NOT** be deleted when tb_files_user_prompt=True
        :param training_params:         training_params for the experiment.
        :param checkpoints_dir_path:    Local root directory path where all experiment logging directories will reside.
        :param tb_files_user_prompt:    Asks user for Tensorboard deletion prompt.
        :param launch_tensorboard:      Whether to launch a TensorBoard process.
        :param tensorboard_port:        Specific port number for the tensorboard to use when launched (when set to None, some free port number will be used
        :param save_checkpoints_remote: Saves checkpoints in s3.
        :param save_tensorboard_remote: Saves tensorboard in s3.
        :param save_logs_remote:        Saves log files in s3.
        :param monitor_system:          Save the system statistics (GPU utilization, CPU, ...) in the tensorboard
        """
        if monitor_system is not None:
            logger.warning("monitor_system not available on DagsHubSGLogger. To remove this warning, please don't set monitor_system in your logger parameters")

        self.s3_location_available = storage_location.startswith("s3")
        super().__init__(
            project_name=project_name,
            experiment_name=experiment_name,
            storage_location=storage_location,
            resumed=resumed,
            training_params=training_params,
            checkpoints_dir_path=checkpoints_dir_path,
            tb_files_user_prompt=tb_files_user_prompt,
            launch_tensorboard=launch_tensorboard,
            tensorboard_port=tensorboard_port,
            save_checkpoints_remote=self.s3_location_available,
            save_tensorboard_remote=self.s3_location_available,
            save_logs_remote=self.s3_location_available,
            monitor_system=False,
        )
        if _import_dagshub_error:
            raise _import_dagshub_error
        
        if _import_mlflow_error:
            raise _import_mlflow_error
        
        self._init_env_dependency()
        
    def _init_env_dependency(self):
        """
        Look for .wandbinclude file in parent dirs and return the list of paths defined in the file.

        file structure is a single relative (i.e. src/) or a single type (i.e *.py)in each line.
        the paths and types in the file are the paths and types to be included in code upload to wandb
        :return: if file exists, return the list of paths and a list of types defined in the file
        """

        self.paths = {
            "dvc_directory": Path("artifacts"),
            "models": Path("models"),
            "artifacts": Path("artifacts"),
        }

        self.repo_name, self.repo_owner = None, None

        token = dagshub.auth.get_token()
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        # Check mlflow environment variable is set:
        if not self.repo_name or not self.repo_owner:
            self.repo_name, self.repo_owner = self.splitter(
                input("Please insert your repository owner_name/repo_name:")
            )

        if not self.remote or "dagshub" not in os.getenv("MLFLOW_TRACKING_URI"):
            dagshub.init(repo_name=self.repo_name, repo_owner=self.repo_owner)
            self.remote = os.getenv("MLFLOW_TRACKING_URI")

        self.repo = Repo(
            owner=self.remote.split(os.sep)[-2],
            name=self.remote.split(os.sep)[-1].replace(".mlflow", ""),
            branch=os.getenv("BRANCH", "main"),
        )
        self.dvc_folder = self.repo.directory(str(self.paths["dvc_directory"]))

        mlflow.set_tracking_uri(self.remote)
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(nested=True)
        return self.run
    
    @multi_process_safe
    def _dvc_add(self, local_path="", remote_path=""):
        if not os.path.isfile(local_path):
            FileExistsError(f"Invalid file path: {local_path}")
        self.dvc_folder.add(file=local_path, path=remote_path)

    @multi_process_safe
    def _dvc_commit(self, commit=""):
        self.dvc_folder.commit(commit, versioning="dvc", force=True)

    @multi_process_safe
    def add_config(self, tag: str, config: dict):
        super(DagsHubSGLogger, self).add_config(tag=tag, config=config)
        mlflow.log_params(config)

    @multi_process_safe
    def add_scalar(self, tag: str, scalar_value: float, global_step: int = 0):
        super(DagsHubSGLogger, self).add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)
        mlflow.log_metric(key=tag, value=scalar_value, step=global_step)

    @multi_process_safe
    def add_scalars(self, tag_scalar_dict: dict, global_step: int = 0):
        super(DagsHubSGLogger, self).add_scalars(tag_scalar_dict=tag_scalar_dict, global_step=global_step)
        mlflow.log_metric(metrics=tag_scalar_dict, step=global_step)

    @multi_process_safe
    def add_text(self, tag: str, text_string: str, global_step: int = 0):
        super().add_text(tag, text_string, global_step)
        mlflow.log_metric(key=tag, value=text_string, step=global_step)

    @multi_process_safe
    def close(self):
        super().close()
        try:
            mlflow.end_run()
        except Exception:
            pass

    @multi_process_safe
    def add_file(self, file_name: str = None):
        super().add_file(file_name)
        self._dvc_add(local_path=file_name, remote_path=os.path.join(self.paths["artifacts"], file_name))
        self._dvc_commit(commit="add file")

    @multi_process_safe
    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = 0):
        name = f"ckpt_{global_step}.pth" if tag is None else tag
        if not name.endswith(".pth"):
            name += ".pth"
        path = os.path.join(self._local_dir, name)
        torch.save(state_dict, path)
        self._dvc_add(local_path=path, remote_path=os.path.join(self.paths["models"], path))
        self._dvc_commit(commit=f"log {global_step} checkpoint model")