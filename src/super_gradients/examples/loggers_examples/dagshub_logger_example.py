from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, models
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5
from super_gradients.training.dataloaders.dataloaders import cifar10_train, cifar10_val

experiment_name = "classification_transfer_learning"
CHECKPOINT_DIR = '/'
dagshub_repository = input("Enter your DagsHub repository path (format <repo-owner>/<repo-name>) :")

trainer = Trainer(experiment_name=experiment_name, ckpt_root_dir=CHECKPOINT_DIR)
model = models.get(Models.RESNET18, num_classes=10)

training_params = {
    "max_epochs": 11,
    "lr_updates": [5, 10, 15],
    "lr_decay_factor": 0.1,
    "lr_mode": "step",
    "initial_lr": 0.1,
    "loss": "cross_entropy",
    "optimizer": "SGD",
    "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
    "train_metrics_list": [Accuracy(), Top5()],
    "valid_metrics_list": [Accuracy(), Top5()],
    "metric_to_watch": "Accuracy",
    "greater_metric_to_watch_is_better": True,
    "sg_logger": "dagshub_sg_logger",
    "sg_logger_params": {"project_name": "auto_pipe_test",
                                       "save_checkpoints_remote": True,
                                       "save_tensorboard_remote": False,
                                       "log_mlflow_only":True,
                                       "save_logs_remote": True,
                                       "dagshub_repository":dagshub_repository
                         },
}

trainer.train(model=model, training_params=training_params, train_loader=cifar10_train(), valid_loader=cifar10_val())
