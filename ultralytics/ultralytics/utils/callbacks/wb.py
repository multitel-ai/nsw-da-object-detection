# Ultralytics YOLO 🚀, AGPL-3.0 license
from ultralytics.utils import SETTINGS, TESTS_RUNNING
from ultralytics.utils.torch_utils import model_info_for_loggers
import os
import yaml
import numpy as np

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS['wandb'] is True  # verify integration is enabled
    import wandb as wb

    assert hasattr(wb, '__version__')

    _processed_plots = {}

except (ImportError, AssertionError):
    wb = None


def _log_plots(plots, step):
    for name, params in plots.items():
        timestamp = params['timestamp']
        if _processed_plots.get(name) != timestamp:
            wb.run.log({name.stem: wb.Image(str(name))}, step=step)
            _processed_plots[name] = timestamp


# def on_pretrain_routine_start(trainer):
#     """Initiate and start project if module is present."""
#     wb.run or wb.init(project = trainer.args.project or 'YOLOv8', name=trainer.args.name, config=vars(trainer.args))

def on_pretrain_routine_start(trainer):
    """Initiate and start project if module is present."""
    yaml_dir = trainer.args.data
    with open(yaml_dir, 'r') as file:
        yaml_file = yaml.safe_load(file)
    train_txt_path = yaml_file['train']#+ 
    val_txt_path = yaml_file['val'] #  + yaml_file['val']
    test_txt_path = yaml_file['test'] # + yaml_file['test']
    with open(train_txt_path) as f: train_images = f.readlines()
    with open(val_txt_path) as f: val_images = f.readlines()
    with open(test_txt_path) as f: test_images = f.readlines()
    train_images = sorted([img.replace('\n', '') for img in train_images])
    val_images = sorted([img.replace('\n', '') for img in val_images])
    test_images = sorted([img.replace('\n', '') for img in test_images])
    train_images_up = np.expand_dims(np.array(train_images), axis=1)
    val_images_up = np.expand_dims(np.array(val_images), axis=1)
    test_images_up = np.expand_dims(np.array(test_images), axis=1)
    control_net = yaml_file['train'].split("/")[-2]
    control_net = control_net[:-3] if control_net[-2]=="." else control_net
    if control_net == "real":
        control_net = "Starting_point" # "baseline"  
    # control_net = control_net + "_coreset" 
    wandb_config = {"control_net":control_net, "data_size":len(train_images), "experiment":"IQA", "sampling":"clipiqa_best"} # , "experiment":"Active"
    wandb_config = {**wandb_config, **vars(trainer.args)}
    wb.run or wb.init(project = trainer.args.project or 'YOLOv8', name = trainer.args.name, 
                      config = wandb_config, entity = "sdcn-nantes")
    table_train = wb.Table(columns=["Train_images"], data = train_images_up)
    table_val = wb.Table(columns=["Val_images"], data=val_images_up)
    table_test = wb.Table(columns=["Test_images"], data=test_images_up)
    wb.log({"Tables/Train": table_train})
    wb.log({"Tables/Val": table_val})
    wb.log({"Tables/Test": table_test})


def on_fit_epoch_end(trainer):
    """Logs training metrics and model information at the end of an epoch."""
    wb.run.log(trainer.metrics, step=trainer.epoch + 1 + trainer.sampling*trainer.epochs)
    _log_plots(trainer.plots, step=trainer.epoch + 1 + trainer.sampling*trainer.epochs)
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1 + trainer.sampling*trainer.epochs)
    if trainer.epoch == 0:
        wb.run.log(model_info_for_loggers(trainer), step=trainer.epoch + 1 + trainer.sampling*trainer.epochs)


def on_train_epoch_end(trainer):
    """Log metrics and save images at the end of each training epoch."""
    wb.run.log(trainer.label_loss_items(trainer.tloss, prefix='train'), step=trainer.epoch + 1 + trainer.sampling*trainer.epochs)
    wb.run.log(trainer.lr, step=trainer.epoch + 1 + trainer.sampling*trainer.epochs)
    if trainer.epoch == 1:
        _log_plots(trainer.plots, step=trainer.epoch + 1 + trainer.sampling*trainer.epochs)


def on_train_end(trainer):
    """Save the best model as an artifact at end of training."""
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1 + trainer.sampling*trainer.epochs)
    _log_plots(trainer.plots, step=trainer.epoch + 1 + trainer.sampling*trainer.epochs)
    art = wb.Artifact(type='model', name=f'run_{wb.run.id}_model')
    if trainer.best.exists():
        art.add_file(trainer.best)
        wb.run.log_artifact(art, aliases=['best'])


callbacks = {
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_train_epoch_end': on_train_epoch_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end} if wb else {}
