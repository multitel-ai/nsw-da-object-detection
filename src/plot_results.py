import gc
import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import glob
import wandb
import csv
import ast
import hydra
import json 
import yaml
from omegaconf import DictConfig
sys.path.append(os.path.join(os.getcwd(), "ultralytics")) 
from ultralytics import YOLO

def create_csv_results(runs_weight_dir,csv_file_, file_name, yaml_path, entity, project, metric='map'):
    """
    Plot mAP metrics
    :param runs_weight_dir: str, Path to the folder that contains the pt files
    :param file_name: str, Name file of the plot
    :param yaml_path: str, Yaml file containing the paths to the txt files
    :param entity: str, wandb username or team
    :param project: str, wandb project
    :param metric: str, ['map', 'map50', 'precision', 'recall', 'fitness']
    :return:
    """
    if os.path.isdir(runs_weight_dir):
        os.system(f"rm {runs_weight_dir}/*")
    if not os.path.isdir(runs_weight_dir):
        os.makedirs(runs_weight_dir)

    api = wandb.Api()
    results_df = pd.DataFrame(columns=['id', 'perc_syn', 'method', 'sampling', 'map', 'map50', 'precision', 'recall',
                                       'fitness'])
    train_results_df = pd.DataFrame(columns=['id', 'perc_syn', 'method', 'sampling', 'map', 'map50', 'prediction',
                                             'recall', 'fitness'])
    n_runs = 143
    csv_file_ = pd.read_csv('result2.csv')
    train_csv_file_ = pd.read_csv('result1.csv')
    for i in range(0, n_runs):
        gc.collect()
        # Download weights from wandb
        os.system("python3 src/download.py --list-all --folder " + str(Path(runs_weight_dir).absolute()) + " --entity " + str(entity)
              + " --project " + str(project) + " -i " + str(i) + " -lf -d")
        weight = [w for w in os.listdir(runs_weight_dir) if w.endswith('.pt')]; print("weight", weight)
        if len(weight)>0:
           weight = weight[0]
           csv_path = [w for w in os.listdir(runs_weight_dir) if w.endswith('.csv')][0]
           csv_file = open(Path(runs_weight_dir) / Path(csv_path))
           csv_reader = csv.reader(csv_file)
           rows = []
           for row in csv_reader:
               rows.append(row)
           idx = np.argwhere(np.array(rows)[:,-1]=='.'.join(weight.split('.')[1:-1]))[0][0]

           run_id = weight.split("_")[0].split(".")[0]
           data_size = ast.literal_eval(rows[idx][2])['data_size']  # perc
           cn_use = ast.literal_eval(rows[idx][2])['control_net'] #method 
           sampling = ast.literal_eval(rows[idx][2])['sampling'] #sampling
           weight = weight
           if 'coreset' in weight:
              sampling = 'coreset'
           if 'conf' in weight:
              sampling = 'confidence'
           if 'baseline' in weight:
              cn_use = 'real-large_dataset'
              sampling = 'baseline'
           if 'baseline2' in weight:
              cn_use = 'real-small_dataset'
              sampling = 'baseline_2'

           if str(run_id) in list(csv_file_['id']):
              idx_ = np.argwhere(np.array(csv_file_['id']) == str(run_id))[0][0]
              csv_file_['method'][idx_] = cn_use
              csv_file_['sampling'][idx_] = sampling
              train_csv_file_['method'][idx_] = cn_use
              train_csv_file_['sampling'][idx_] = sampling
              print("present")
           else:
              weight_path = Path(runs_weight_dir) / Path(weight)
              model = YOLO(weight_path) #weight_path
              new_yaml_path = ast.literal_eval(rows[idx][2])['data']
              with open(Path(yaml_path).absolute(), 'r') as f:
                  yaml_file = yaml.safe_load(f)
              with open(new_yaml_path, 'w') as f:
                  json.dump(yaml_file, f)
              results = model.val(split='test', batch=100) #data = Path(yaml_path).absolute(), split='test') #conf= #MODIFIER CONF
              map = results.box.map        # map50-95
              map50 = results.box.map50    # map50
              precision = results.box.mp   # precision
              recall = results.box.mr      # recall
              fitness = results.fitness    # fitness
              #map = 0; map50 = 0; precision = 0; recall = 0; fitness = 0
              new_results = {
                  'id': run_id,
                  'perc_syn': data_size,
                  'method': cn_use,
                  'map': map,
                  'map50': map50,
                  'precision': precision,
                  'recall': recall,
                  'fitness': fitness,
                  'set': 'test',
                  'sampling': sampling
              }
              results_df = pd.concat([results_df, pd.DataFrame([new_results])], ignore_index=True)

              run = api.run(str(entity) + '/' + str(project) + '/' + str(run_id))
              map50 = run.history(keys=['metrics/mAP50(B)'])
              map50 = np.mean((map50['metrics/mAP50(B)'].to_numpy())[-50:])
              map = run.history(keys=['metrics/mAP50-95(B)'])
              map = np.mean((map['metrics/mAP50-95(B)'].to_numpy())[-50:])
              precision = run.history(keys=['metrics/precision(B)'])
              precision = np.mean((precision['metrics/precision(B)'].to_numpy())[-50:])
              recall = run.history(keys=['metrics/recall(B)'])
              recall = np.mean((recall['metrics/recall(B)'].to_numpy())[-50:])
              train_new_results = {
                  'id': run_id,
                  'perc_syn': data_size,
                  'method': cn_use,
                  'map': map,
                  'map50': map50,
                  'precision': precision,
                  'recall': recall,
                  'set': 'train',
                  'sampling': sampling
              }
              train_results_df = pd.concat([train_results_df, pd.DataFrame([train_new_results])], ignore_index=True)

           # Remove weight file
           os.system(f"rm {runs_weight_dir}/*")
        else:
           print("Probl", i); pass

    # save results
    train_results_df =  pd.concat([train_results_df, train_csv_file_], ignore_index=True)
    train_results_df.to_csv("result1.csv")
    results_df =  pd.concat([results_df, csv_file_], ignore_index=True)
    results_df.to_csv("result2.csv")

def plot_from_csv(csv_path, file_path, metric):
    results_df = pd.read_csv(csv_path)
    sort_results = results_df.sort_values(by=['perc_syn'])
    
    stp = results_df["map"].values[results_df["perc_syn"].values==250]
    fp = 0.65 # results_df["map"].values[np.logical_and(results_df["sampling"].values=='baseline', results_df["perc_syn"].values=='625')]
    
    results_1 = pd.concat([sort_results[sort_results["sampling"] == 'random'],
                          sort_results[np.logical_and(sort_results["sampling"] == 'baseline', sort_results["perc_syn"].values>250)],
                          sort_results[sort_results["sampling"] == 'baseline_2']], ignore_index=True)

    fig, ax = plt.subplots(1,1, figsize=(8,7))
    ax = sns.lineplot(
        data=results_1, 
        x='perc_syn', y=metric, style="method",
        hue='method', 
        markers=True, #["o","s","D",'v',"*"],
        dashes=[(2,1),(2,1),(2,1),(2,1),(2,1)],
        ax=ax,
    )
    lims =  (0,1) 
    ax.axhline(stp[0], lims[0], lims[1], linestyle='--', color="black")
    ax.text(-20, 0.01+stp[0], f'  mAP={stp[0]:.2f}        ')
    ax.axhline(fp, lims[0], lims[1]) 
    ax.text(-20, 0.01+fp, f'  mAP={fp:.2f}         ')
    ax.set_xlabel('Number of samples in the dataset')
    ax.set_ylabel(metric)
    ax.set_title(str(metric) + ' as function of the number of images in the dataset')
    fig.savefig(Path(file_path).absolute()/Path('random_sampling.png'))
    
    control_nets = ['lllyasviel_canny', 'lllyasviel_openpose', 'controlnet_segmentation', 'crucible_mediapipe_face']
    markers = ['o', 'X', 's', 'P', 'D']
    for i, cn in enumerate(control_nets):
        sort_results_i = sort_results[results_df["sampling"]!='baseline']
        sort_results_i = sort_results_i[sort_results_i["sampling"]!='baseline_2']
        result_2 = sort_results_i[sort_results_i["method"] == cn]
        
        fig, ax = plt.subplots(1,1, figsize=(8,7))
        ax = sns.lineplot(
            data=result_2, 
            x='perc_syn', y=metric, style="sampling",
            hue='sampling', 
            marker=markers[i], 
            dashes=[(2,1),(2,1),(2,1),(2,1),(2,1)],
            ax=ax,
        )
        lims =  (0, 1)
    
        ax.axhline(stp[0], lims[0], lims[1], linestyle='--', color="black")
        ax.text(-20, 0.01+stp[0], f'  mAP={stp[0]:.2f}        ')
        ax.axhline(fp, lims[0], lims[1]) 
        ax.text(-20, 0.01+fp, f'  mAP={fp:.2f}         ')
        ax.set_xlabel('Number of samples in the dataset')
        ax.set_ylabel(metric)
        ax.set_title(str(metric) + ' as function of the number of images in the dataset')
        fig.savefig(Path(file_path)/Path(str(cn)+'_samplings.png'))

def run():
    parser = argparse.ArgumentParser(description="Plot")
    parser.add_argument("--runs_weight_dir", type=str,
                        help="Directory of the folder that contains the pt files")
    parser.add_argument("--metric", type=str, default='map',
                        help="Metric to plot, [map, map50, precision, recall, fitness")
    parser.add_argument("--file_name", type=str, help="Name file", default='test.png')
    parser.add_argument("--yaml_path", type=str, help="Yaml file containing the testing path")
    parser.add_argument("--img_dir", type=str, help='Directory of the folder containing images to test', default=None)
    parser.add_argument("--entity", type=str, help='wandb team or username')
    parser.add_argument("--project", type=str, help='wandb project')
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    plot(args.runs_weight_dir, args.file_name, args.yaml_path, args.entity, args.project, args.metric)

@hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
def run(cfg: DictConfig) -> None:
   runs_weight_dir = cfg['plot']['weights_dir']; file_name = cfg['plot']['name']; yaml_file = cfg['plot']['yaml_file']
   entity = cfg['plot']['entity']; project = cfg['plot']['project']; metric = cfg['plot']['metric']
   if cfg['plot']['plot_from_csv']: 
      plot_from_csv(cfg['plot']['csv_path'], file_name, metric)
   else:
      create_csv_results(runs_weight_dir, cfg['plot']['csv_path'], file_name, yaml_file, entity, project, metric)

if __name__ == '__main__':
   run()

