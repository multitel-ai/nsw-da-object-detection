#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:49:24 2023

@author: tgodelaine
"""

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
sys.path.append(os.path.join(sys.path[0], "yolov8"))

os.environ["N_real_images"] = "5000"
from ultralytics import YOLO

def plot(runs_weight_dir, file_name, yaml_path, entity, project, metric='map'):
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

    # Download weights from wandb
    #os.system("python3 download.py --list-all --folder " + str(runs_weight_dir) + " --entity " + str(entity)
    #          + " --project " + str(project) + " -lf -d")
    weights = [w for w in os.listdir(runs_weight_dir) if w.endswith('.pt')] #32
    #print("weights", weights)

    api = wandb.Api()
    # Create dataframe containing results
    results_df = pd.DataFrame(columns=['id', 'perc_syn', 'method', 'map', 'map50', 'precision', 'recall', 'fitness'])
    train_results_df = pd.DataFrame(columns=['id', 'perc_syn', 'method', 'sampling', 'map', 'map50', 'prediction', 'recall', 'fitness'])
    for i, weight in enumerate(weights):
        sampling = 'baseline'
        iqa = False
        if 'iqa' in weight:
            iqa = True
            weight = ('_').join(weight.split('_')[:-1])+'.pt'
            sampling = 'iqa'
        active = False
        if 'active' in weight:
            active = True
            weight = ('_').join(weight.split('_')[:-2]+[weight.split('_')[-1]])
            sampling = 'active'

        run_id = weight.split('_')[0].split('.')[0]
        method = weight.split('_')[1:-1]
        method = '_'.join(method)
        perc = weight.split('_')[-1].split('.')[:-1]
        perc = '.'.join(perc)
        if perc == '0': perc = '0.0'
        if active:
            if perc=='1': perc='0.5'
            if perc=='2': perc='1.0'
            if perc=='3': perc='1.5'
            if perc=='4': perc='2.0'
            if perc=='5': perc='2.5'
            #method = method + '_active'
        #if iqa:
            #method = method + '_iqa'


        weight_path = Path(runs_weight_dir) / Path(weight)
        #model = YOLO(weight_path) #weight_path
        #results = model.val(data = yaml_path, split='test') #conf= #MODIFIER CONF
        #map = results.box.map        # map50-95
        #map50 = results.box.map50    # map50
        #precision = results.box.mp   # precision
        #recall = results.box.mr      # recall
        #fitness = results.fitness    # fitness
        map = 0; map50 = 0; precision = 0; recall = 0; fitness = 0
        new_results = {
            'id': run_id,
            'perc_syn': perc,
            'method': method,
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
            'perc_syn': perc,
            'method': method,
            'map': map,
            'map50': map50,
            'precision': precision,
            'recall': recall,
            'set': 'train',
            'sampling': sampling
        }
        train_results_df = pd.concat([train_results_df, pd.DataFrame([train_new_results])], ignore_index=True)
        #if perc == '0.5': print("concat", train_new_results)
    #results_perc0 = results_df[results_df['perc_syn'] == str(0.0)]
    #mean_results_perc0 = results_perc0[metric].mean()
    #results_perc = results_df[results_df['perc_syn'] != str(0.0)]
    #sort_results = results_perc.sort_values(by=['perc_syn'])
    sort_results = results_df.sort_values(by=['perc_syn'])
    train_sort_results = train_results_df.sort_values(by=['perc_syn'])
    concat = train_results_df[train_results_df["method"]=='crucible_mediapipe_face']#pd.concat([results_df, train_results_df], ignore_index=True)
    concat = concat.sort_values(by=['perc_syn'])

    # Plot and save file
    fig = sns.relplot(
        data=concat, kind='line',
        x='perc_syn', y=metric,
        hue='method', size='sampling',
        #palette='pastel'
    )
    #fig.map(plt.axhline, y=mean_results_perc0, color=".7", dashes=(2, 1), zorder=0, label='baseline')
    fig.set(xlabel = 'Percentage of generated data compared to real ones',
            ylabel = metric,
            title = str(metric) + ' as function of the percentage of generated images in the dataset')
    fig.set(ylim=(0.3, 1))
    fig.savefig(file_name)

# def plot_2():
#     api = wandb.Api()
#     entity = 'sdcn-nantes'
#     project = 'sdcn'
#     run_id = 'g6nv4b18'
#
#     run = api.run(entity + '/' + project + '/' + run_id)
#     mAP50 = run.history(keys=['metrics/mAP50(B)'])
#     mean_mAP50 = np.mean((mAP50['metrics/mAP50(B)'].to_numpy())[-50:])
#     print("mean mAP", mean_mAP50)


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
    #plot_images(args.runs_weight_dir, img_dir)

if __name__ == '__main__':
    run()
    #plot_2()

# Load a model
#model = YOLO("yolov8n.yaml")

# Use the model
#model.train(data = '/Users/tgodelaine/Desktop/data/coco_0.yaml', epochs = 1)  # train the model