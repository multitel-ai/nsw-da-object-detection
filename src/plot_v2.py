import os
import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

# TODO: fix this, I do not like this at all
sys.path.append(os.path.join(os.getcwd(), "ultralytics")) 
from ultralytics import YOLO

os.environ["N_real_images"] = "5000"

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
    os.system("python3 download.py --list-all --folder " + str(runs_weight_dir) + " --entity " + str(entity)
              + " --project " + str(project) + " -lf -d")
    weights = glob.glob(runs_weight_dir + "*.pt")
    print("weights", weights)
    #weights = sorted(os.listdir(runs_weight_dir))
    if weights and weights[0] == '.DS_Store': weights = weights[1:]

    # Create dataframe containing results
    results_df = pd.DataFrame(columns=['id', 'perc_syn', 'method', 'map', 'map50', 'precision', 'recall', 'fitness'])
    for i, weight in enumerate(weights):
        #weight_path = runs_weight_dir + weight
        model = YOLO(weight) #weight_path
        results = model.val(data = yaml_path, split='test') #conf= #MODIFIER CONF

        wandb_id = weight.split('_')[0]
        method = weight.split('_')[1:-1]
        method = '_'.join(method)
        perc = weight.split('_')[-1].split('.')[:-1]
        perc = '.'.join(perc)
        map = results.box.map        # map50-95
        map50 = results.box.map50    # map50
        precision = results.box.mp   # precision
        recall = results.box.mr      # recall
        fitness = results.fitness    # fitness

        new_results = {'id': wandb_id,
                        'perc_syn': perc,
                        'method': method,
                        'map': map,
                        'map50': map50,
                        'precision': precision,
                        'recall': recall,
                        'fitness': fitness,
                       }
        results_df = pd.concat([results_df, pd.DataFrame([new_results])], ignore_index=True)

    print("results_df", results_df)

    results_perc0 = results_df[results_df['perc_syn'] == str(0.0)]
    mean_results_perc0 = results_perc0[metric].mean()
    results_perc = results_df[results_df['perc_syn'] != str(0.0)]
    sort_results = results_perc.sort_values(by=['perc_syn'])

    # Plot and save file
    fig = sns.relplot(
        data = sort_results, kind = 'line',
        x = 'perc_syn', y = metric,
        hue = 'method',
        palette = 'pastel'
    )
    fig.map(plt.axhline, y=mean_results_perc0, color=".7", dashes=(2, 1), zorder=0, label='baseline')
    fig.set(xlabel = 'Percentage of generated data compared to real ones',
            ylabel = metric,
            title = str(metric) + ' as function of the percentage of generated images in the dataset')
    fig.savefig(file_name)

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