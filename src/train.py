import sys
sys.path.append('/home/ucl/elen/tgodelai/.local/lib/python3.9/site-packages') ; sys.path.pop(5) 

import hydra
import os
import sys
import uuid
import yaml
import wandb
import numpy as np
from pathlib import Path
from omegaconf import DictConfig  
import networkx as nx
import torch.nn as nn
import torch
import scipy as sc
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

# TODO: fix this, I do not like this at all
sys.path.append(os.path.join(os.getcwd(), "ultralytics")) 
from ultralytics import YOLO


def find_closest(real_data, gen_data, model, feat=False):

    dist_matrix = []
    gen_data_features = [] 
    for p in model.predict(source=gen_data):
        gen_data_features += [p.cpu().numpy()]
        # print(len(p))
    # print(len(gen_data_features))
    gen_data_features = np.array(gen_data_features)
    
    print("Starting distances computations")
    
    if feat:
        for real in real_data:
            dist_matrix += [np.sqrt(np.sum((real - gen_data_features)**2, axis=-1))]
    else:
        for real_data in model.predict(source=real_data):
            dist_matrix += [np.sqrt(np.sum((real_data.cpu().numpy() - gen_data_features)**2, axis=-1))]
    dist_matrix = np.array(dist_matrix)
    # dist_matrix = (dist_matrix - dist_matrix.min()) / (dist_matrix.max() - dist_matrix.min())
    dist_matrix = dist_matrix.astype(np.float16)
    plt.hist(dist_matrix.flatten())
    plt.savefig("/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/matching.png")
    
    selected = sc.optimize.linear_sum_assignment(dist_matrix)[1]
    """
    selected = algorithm.find_matching(nx.from_numpy_matrix(dist_matrix), 
                                       matching_type = 'min', 
                                       return_type = 'reduced_list')
    """
    return np.array(gen_data)[selected]

class Coreset_Greedy:
    def __init__(self, all_pts):
        self.all_pts = np.array(all_pts)
        self.dset_size = len(all_pts)
        self.min_distances = None
        self.already_selected = [] 
        
    def update_dist(self, centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            centers = [p for p in centers if p not in self.already_selected]

        if centers is not None: #is not None:
            x = self.all_pts[centers]  # pick only centers

            dist = pairwise_distances(self.all_pts, x, metric='euclidean')
            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def sample(self, already_selected, sample_size):
        # initially updating the distances
        if not self.already_selected == []:
            self.update_dist(already_selected, only_new=False, reset_dist=True)
        self.already_selected = already_selected
        new_batch = []
        for i in range(sample_size):
            print("coreset i", i)
            if self.already_selected == []:
                ind = np.random.choice(np.arange(self.dset_size))
            else:
                ind = np.argmax(self.min_distances)
            already_selected.append(ind)
            #assert ind not in already_selected
            self.update_dist([ind], only_new=False, reset_dist=False)
            new_batch.append(ind)

        return new_batch, max(self.min_distances)

def query_coreset(model, real_data, gen_data, data_yaml, sel, selected_path, fold, it):
    print("START coreset")
    used_data = None
    with open(data_yaml, 'r') as _file:
        used_data = yaml.safe_load(_file) 

    old_train = used_data["train"] + ""
    train = used_data["train"]
    if not fold in train:
        train = used_data["train"].replace("train", f"/{fold}/train_" + str(gen_data[0]).split("/")[-2])

    data_yaml_file = used_data
    data_yaml_file["train"] = train
    
    with open(data_yaml, 'w') as _file:
         yaml.dump(used_data, _file) 

    with open(old_train, 'r') as _file:
        used_data = _file.readlines()

    U = [u.replace('\n','') for u in used_data]
    used_data = [u.split("/")[-1] for u in U] 
    real_data = [r for r in real_data if not str(r).split("/")[-1] in used_data]
    gen_data = [g for g in gen_data if not str(g).split("/")[-1] in used_data]

    model.model.query = True 
    real_features = [] 
    # gen_features = []
    # for i, p in enumerate(model.predict(source=gen_data)):
    #     gen_features += [p.cpu().numpy()]
    for i, p in enumerate(model.predict(source=real_data)):
        real_features += [p.cpu().numpy()]

    # real images selection with AL
    coreset = Coreset_Greedy(real_features)
    real_idx_selected, max_distance = coreset.sample([], sel)

    # generated images selection with AL 
    # coreset = Coreset_Greedy(gen_features)
    # gen_idx_selected, max_distance = coreset.sample(used_data, sel)  #([], sel)
    # synt_dataset_selected = np.array(gen_data)[gen_idx_selected]
    # synthetic images selection
    real_features = np.array(real_features)[real_idx_selected]
    print("START hungarian")
    synt_dataset_selected = find_closest(real_features, gen_data, model, feat=True)
    print("STOP  hungarian")
    selected = [str(selected_path / s) for s in synt_dataset_selected]
    print("selected", selected); used_data = list(U) + selected
    if it==1 and False:
        data_yaml_file = used_data_
        train = used_data_["train"].replace("train", f"{fold.split('/')[-1]}/train_" + str(gen_data[0]).split("/")[-2])
        data_yaml_file["train"] = train 
        with open(data_yaml, 'w') as _file: 
           yaml.dump(data_yaml_file, _file)
    with open(train, "w") as _file:
        _file.write("\n".join(used_data))
    print("STOP coreset")

def query(model, real_data, gen_data, data_yaml, sel, selected_path, fold, it):
    print("START query")

    used_data = None
    with open(data_yaml, 'r') as _file:
        used_data = yaml.safe_load(_file) 

    old_train = used_data["train"] + ""
    train = used_data["train"]
    if not fold in train:
        train = used_data["train"].replace("train", f"/{fold}/train_" + str(gen_data[0]).split("/")[-2])
        
    data_yaml_file = used_data
    data_yaml_file["train"] = train
    
    with open(data_yaml, 'w') as _file:
         yaml.dump(used_data, _file) 

    with open(old_train, 'r') as _file: # train plutot que old_train
        used_data = _file.readlines()

    U = [u.replace('\n','') for u in used_data]
    used_data = [u.split("/")[-1] for u in U] 
    #real_data = [r for r in real_data if not str(r).split("/")[-1] in used_data]
    gen_data = [g for g in gen_data if not str(g).split("/")[-1] in used_data]

    results_gen = model.predict(source = gen_data)
    results_gen = [r.boxes.conf.cpu().numpy() for r in results_gen]
    results_gen = [r.max() if len(r)>=1 else 0.9 for r in results_gen]
    results_gen = np.array(gen_data)[np.argsort(results_gen)]

    #results_real = model.predict(source = real_data)
    #results_real = [r.boxes.conf.cpu().numpy() for r in results_real]
    #results_real = [r.max() if len(r)>=1 else 0.9 for r in results_real]
    #results_real = np.array(real_data)[np.argsort(results_real)]

    model.model.query = True  
    gen_features = []
    for i, p in enumerate(model.predict(source=gen_data)):
         gen_features += [p.cpu().numpy()]
    # generated images selection with AL 
    coreset = Coreset_Greedy(gen_features)
    gen_idx_selected, max_distance = coreset.sample([], sel)
    synt_dataset_selected = np.array(gen_data)[gen_idx_selected]

    #model.model.query = True
    #real_data = real_data[::-1][:sel]
    #print("START hungarian")
    #selected = find_closest(real_data, gen_data, model)
    #print("STOP hunagrian")
    #selected = [str(selected_path / s) for s in selected]
    selected = [str(selected_path / s) for s in synt_dataset_selected]
    used_data = list(U) + selected
    if it==1 and False:
        data_yaml_file = used_data_
        train = used_data_["train"].replace("train", f"{fold.split('/')[-1]}/train_" + str(gen_data[0]).split("/")[-2])
        data_yaml_file["train"] = train; print("train it==1", train)
        with open(data_yaml, 'w') as _file:
            yaml.dump(data_yaml_file, _file)
    with open(train, "w") as _file: #train
        _file.write("\n".join(used_data))
    print("STOP query")
    
def query_real(model, real_data, gen_data, data_yaml, sel, selected_path, fold, it):
    print("START query")

    used_data = None
    with open(data_yaml, 'r') as _file:
        used_data = yaml.safe_load(_file) 

    old_train = used_data["train"] + ""
    train = used_data["train"]
    if not fold in train:
        train = used_data["train"].replace("train", f"/{fold}/train_" + str(gen_data[0]).split("/")[-2])
        
    data_yaml_file = used_data
    data_yaml_file["train"] = train
    
    with open(data_yaml, 'w') as _file:
         yaml.dump(used_data, _file) 

    with open(old_train, 'r') as _file: # train plutot que old_train
        used_data = _file.readlines()

    U = [u.replace('\n','') for u in used_data]
    used_data = [u.split("/")[-1] for u in U] 
    # real_data = [r for r in real_data if not str(r).split("/")[-1] in used_data]
    gen_data = [g for g in real_data if not str(g).split("/")[-1] in used_data]
 
    results_gen = np.array(gen_data)  
    
    model.model.query = True   
    
    selected = [str(selected_path / s) for s in results_gen[:sel]]
    used_data = list(U) + selected
    if it==1 and False:
        data_yaml_file = used_data_
        train = used_data_["train"].replace("train", f"{fold.split('/')[-1]}/train_" + str(gen_data[0]).split("/")[-2])
        data_yaml_file["train"] = train; print("train it==1", train)
        with open(data_yaml, 'w') as _file:
            yaml.dump(data_yaml_file, _file)
    with open(train, "w") as _file: #train
        _file.write("\n".join(used_data))
    print("STOP query")

@hydra.main(version_base=None, config_path=f"..{os.sep}conf", config_name="config")
def main(cfg: DictConfig) -> None:
    data = cfg['data'] ; active = cfg['active']
    base_path = Path(data['base']) 
    GEN_DATA_PATH =  Path(base_path) / data['generated'] / cfg['model']['cn_use']
    
    if cfg['ml']['augmentation_percent']==0 or active["abled"]:
        REAL_DATA = Path(base_path) / data['real']
    else:
        fold = cfg['model']['cn_use'] + str(cfg['ml']['augmentation_percent'])
        REAL_DATA = Path(base_path) / data['real'] / fold
    
    run = "" if cfg['ml']['augmentation_percent']==0 else cfg['model']['cn_use'] + str(cfg['ml']['augmentation_percent'])
    data_yaml_path = REAL_DATA / 'data.yaml'
    model_path = "/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/" / Path(base_path) / str( cfg['model']['cn_use'] + str(cfg['ml']['augmentation_percent']) + ".pt")

    cn_use = cfg['model']['cn_use']
    aug_percent = cfg['model']['augmentation_percent']
    name_ = f"{uuid.uuid4().hex.upper()[0:6]}_{cn_use}_{aug_percent}"

    data_yaml_path = str(data_yaml_path.absolute())
    if active["abled"]: 
        
        if active["sampling"]=="confidence":
            log_name = ""
        elif active["sampling"]=="coreset":
            log_name = "_coreset"
        elif active["sampling"]=="baseline":
            log_name = "_baseline"
    
        START = 0
        if os.path.exists(model_path):
            START = 1
            
        for _ in range(START, active['rounds']):
            name = "_".join(name_.split("_")[:-1]) + f"_active_{_}"
            if _==0: 
                model = YOLO("yolov8n.yaml")
                torch.save({"model":model.model.cpu()}, model_path)
                train = '/'.join(str(data_yaml_path).split('/')[:-1]) + f"/active{log_name}_{cn_use}"
                if os.path.isdir(train):
                    os.system(f"rm {train}/*") 
            
            _data_yaml_path = data_yaml_path + ""
            _data_yaml_path2 = _data_yaml_path + ""
            print("data_yaml_path", data_yaml_path)  
            if os.path.exists(model_path) and _>0:
                model = YOLO("yolov8n.yaml")
                mod = torch.load(model_path)['model']
                model.model = mod
                model.model.query = False
                model.predictor = None
                if _>=1: 
                    _data_yaml_path2 = _data_yaml_path.replace(".yaml", "_" + cn_use + ".yaml")
                    fold = '/'.join(str(_data_yaml_path2).split('/')[:-1]) + f"/active{log_name}_{cn_use}"; print("fold", fold) #f"active_corest_{cn_use}"
                    _data_yaml_path2 = fold + "/" + str(_data_yaml_path2).split('/')[-1] # enlever le +"/"
                    print(f"data_yaml_path2 it=={_}", _data_yaml_path2) 
                    if not os.path.isdir(fold): 
                        os.makedirs(fold)
                    if not os.path.exists(_data_yaml_path2):
                        os.system(f"cp {_data_yaml_path} {_data_yaml_path2}") 
                #if not os.path.exists(_data_yaml_path2):
                #    if _==1:
                #         os.system(f"cp {_data_yaml_path} {_data_yaml_path2}")
                #    else:
                #        assert False, "There's smthg wrong here"
                 
                #print("QUERY CORESET")
                #print("im real", os.listdir(Path(base_path)/data['real']/"coco/Coco_1FullPerson")[0])
                query_f = None
                if active["sampling"]=="confidence":
                    query_f = query
                elif active["sampling"]=="coreset":
                    query_f = query_coreset
                elif active["sampling"]=="baseline":
                    query_f = query_real
                query_f(model,  # _coreset
                      [Path(base_path) / data['real'] / "coco/Coco_1FullPerson" / im for im in sorted(os.listdir(Path(base_path) / data['real'] / "coco/Coco_1FullPerson") )
                       if not im in os.listdir(Path(base_path) / data['real'] / "images")],  # on rajoute a chaque fois des images reelles?
                      # [Path(base_path) / data['real'] / "coco/Coco_1FullPerson" / im for im in sorted(os.listdir(Path(base_path) / data['real'] / "coco/Coco_1FullPerson") )
                      #  if not im in os.listdir(Path(base_path) / data['real'] / "images")],
                      [GEN_DATA_PATH / im for im in sorted(os.listdir(GEN_DATA_PATH))], 
                      str(_data_yaml_path2), 
                      active['sel'],
                      Path("/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/"),
                      f"active{log_name}_{cn_use}", _)
                print("end QUERY CORESET") 
    
            print("data_yaml_path2", _data_yaml_path2)   
            model = YOLO("yolov8n.yaml")
            model.train(
                sampling=0,
                data = _data_yaml_path2,
                epochs = cfg['ml']['epochs'],
                project = 'sdcn-coco',
                control_net=cn_use,
                ALsampling=cfg['logs']['sampling'], 
                experiment=cfg['logs']['experiment'],
                name = name,
            )
            if not wandb.run is None:
                wandb.run.finish()
            torch.save({"model":model.model.cpu()}, model_path)
        # os.system(f"rm {model_path}")
    elif cfg['ml']['baseline']:
       print("enter baseline loop")
       name = f"{uuid.uuid4().hex.upper()[0:6]}_baseline2_{str(cfg['ml']['augmentation_percent_baseline'])}" 
      
       data_yaml_path_2 = data_yaml_path + ""
       data_yaml_path_2 = data_yaml_path.replace(".yaml", "_" + 'baseline2' +'_' + str(cfg['ml']['augmentation_percent_baseline']) + ".yaml")
       fold = '/'.join(str(data_yaml_path_2).split('/')[:-1]) + f"/baseline2_{str(cfg['ml']['augmentation_percent_baseline'])}"
       data_yaml_path_2 = fold + "/" + str(data_yaml_path_2).split('/')[-1]
       if not os.path.isdir(fold):
          os.makedirs(fold)
       if not os.path.exists(data_yaml_path_2):
          os.system(f'cp {data_yaml_path} {data_yaml_path_2}')
       
       used_data = None
       with open(data_yaml_path, 'r') as _file:
          data = yaml.safe_load(_file) 

       old_train = data["train"] + ""
       train = data["train"]
       if not fold in train:
          print("enter fold loop")
          train = data["train"].replace("train", f"{fold.split('/')[-1]}/train") #("train", "/train_" + str(cn_use) + '_' + str(cfg['ml']['augmentation_percent']))
          print("train", train)
       data_yaml_file = data
       data_yaml_file["train"] = train

       with open(data_yaml_path_2, 'w') as _file:
          yaml.dump(data, _file) 

       with open(old_train, 'r') as _file: # train plutot que old_train
          used_data = _file.readlines()
       used_data = [u.replace('\n','') for u in used_data]
       print("used data", used_data)
       sel = int(250*(cfg['ml']['augmentation_percent_baseline']))
       if sel<len(used_data):
          used_data_ = used_data + used_data[:sel] 
       else: 
          used_data_ = used_data + int(cfg['ml']['augmentation_percent_baseline'])*used_data
       print("len used_data", len(used_data_))
       with open(train, 'w') as _file: 
          _file.write('\n'.join(used_data_))
       print("start training"); print("yaml_path", data_yaml_path_2)
       model = YOLO("yolov8n.yaml")
       model.model.query = False
       model.train(
          data = str(Path(data_yaml_path_2).absolute()),
          epochs = cfg['ml']['epochs'],
          project = 'sdcn-coco',
          control_net=cn_use,
          ALsampling=cfg['logs']['sampling'], 
          experiment=cfg['logs']['experiment'],
          name = name
       )

    else:
        model = YOLO("yolov8n.yaml")
        model.model.query = False
        model.train(
            data = str(Path(data_yaml_path).absolute()),
            epochs = cfg['ml']['epochs'],
            project = 'sdcn-coco',
            # entity = 'sdcn-nantes',
            control_net=cn_use,
            ALsampling=cfg['logs']['sampling'], 
            experiment=cfg['logs']['experiment'],
            name = name_
        )


if __name__ == '__main__':
    main()
