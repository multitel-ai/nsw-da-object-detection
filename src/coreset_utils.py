import yaml
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


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
    # plt.savefig("./matching.png")
    
    selected = sc.optimize.linear_sum_assignment(dist_matrix)[1]
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
        train = used_data["train"].replace("train", f"{fold}/train_" + str(gen_data[0]).split("/")[-2])

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
        train = used_data["train"].replace("train", f"{fold}/train_" + str(gen_data[0]).split("/")[-2])
        
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

    print('===>', gen_data)

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

    selected = [str(selected_path / s) for s in synt_dataset_selected]
    used_data = list(U) + selected
    
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
        train = used_data["train"].replace("train", f"{fold}/train_" + str(gen_data[0]).split("/")[-2])
        
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
    with open(train, "w") as _file: #train
        _file.write("\n".join(used_data))
    print("STOP query")
