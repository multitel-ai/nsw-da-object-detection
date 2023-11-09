from create_dataset import *
from scipy import ndimage as nd
import numpy as np

def create_faces_dataset(formats: List[str] = ['jpg', 'png', 'jpeg']):
    """
    Construct the txt file containing a percentage of real and synthetic data

    :param str real_images_dir: path to the folder containing real images
    :param str synth_images_dir: path to the folder containing synthetic images
    :param str txt_dir: path used to create the txt file
    :param float per_synth_data: percentage of synthetic data compared to real (ranges in [0, 1])

    :return: None
    :rtype: NoneType
    """

    N = 250 ; V = 300 ; T = 300
    images_path = "/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/data_faces/images/"
    real_dir = "/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/data_faces/real/"
    val_dir = "/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/data_faces/val/"
    test_dir = "/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/data_faces/test/"

    real_images = list_images(Path(images_path), formats)
    to_keep = [] ; threshold = 0.2
    
    for ire, real in enumerate(real_images):
        label_path = real.replace("images","labels").split(".")[0] + ".txt"
        if not os.path.exists(label_path):
            continue
        label = np.loadtxt(label_path)
        
        if len(label.shape)==1: # and len(label)>1: # skip if several boxes
            # continue
            label = label[None,:]
        
        area = label[:,-1]*label[:,-2]
        if area.max() >= threshold:
            to_keep.append(real)
            print(len(to_keep), ire)
            
    if len(to_keep)<N + V + T:
        print(len(to_keep))
        assert False, "threshold too high"
    
    real = str(to_keep[:N]).replace("[","").replace("]","").replace(",","")
    val = str(to_keep[N:N+V]).replace("[","").replace("]","").replace(",","")
    test = str(to_keep[N+V:N+V+T]).replace("[","").replace("]","").replace(",","")
    
    os.system(f"cp {real} {real_dir}")
    os.system(f"cp {val} {val_dir}")
    os.system(f"cp {test} {test_dir}")
    """
    for img_path in real:
        img = nd.imread(img_path)
        H, W = img.shape
        label_path = real_images_path.replace("images","labels") + f"{os.sep}" + img_path.split(f"{os.sep}")[-1].split(".")[0] + ".txt"
        label = np.loadtxt(label_path)[:,1:]
        area = label[:,-1]*label[:,-2]
        label = label[area.argmax(), :]
        x,y,h,w = label
        h = 1.25*h
        w = 1.25*w
        x,y,h,w = int(x*W), int(y*H), int(h*H), int(w*W)
        crop = img[y:y+h,x:x+w]
        nd.imwrite(crop, img_path)
    """   
        
        
if __name__=="__main__":
    
    create_faces_dataset()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
