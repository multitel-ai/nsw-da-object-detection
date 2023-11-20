from PIL import Image
import sys, os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image 
sys.path.append(os.path.join(os.getcwd(), "ultralytics")) 
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
mod = "D250" ; img = "data/real/images/000000241565"
model = YOLO(f'{mod}.pt')

# Run inference on 'bus.jpg'
results = model(f'/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/{img}.jpg')  # results list

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image
    # im.save(f'{mod}_{i}.jpg', quality=95)  # save image

# Label 
label_path = "/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/data/real/labels/000000241565.txt"
label = np.loadtxt(label_path) 

# Create figure and axes
fig, ax = plt.subplots() 
# Display the image
ax.imshow(im) 

x, y, w, h = label[1]*640, label[2]*424, label[3]*640 , label[4]*424
x = int(x - w/2)
y = int(y - h/2)
# Create a Rectangle patch
rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lightgreen', facecolor='none') 
# Add the patch to the Axes
ax.add_patch(rect)

plt.savefig(f'/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/BBOX/{mod}.png', dpi=300)

models = ["canny", "mediapipe"] # , "segmentation"] # "D250",
# models = ["openpose"] # , "segmentation"] # "D250",
for i in [1,2,3,4,5]:
    images = [f"data/generated/lllyasviel_canny/000000241565_{i}", #  f"data/generated/lllyasviel_openpose/000000241565_{i}", # "data/real/images/000000241565", 
              f"data/generated/crucible_mediapipe_face/000000241565_{i}"] # , "data/real/images/000000241565_4"] 
    # images = [f"data/generated/lllyasviel_openpose/000000241565_{i}"] # , "data/real/images/000000241565_4"] 
    
    for mod, img in zip(models,images): 
        # Load a pretrained YOLOv8n model
        model = YOLO(f'{mod}.pt')
        
        # Run inference on 'bus.jpg'
        results = model(f'/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/{img}.png')  # results list
        
        # Show the results
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            # im.show()  # show image
            # im.save(f'{mod}_{i}.jpg', quality=95)  # save image
        
        print(np.shape(im_array))
        # Label 
        label_path = "/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/data/real/labels/000000241565.txt"
        label = np.loadtxt(label_path) 
        
        # Create figure and axes
        fig, ax = plt.subplots() 
        # Display the image
        ax.imshow(im) 
        
        x, y, w, h = label[1]*640, label[2]*424, label[3]*640 , label[4]*424
        
        x = int(x - w/2)
        y = int(y - h/2)
        
        # Create a Rectangle patch
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lightgreen', facecolor='none') 
        # Add the patch to the Axes
        ax.add_patch(rect)
        
        plt.savefig(f'/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/BBOX/{mod}_{i}.png', dpi=300)
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    