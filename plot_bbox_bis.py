from PIL import Image
import sys, os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image 

sys.path.append(os.path.join(os.getcwd(), "ultralytics")) 
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
i = 1
mod = "best" ; img = "momo" # f"data/generated/controlnet_segmentation/000000241565_4"
model = YOLO(f'{mod}.pt')

# Run inference on 'bus.jpg'
results = model(f'/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/{img}.png', conf=0.5)  # results list

# Show the results
for ri, r in enumerate(results): 
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image
    # im.save(f'{mod}_{i}.jpg', quality=95)  # save image

# Label 
label_path = "/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/data/real/labels/000000241565.txt"
label = np.loadtxt(label_path) 
print(np.shape(im_array))
# Create figure and axes
fig, ax = plt.subplots() 
# Display the image
ax.imshow(im) 

dx, dy = 640, 448
x, y, w, h = label[1]*dx, label[2]*dy, label[3]*dx , label[4]*dy
x = int(x - w/2)
y = int(y - h/2)
# Create a Rectangle patch
rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none') 
# Add the patch to the Axes
ax.add_patch(rect)

plt.savefig(f'/auto/home/users/t/g/tgodelai/after_nantes/nsw-da-object-detection/momo-seg.png', dpi=300) # _{i}