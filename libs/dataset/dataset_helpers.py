#from torchvision import transforms
import numpy as np
from PIL import Image
from skimage import io

def getPILimage(path):
        #Get orignal image
        img = Image.open(path)
        img = img.convert('RGB')
        
        return img
  
def getSkiimage(path):
        img = io.imread(path)
         #Check 3 channels 
        if(len(img.shape)<3):
              w, h = img.shape
              ret = np.empty((w,h,3), dtype=np.uint8)
              ret[:,:,2] = ret[:,:,1] = ret[:,:,0] = img
              img = ret
        
        return img
    
