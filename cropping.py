#import packages
from PIL import Image
from os import listdir

originalDirectory = '/Users/blaiseswartwood/Downloads/Dataprep/Scalogram/train/normal'
for file in listdir(originalDirectory):

   newSaveDir = '/Users/blaiseswartwood/Downloads/Dataprep2/Scalogram/train/Normal'
   saveCroppedFile = newSaveDir + '/' + file
   originalFile = originalDirectory + '/' + file
   im = Image.open(originalFile)

   # Size of the image in pixels (size of original image)
   width, height = im.size
   # Setting the points for cropped image

   left = 60
   top = 65
   right = 420
   bottom = 415

   # Cropped image of above dimension

   im1 = im.crop((left, top, right, bottom))
   im1 = im1.save(saveCroppedFile)
