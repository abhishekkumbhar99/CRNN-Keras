import cv2
import os, random
import  numpy  as  np
from parameters import *
import keras
from albumentations import (GaussNoise, Blur,MotionBlur, MedianBlur,IAAPerspective,IAASharpen, ShiftScaleRotate,
    Rotate, RandomBrightnessContrast,RandomBrightness, InvertImg,CLAHE,Compose,OneOf)
# # Input data generator
def labels_to_text(labels):     # letters index -> text (string)
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def  text_to_labels ( text ):       # convert text to index value in letters array
    return list(map(lambda x: letters.index(x), text))

class TextImageGenerator(keras.utils.Sequence):
    def __init__(self, img_dirpath, img_w, img_h, batch_size, 
        downsample_factor, max_text_len=10, shuffle=False, augment = False):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath                  # image dir path
        self.img_list = os.listdir(self.img_dirpath)     # images list
        self.n = len(self.img_list)                      # number of images
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()
        
    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(self.n/self.batch_size))

    def __getitem__(self,index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        batch_filenames = [self.img_list[i] for i in indices]

        X, y = self.__generate_data(batch_filenames)
        
        return X, y


    def on_epoch_end(self):
        self.indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def load_img_and_label(self, filename):
        img = cv2.imread(self.img_dirpath + filename, cv2.IMREAD_GRAYSCALE)
        #Resize the image to reqd height keeping the aspect ratio unchanged
        img = cv2.resize(img, (int(self.img_h * (img.shape[1]/img.shape[0])), self.img_h))
        
        #pad the image if width is less than img_w
        if img.shape[1]<self.img_w:
            new_img = np.ones((self.img_h,self.img_w), dtype=np.uint8)*255
            new_img[:img.shape[0], :img.shape[1]] = img[:, :]
            img = new_img
        elif img.shape[1]>self.img_w:
            img = cv2.resize(img, (self.img_w, self.img_h))

        img_name = filename[0:-4].split('_')[0]
        img_name = "".join(img_name.split())

        return img, img_name

    def augmentation(self,img, prob=0.75):
        aug = Compose([
        GaussNoise(p=0.2),
        OneOf([
            MotionBlur(p=.2,blur_limit=5),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        IAAPerspective(scale=(0.00,0.05),p=1),
        # Either shiftscale or rotate
        OneOf([ShiftScaleRotate(shift_limit=0.01, scale_limit=(-0.2,0.1), rotate_limit=0, 
                                border_mode = cv2.BORDER_CONSTANT),
               Rotate(limit = 5, interpolation=cv2.INTER_CUBIC, border_mode = cv2.BORDER_CONSTANT)], p=0.5),
        OneOf([
            IAASharpen(),
            RandomBrightnessContrast(),            
        ], p=0.5),
        RandomBrightness(),
        CLAHE(clip_limit=2),
        # InvertImg(always_apply=True, p=1.0)
        ], p=prob)

        img = aug(image=img)['image']

        return img 


    def __generate_data(self, batch_filenames):

        X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])    # (bs, 128, 64, 1)
        Y_data = np.ones([self.batch_size, self.max_text_len])            # (bs, 10)
        input_length = np.zeros((self.batch_size, 1))                       # (bs, 1)
        label_length = np.zeros((self.batch_size, 1))           # (bs, 1)

        for  i, f  in  enumerate(batch_filenames):
            img, text = self.load_img_and_label(f)
            # print(text)
            if self.augment:
                img = self.augmentation(img, prob=0.75)


            # img = 255-img
            img = img.astype(np.float32)
            img = img / 255.0
            img = img.T
            img = np.expand_dims(img, -1)
            X_data[i] = img
            input_length[i] = self.img_w // self.downsample_factor - 2
            Y_data[i ,:len(text)] = text_to_labels(text)[:]
            label_length[i] = len(text)

        # Copy in dict form
        inputs = {
            'the_input': X_data,  # (bs, 128, 64, 1)
            'the_labels': Y_data,  # (bs, 10)
            'input_length': input_length,  # (bs, 1)
            'label_length': label_length  # (bs, 1) 
        }
        outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1) 
        return (inputs, outputs)

# train_file_path = 'E:/License Plate/CRNN/test/'

# gen = TextImageGenerator(train_file_path, img_w, img_h, 100, downsample_factor, shuffle=True,augment=True)

# ct=0
# for j in range(5):
#     for i,(img,lab) in enumerate(zip(gen[j][0]['the_input'],gen[j][0]['the_labels'])):
#         ct+=1
#         print(img.shape)

#         lab = labels_to_text(lab)
#         print(lab)

#         cv2.imshow(lab, img.squeeze().T)
#         cv2.waitKey(0)
#         cv2.imwrite('test/'+lab+'_1'+'.png', 255*img.squeeze().T)

#         if ct==100:
#             exit()
        



# print('------------------------------------------------------')

