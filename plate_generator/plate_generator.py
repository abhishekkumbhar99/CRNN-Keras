from itertools import permutations,combinations
import random
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
import pandas as pd 

plates = []
for i in range(1,4):
    plates.append(cv2.imread(f"plate_templates/indian_number_plate_{i}.jpg"))

save_dir = 'synthetic_plates/'
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
numbers = '123456789'
state_codes = ['AP','TG','AR','AS','BR','CG','GA','GJ','HR','HP','JK','JH','KA','KL',
'MP','MH','MN','ML','MZ','NL','OR','OD','PB','RJ','SK','TN','TR','UK','UP','WB','AN',
'CH','DN','DD','DL','LD','PY']
district_codes = ['01','02','03','04','05','06','07','08','09','10']


fonts = pd.read_csv('font_config.csv')
print(fonts.head(5))

for i in range(len(fonts)):

    font = fonts.iloc[i,:]
    
    # Added more images of this particular configuration as its more common
    if font['font_name']=='LicensePlate' and font['spaces']==False:
        num_images = 3
    else:
        num_images = 1

    for i in range(num_images):
        plate = plates[font['template']]
        # Convert to PIL Image
        cv2_im_rgb = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)

        draw = ImageDraw.Draw(pil_im, mode='RGB')
        # read the font TrueTypefile
        ft = ImageFont.truetype('fonts/' + font['font_name'] + '.ttf', font['size'])
        # Form a random 10 character number plate according to indian format
        # choose a random state code
        g1 = random.choice(state_codes)
        # choose a random district code
        g2 = random.choice(district_codes)
        #choose a random sequence of 2 letters
        g3 = ''.join(random.choice(list(permutations(letters, r=2))))
        #choose a random sequence of 4 numbers
        g4 = ''.join(random.choice(list(permutations(numbers, r=4))))

        if font['spaces']:
            word = g1+' '+g2+' '+g3+' '+g4
        else:
            word = g1+g2+g3+g4

        print(word)

        (x,y) = map(int,font['coord'].split(','))

        # Draw the text
        draw.text((x,y), word, font=ft, fill = (0,0,0))

        # Save the image
        cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_dir + word + "_" + ".png", cv2_im_processed)

        # cv2.imshow('.', cv2_im_processed)
        # cv2.waitKey(0)
        # exit()