import cv2
import itertools, os, time
import  numpy  as  np
from  model  import  get_Model
from parameters import *
import  argparse
from keras import backend as K
K.set_learning_phase(0)


def decode_label(out):
    # out : (1, 32, 42)
    out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    outstr = ''
    for  i  in  out_best :
        if i < len(letters):
            outstr += letters[i]
    return outstr


parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weight", help="weight file directory",
                    type=str, default="CRNN--17--0.0173--0.9746.hdf5")
parser.add_argument("-t", "--test_img", help="Test image directory",
                    type=str, default="E:/License Plate/CRNN/test/")
args = parser.parse_args()

# Get CRNN model
model = get_Model(training=False)

try:
    model.load_weights(args.weight)
    print("...Previous weight data...")
except:
    raise Exception("No weight file!")


test_dir =args.test_img
test_imgs = os.listdir(args.test_img)
total = 0
acc = 0
letter_total = 0
letter_acc = 0
start = time.time()
for  test_img  in  test_imgs :
    img = cv2.imread(test_dir + test_img, cv2.IMREAD_GRAYSCALE)
    #Resize the image to reqd height keeping the aspect ratio unchanged
    img = cv2.resize(img, (int(img_h * (img.shape[1]/img.shape[0])), img_h))
    
    #pad the image if width is less than img_w
    if img.shape[1]<img_w:
        new_img = np.zeros((img_h,img_w), dtype=np.uint8)
        new_img[:img.shape[0], :img.shape[1]] = img[:, :]
        img = new_img
    elif img.shape[1]>img_w:
            img = cv2.resize(img, (img_w, img_h))

    img_name = test_img[0:-4].split('_')[0]
    img_name = "".join(img_name.split())

    img = 255-img
    img_pred = img.astype(np.float32)
    img_pred  = img_pred / 255.0
    img_pred = img_pred.T
    img_pred = np.expand_dims(img_pred, axis=-1)

    net_out_value = model.predict(np.array([img_pred]))

    pred_texts = decode_label(net_out_value)

    for  i  in  range ( min ( len ( pred_texts ), len ( img_name))):
        if pred_texts[i] == img_name[i]:
            letter_acc += 1
    letter_total  +=  max(len(pred_texts), len(img_name))

    if pred_texts == img_name:
        acc += 1
    total += 1
    print('Predicted: %s  /  True: %s' % (pred_texts, img_name))
    
    # cv2.rectangle(img, (0,0), (150, 30), (0,255,0), -1)
    # cv2.putText(img, pred_texts, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

    # cv2.imshow(pred_texts, img)
    # if cv2.waitKey(0) == 27:
    #   break
    # cv2.destroyAllWindows()

end = time.time()
total_time = (end - start)
print("Time : ",total_time / total)
print("ACC : ", acc / total)
print("letter ACC : ", letter_acc / letter_total)