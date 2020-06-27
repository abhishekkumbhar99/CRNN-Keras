from keras import backend as K
from keras.optimizers import Adadelta, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau, TensorBoard
from Image_Generator import TextImageGenerator
from  model  import  get_Model
from parameters import *

def main():
	# # Model description and training

	model = get_Model(training=True)

	try:
	    model.load_weights('')
	    print("...Previous weight data...")
	except:
	    print("...New weight data...")
	    pass

	train_file_path = 'E:/License Plate/CRNN/synthetic_plates/train/'
	train_gen = TextImageGenerator(train_file_path, img_w, img_h, batch_size, downsample_factor,shuffle=True,augment=False)
	print('created train generator')

	valid_file_path = 'E:/License Plate/CRNN/synthetic_plates/valid/'
	val_gen = TextImageGenerator(valid_file_path, img_w, img_h, val_batch_size, downsample_factor,shuffle=False,augment=False)
	print('created valid generator')

	# ada  =  Adadelta ()

	early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
	checkpoint = ModelCheckpoint(filepath='CRNN--{epoch:02d}--{val_loss:.4f}--{val_acc:.4f}.hdf5', monitor='loss', verbose=1, mode='min', period=1)
	lr_reduce = ReduceLROnPlateau(factor=0.5, patience=2, verbose=1, min_lr=1e-6)
	tensorboard = TensorBoard(log_dir='logs/')

	# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(1e-3), metrics=['acc'])

	print('starting training')
	# captures output of softmax so we can decode the output during visualization
	model.fit_generator(generator=train_gen,
	                    steps_per_epoch=1250,
	                    epochs=20,
	                    callbacks=[checkpoint, early_stop, lr_reduce, tensorboard],
	                    # use_multiprocessing=True,
	                    # workers = 4,
	                    validation_data=val_gen,
	                    validation_steps=250
	                    )


#always use main function when using Multiprocessing in your program
#using multiprocessing doesnt affect performance
if __name__=='__main__':
	main()