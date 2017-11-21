import csv
import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D,Flatten,Dense,Lambda
from keras import optimizers
from keras import regularizers

BATCH_SIZE=128
BINS=25
BIN_RANGE=[-1.0,1.0]
EPOCHS=5
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.0001
L2_REGULARIZATION = 0.001
ANGLE_CORRECTION_FACTOR = 0.20

def load_driving_log(csv_path):
    '''
    Loads the driving data log(csv).
    Returns the line data as a string array.
    '''
    samples = []
    with open(csv_path) as csvfile:
        header_present = csv.Sniffer().has_header(csvfile.read(1024))
        csvfile.seek(0)  # back to first line
        reader = csv.reader(csvfile)
        if header_present:
            next(reader)  # skip the header   
        for line in reader:
            samples.append(line) 
    return samples

def cleanup_data(samples):
    '''
    Removes any data with speed = 0.
    Returns cleansed data array.
    '''
    cleansed_samples = []
    for sample in samples:
        if (float(sample[6]) != 0.0):# don't add zero speed frames
            cleansed_samples.append(sample)
    return cleansed_samples
 
def draw_angles_distribution(samples,bins,angle_range):
    '''
    Draws a bar chart showing the histogram of the passed in data.
    Returns the left edge for each bin (apart form the last one for which right edge is returned) 
    and the bin value. The no. of bin edges is 'bin' + 1.
    '''
    angles = []
    for sample in samples:
        angle = float(sample[3])
        angles.append(angle)
    plt.figure(figsize=(14,7))
    plt.ylabel('Count');
    plt.xlabel('Angle');
    bar_height_if_uniform_dist = len(samples)/bins
    plt.plot(angle_range,[bar_height_if_uniform_dist,bar_height_if_uniform_dist])
    plt.text(angle_range[0],bar_height_if_uniform_dist+50,'Uniform Distribution')
    plt.title('Angle Histogram')
    bin_values,bin_edges,_=plt.hist(angles,bins=bins,range=angle_range)
    plt.show() 
    return bin_edges,bin_values
        
def balance_dataset(samples,bin_edges,bin_values,bins):
    '''
    Removes data where:
        (i)  angle is = +- 1.0
        (ii) the bin size is greater than the average bin size
    Returns the balanced array of sample data.
    '''
    balanced_samples = []
    for sample in samples:
        angle = float(sample[3])
        if (angle == 1.0 or angle == -1.0): # Remove extreme angles
            continue
        # Total bin edges are = no. of bins + 1
        # Bin edges are the left most value of the bin range aprt from the last one which is the right most, 
        # hence check if less than
        potential_bins = np.where(bin_edges < angle)
        # if no bin found
        if (len(potential_bins[0]) == 0):
            # For catching cases where the angle is exactly -1 or +1
            potential_bins = np.where(bin_edges == angle)
            if (len(potential_bins[0]) == 0):
                raise Exception('No bin match found for angle:{}'.format(angle))
        matched_bin_index = np.max(potential_bins)
        matched_bin_value = bin_values[matched_bin_index]
        avg_bin_size = len(samples)/bins
        # Higher the %, the more that bin gets penalized
        keep_probability = 1 - ((matched_bin_value + 10*avg_bin_size)/len(samples)) 
        if (matched_bin_value > avg_bin_size):
            if (np.random.rand() < keep_probability):
                balanced_samples.append(sample)            
        else:
            balanced_samples.append(sample)
           
    return balanced_samples
           
def generator(samples,data_dir,batch_size=32):
    '''
    Generates a batch of images and angles.
    Reads-in the sample data and for each record, adds center,left & right images + corresponding angles
    Keep in mind that the returned batch is 3 X the passed in batch_size because for each record, 3 images are added.
    The benefit of using a generator is that the entire dataset doesn't need to be processed at the same time,
    rather only a subset is processed and fed to the model, which greatly helps when working with constrained memory.
    '''
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for line in batch_samples: 
                center_angle = float(line[3])
                angles.append(center_angle)
                
                left_angle = center_angle + ANGLE_CORRECTION_FACTOR
                angles.append(left_angle)
                               
                right_angle = center_angle - ANGLE_CORRECTION_FACTOR 
                angles.append(right_angle)
                
                center_img_path = data_dir + line[0]
                center_img = cv2.cvtColor(cv2.imread(center_img_path),cv2.COLOR_BGR2RGB)
                # Crop 70 pixels from top and 24 pixels from bottom, output = 66 x 320
                center_img = center_img[70:136,:]
                # Resize to 66 x 200 as required by nVidia architecture
                center_img = cv2.resize(center_img,(200,66),interpolation = cv2.INTER_AREA)
                images.append(center_img)
                
                left_img_path = data_dir + line[1]
                left_img = cv2.cvtColor(cv2.imread(left_img_path),cv2.COLOR_BGR2RGB)
                # Crop 70 pixels from top and 24 pixels from bottom, output = 66 x 320
                left_img = left_img[70:136,:]
                # Resize to 66 x 200 as required by nVidia architecture
                left_img = cv2.resize(left_img,(200,66),interpolation = cv2.INTER_AREA)
                images.append(left_img)
                
                right_img_path = data_dir + line[2]
                right_img = cv2.cvtColor(cv2.imread(right_img_path),cv2.COLOR_BGR2RGB)
                # Crop 70 pixels from top and 24 pixels from bottom, output = 66 x 320
                right_img = right_img[70:136,:]
                # Resize to 66 x 200 as required by nVidia architecture
                right_img = cv2.resize(right_img,(200,66),interpolation = cv2.INTER_AREA)
                images.append(right_img)

            X_train = np.array(images)
            y_train = np.array(angles)
            # Return processed images for this batch but remember the value of local variables for next iteration
            yield sklearn.utils.shuffle(X_train, y_train)


def nVidiaNet(train_generator,validation_generator,steps_per_epoch,validation_steps,save_model_dir):
    '''
    Impelments the nVidia CNN architecture (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).
    Returns the model history object + also saves the model as 'model.h5' in the current working directory.    
    '''
    
    nVidiaModel = Sequential()
    
    nVidiaModel.add(Lambda(lambda x:(x/255.0)-0.5,input_shape=(66,200,3)))
    print('Input shape:{}'.format(nVidiaModel.input_shape))
    print('Output shape - after normalization:{}'.format(nVidiaModel.output_shape))
    
    nVidiaModel.add(Convolution2D(24,(5,5),strides=(2,2),kernel_initializer='normal',kernel_regularizer=regularizers.l2(L2_REGULARIZATION),activation='elu'))
    print('Output shape - after first convolution:{}'.format(nVidiaModel.output_shape))   
    
    nVidiaModel.add(Convolution2D(36,(5,5),strides=(2,2),kernel_initializer='normal',kernel_regularizer=regularizers.l2(L2_REGULARIZATION),activation='elu'))
    print('Output shape - after second convolution:{}'.format(nVidiaModel.output_shape))
        
    nVidiaModel.add(Convolution2D(48,(5,5),strides=(2,2),kernel_initializer='normal',kernel_regularizer=regularizers.l2(L2_REGULARIZATION),activation='elu'))
    print('Output shape - after third convolution:{}'.format(nVidiaModel.output_shape))    
    
    nVidiaModel.add(Convolution2D(64,(3,3),strides=(1,1),kernel_initializer='normal',kernel_regularizer=regularizers.l2(L2_REGULARIZATION),activation='elu'))
    print('Output shape - after fourth convolution:{}'.format(nVidiaModel.output_shape))    
    
    nVidiaModel.add(Convolution2D(64,(3,3),strides=(1,1),kernel_initializer='normal',kernel_regularizer=regularizers.l2(L2_REGULARIZATION),activation='elu'))
    print('Output shape - after fifth convolution:{}'.format(nVidiaModel.output_shape))    
    
    nVidiaModel.add(Flatten())
    print('Output shape - after flattening:{}'.format(nVidiaModel.output_shape))
    nVidiaModel.add(Dense(100,kernel_initializer='normal',kernel_regularizer=regularizers.l2(L2_REGULARIZATION),activation='elu'))
    print('Output shape - after first dense:{}'.format(nVidiaModel.output_shape))
    nVidiaModel.add(Dense(50,kernel_initializer='normal',kernel_regularizer=regularizers.l2(L2_REGULARIZATION),activation='elu'))
    print('Output shape - after second dense:{}'.format(nVidiaModel.output_shape))
    nVidiaModel.add(Dense(10,kernel_initializer='normal',kernel_regularizer=regularizers.l2(L2_REGULARIZATION),activation='elu'))
    print('Output shape - after third dense:{}'.format(nVidiaModel.output_shape))
    nVidiaModel.add(Dense(1))
    print('Output shape - after fourth dense:{}'.format(nVidiaModel.output_shape))
    
    adam_optzr = optimizers.Adam(lr=LEARNING_RATE,decay=LEARNING_RATE_DECAY)
    nVidiaModel.compile(optimizer=adam_optzr,loss='mse',metrics = ['accuracy'])
    
    nVidiaModel_history = nVidiaModel.fit_generator(train_generator,
                                                  validation_data=validation_generator,
                                                  steps_per_epoch=steps_per_epoch,
                                                  validation_steps=validation_steps,
                                                  epochs=EPOCHS)
    dt = datetime.now()
    model_name_prefix = dt.strftime("%y-%m-%d-%H-%M")
    
    nVidiaModel.save(save_model_dir + model_name_prefix  + '-model.h5')
    
    # Write out the model params  
    model_params_file = open(save_model_dir + model_name_prefix + '-model-params.txt', 'w')                     
    model_params_file.write('EPOCHS >>> {}\n'.format(EPOCHS))
    model_params_file.write('BATCH SIZE >>> {}\n'.format(BATCH_SIZE))
    model_params_file.write('LEARNING RATE >>> {}\n'.format(LEARNING_RATE))
    model_params_file.write('LEARNING RATE DECAY >>> {}\n'.format(LEARNING_RATE_DECAY))
    model_params_file.write('ANGLE CORRECTION FACTOR >>> {}\n'.format(ANGLE_CORRECTION_FACTOR))
    model_params_file.write('BINS >>> {}\n'.format(BINS))
    model_params_file.write('BIN RANGE >>> {}\n'.format(BIN_RANGE))
    model_params_file.close()
    
    return nVidiaModel_history

def main():
    data_dir = 'C:/Users/Admin/Desktop/Behavioral Cloning/driving-data/'
    driving_log_filename = 'driving_log.csv'
    save_model_dir = './saved-models/'

    samples = load_driving_log(data_dir + driving_log_filename)
    print('Total samples:{}'.format(len(samples)))
    samples = cleanup_data(samples)
    print('Total samples after removing zero angles:{}'.format(len(samples)))
    bin_edges,bin_values = draw_angles_distribution(samples,BINS,BIN_RANGE)
    samples = balance_dataset(samples,bin_edges,bin_values,BINS)     
    _,_ = draw_angles_distribution(samples,BINS,BIN_RANGE)       
    train_samples,validation_samples = train_test_split(samples,test_size=0.2)
    
    # Set up the data generators
    train_generator = generator(train_samples,data_dir,batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples,data_dir,batch_size=BATCH_SIZE)
    
    # As we are adding the left & right images as well, so need x 3 times
    total_samples = len(samples) * 3
    actual_batch_size = BATCH_SIZE * 3
    len_train = len(train_samples) * 3
    len_valid = len(validation_samples) * 3
    steps_per_epoch = len_train/actual_batch_size
    validation_steps = len_valid/actual_batch_size
    print('Total number of images used for training & validation:{}'.format(total_samples))
    
    nVidiaModel_history = nVidiaNet(train_generator,validation_generator,steps_per_epoch,validation_steps,save_model_dir)
    plt.plot(nVidiaModel_history.history['loss'])
    plt.plot(nVidiaModel_history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()