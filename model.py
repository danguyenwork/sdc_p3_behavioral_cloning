import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn
import cv2
import datetime

TRAIN_PATH_CSV = '/home/adnguyen1989/Documents/SDC_P3/data/train.csv'
IMG_PATH = '/home/adnguyen1989/Documents/SDC_P3/data/IMG/'
RECOVERY_PATH_CSV = '/home/adnguyen1989/Downloads/simulator-linux/recovery.csv'

ANGLE_THROTTLE_ADJUSTMENT = .25

Y_total = []

def load_data():
    df = pd.read_csv(TRAIN_PATH_CSV, names=['center_img','left_img','right_img','steering_angle','throttle','break','speed'])

    # recovery = pd.read_csv(RECOVERY_PATH_CSV,names=['center_img','left_img','right_img','steering_angle','throttle','break','speed'])
    # recovery = recovery[recovery.steering_angle < -.2]
    #
    #
    #
    # df = pd.concat([df, recovery])

    df['center_img'] = df['center_img'].map(lambda x: x.strip())

    X = df.center_img.values
    y = df.steering_angle.values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = .2, shuffle=True)

    return X_train, X_valid, y_train, y_valid

def _transform_resize(x):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(x, (66, 200))

def _transform_center(x):
    return (x / 255.0) - 0.5

def _transform_rotation(x, y):
    return np.fliplr(x), -y

def _transform_brightness(x,y):
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    #Generate new random brightness
    rand = min(np.random.rand()+.1, 1)
    hsv[:,:,2] = rand*hsv[:,:,2]
    #Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img, y

def _transform_shadow(image, y):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = 320 * np.random.rand(), 0
    x2, y2 = 320 * np.random.rand(), 160
    xm, ym = np.mgrid[0:160, 0:320]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB), y

def _adjust_steering_angle_for_side_camera(steering_angle, camera_angle):
    if camera_angle == 'left':
        return steering_angle + ANGLE_THROTTLE_ADJUSTMENT
    elif camera_angle == 'right':
        return steering_angle - ANGLE_THROTTLE_ADJUSTMENT
    else:
        return steering_angle

def _select_random_angle(random_index, features, labels, train):
    name = (IMG_PATH + features[random_index].split("/")[-1])
    if not train:
        x = mpimg.imread(name)
        y = float(labels[random_index])
    else:
        random_angle = np.random.choice(['center', 'left', 'right'])
        x = mpimg.imread(name.replace('center', random_angle))
        y = _adjust_steering_angle_for_side_camera(float(labels[random_index]), random_angle)
    return x, y


def _generator(features, labels, hyperparams, train=True):
    batch_size = hyperparams['batch_size']
    flip_prob = hyperparams['flip_prob']
    darken_prob = hyperparams['darken_prob']
    drop_low_angle_prob = hyperparams['drop_low_angle_prob']
    shadow_prob = hyperparams['shadow_prob']

    num_batch = features.shape[0] // batch_size + 1
    while 1: # Loop forever so the generator never terminates
        shuffle(features, labels)

        for _ in range(num_batch):
            batch_X = []
            batch_y = []

            while len(batch_X) < batch_size:
                random_index = np.random.randint(0, features.shape[0])

                y = float(labels[random_index])

                x, y = _select_random_angle(random_index, features, labels, train)

                if train and abs(y) <.1 and np.random.rand() < drop_low_angle_prob:
                    continue

                if np.random.rand() < flip_prob and train:
                    x, y = _transform_rotation(x,y)

                if np.random.rand() < darken_prob and train:
                    x, y = _transform_brightness(x,y)

                if np.random.rand() < shadow_prob and train:
                    x, y = _transform_shadow(x, y)

                batch_X.append(x)
                batch_y.append(y)
                Y_total.append(y)

                if np.random.rand() < .001:
                    tag = 'train' if train else 'validation'
                    mpimg.imsave('img/' + tag + '_example_' + str(random_index)+'.jpg', x)

            batch_X = np.array(batch_X)
            batch_y = np.array(batch_y)
            yield shuffle(batch_X, batch_y)

def run_model(choice, X_train_img, y_train, X_test_img, y_test, hyperparams):

    activation = hyperparams['activation']
    nb_epoch = hyperparams['epoch']
    samples_per_epoch = hyperparams['samples_per_epoch']
    keep_prob = hyperparams['keep_prob']
    learning_rate = hyperparams['learning_rate']

    model = Sequential()
    model.add(Cropping2D(cropping=((40,14), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(_transform_resize))
    model.add(Lambda(_transform_center))

    # ===

    if choice == 'simple':
        model.add(Flatten(input_shape=(160,320,3)))
        model.add(Dense(1))
    elif choice == 'nvidia':
        model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation=activation))
        model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation=activation))
        model.add(Convolution2D(48, 5, 5, subsample = (2,2), activation=activation))
        model.add(Convolution2D(64, 3, 3, activation=activation))
        model.add(Convolution2D(64, 3, 3, activation=activation))
        model.add(Flatten())
        model.add(Dense(100, activation=activation))
        # model.add(Dropout(keep_prob))
        model.add(Dense(50, activation=activation))
        # model.add(Dropout(keep_prob))
        model.add(Dense(10, activation=activation))
        # model.add(Dropout(keep_prob))
        model.add(Dense(1))
    # ===

    train_generator = _generator(X_train_img, y_train, hyperparams)
    validation_generator = _generator(X_test_img, y_test, hyperparams)

    optimizer = Adam(lr=learning_rate)

    model.compile(loss='mse', optimizer=optimizer)
    model.fit_generator(train_generator, samples_per_epoch= samples_per_epoch, validation_data=validation_generator , nb_val_samples=X_test_img.shape[0], nb_epoch=nb_epoch)
    return model

X_train_img, X_test_img, y_train, y_test = load_data()

# keep_prob = [.6, .5]
# samples_per_epoch = [9000, X_train_img.shape[0]]
# epoch = [10, 3]
# batch_size = [64, 128]
# activation = ['relu', 'elu']
# flip_prob = [.4, .5, .6]
# flip_prob = [.4, .5, .6]
# learning_rate = [.001]

hyperparams = {}
hyperparams['keep_prob'] = .7
hyperparams['epoch'] = 20
hyperparams['batch_size'] = 256
hyperparams['samples_per_epoch'] = 10000 // hyperparams['batch_size'] * hyperparams['batch_size']
hyperparams['activation'] = 'relu'
hyperparams['flip_prob'] = .5
hyperparams['darken_prob'] = .5
hyperparams['learning_rate'] = .001
hyperparams['drop_low_angle_prob'] = .4
hyperparams['shadow_prob'] = .2

model = run_model('nvidia', X_train_img, y_train, X_test_img, y_test, hyperparams)
model.save('model.h5')
filename = 'model_' + str(datetime.datetime.now()) + '.json'
with open('model.json', 'w') as outfile:
    outfile.write(model.to_json())


plt.hist(Y_total)
