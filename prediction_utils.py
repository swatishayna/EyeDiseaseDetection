import tensorflow as tf
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import IPython.display as display
import numpy as np
import glob
import random
import cv2
import shutil


def get_model_path():
    model_path = os.path.join(os.getcwd(), 'main_model', 'mainmodel')
    return model_path


def get_disease():

    diseases = {'abnormal pigment ': 'O', 'age-related macular degeneration': 'A',
                'anterior segment image': 'DELETE', 'arteriosclerosis': 'O',
                'asteroid hyalosis': 'O', 'atrophic change': 'O', 'atrophy': 'O',
                'branch retinal artery occlusion': 'O', 'branch retinal vein occlusion': 'O',
                'cataract': 'C', 'central retinal artery occlusion': 'O', 'central retinal vein occlusion': 'O',
                'central serous chorioretinopathy': 'O', 'chorioretinal atrophy': 'O',
                'chorioretinal atrophy with pigmentation proliferation': 'O',
                'choroidal nevus': 'NaN', 'congenital choroidal coloboma': 'O',
                'depigmentation of the retinal pigment epithelium': 'O',
                'diabetic retinopathy': 'D', 'diffuse chorioretinal atrophy': 'O', 'diffuse retinal atrophy': 'O',
                'drusen': 'O',
                'dry age-related macular degeneration': 'A', 'epiretinal membrane': 'O',
                'epiretinal membrane over the macula': 'O',
                'fundus laser photocoagulation spots': 'O', 'glaucoma': 'G',
                'glial remnants anterior to the optic disc': 'O',
                'hypertensive retinopathy': 'H', 'hypertensive retinopathy,diabetic retinopathy': 'D',
                'idiopathic choroidal neovascularization': 'O', 'image offset': 'DELETE',
                'intraretinal hemorrhage': 'O',
                'intraretinal microvascular abnormality': 'O', 'laser spot': 'O', 'lens dust': 'DELETE',
                'low image quality': 'DELETE', 'low image quality,maculopathy': 'DELETE', 'macular coloboma': 'O',
                'macular epiretinal membrane': 'O', 'macular hole': 'O', 'macular pigmentation disorder': 'NaN',
                'maculopathy': 'O', 'mild nonproliferative retinopathy': 'D',
                'moderate non proliferative retinopathy': 'D',
                'morning glory syndrome': 'O', 'myelinated nerve fibers': 'O', 'myopia retinopathy': 'M',
                'myopic maculopathy': 'M',
                'myopic retinopathy': 'M', 'no fundus image': 'DELETE', 'normal fundus': 'N',
                'old branch retinal vein occlusion': 'O',
                'old central retinal vein occlusion': 'O', 'old chorioretinopathy': 'O', 'old choroiditis': 'O',
                'optic disc edema': 'O',
                'optic discitis': 'O', 'optic disk epiretinal membrane': 'O',
                'optic disk photographically invisible': 'DELETE',
                'optic nerve atrophy': 'O', 'oval yellow-white atrophy': 'O', 'pathological myopia': 'M',
                'peripapillary atrophy': 'O',
                'pigment epithelium proliferation': 'O', 'pigmentation disorder': 'O',
                'post laser photocoagulation': 'O', 'post retinal laser surgery': 'O',
                'proliferative diabetic retinopathy': 'D',
                'punctate inner choroidopathy': 'O',
                'refractive media opacity': 'O', 'retina fold': 'O', 'retinal artery macroaneurysm': 'O',
                'retinal detachment': 'O',
                'retinal pigment epithelial hypertrophy': 'O', 'retinal pigment epithelium atrophy': 'O',
                'retinal pigmentation': 'O',
                'retinal vascular sheathing': 'O', 'retinitis pigmentosa': 'O',
                'retinochoroidal coloboma': 'O', 'rhegmatogenous retinal detachment': 'O',
                'severe nonproliferative retinopathy': 'D',
                'severe proliferative diabetic retinopathy': 'D', 'silicone oil eye': 'O',
                'spotted membranous change': 'O',
                'suspected abnormal color of  optic disc': 'O', 'suspected cataract': 'C',
                'suspected diabetic retinopathy': 'D', 'suspected glaucoma': 'G',
                'suspected macular epimacular membrane': 'O',
                'suspected microvascular anomalies': 'O', 'suspected moderate non proliferative retinopathy': 'D',
                'suspected retinal vascular sheathing': 'O',
                'suspected retinitis pigmentosa': 'O', 'suspicious diabetic retinopathy': 'D',
                'tessellated fundus': 'O', 'vascular loops': 'O', 'vessel tortuosity': 'O',
                'vitreous degeneration': 'O', 'vitreous opacity': 'O',
                'wedge white line change': 'O', 'wedge-shaped change': 'O', 'wet age-related macular degeneration': 'A',
                'white vessel': 'O'}
    return diseases


def get_classes():
    CLASS_NAMES = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    return CLASS_NAMES


def loadAndCropCenterResizeCV2(img, newSize):
    width, height, ______ = img.shape
    if width == height:
        return cv2.resize(img, newSize)
    length = min(width, height)
    left = (width - length) // 2
    top = (height - length) // 2
    right = (width + length) // 2
    bottom = (height + length) // 2
    return cv2.resize(img[left:right, top:bottom, :], newSize)


def image_resize(impath, dst_dir):
    img = cv2.imread(impath)
    eq_image = loadAndCropCenterResizeCV2(img, (250, 250))
    image_name = os.path.basename(impath)
    dst_path = os.path.join(dst_dir, image_name)
    cv2.imwrite(dst_path, eq_image)


def pair_image_resize(image1, image2):
    dst_dir = os.path.join(os.getcwd(), "test_dir")

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)

    os.mkdir(dst_dir)
    image_resize(image1, dst_dir)
    image_resize(image2, dst_dir)


def decode_img(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def merge_two(left, right):
    final_table = [0, 0, 0, 0, 0, 0, 0, 0]
    counter = 0
    not_n = 0
    for l in left:
        r = right[counter]
        if l >= r:
            final_table[counter] = l
        else:
            final_table[counter] = r
        if (l > 0.5 or r > 0.5) and counter > 0:
            print(l, r, counter)
            not_n = 1
        counter += 1

    if not_n == 1:
        final_table[0] = 0
    return final_table


def preprocess_data():
    test_images_file = os.path.join(os.getcwd(), "test_dir")
    test_ds = tf.data.Dataset.list_files(test_images_file + "/*.jpg", shuffle=False)

    # tf.data builds a performance model of the input pipeline and runs an optimization algorithm to find a good
    # allocation of its CPU budget across all parameters specified as AUTOTUNE. While the input pipeline is running,
    # tf.data tracks
    # the time spent in each operation, so that these times can be fed into the optimization algorithm.

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    test_ds2 = test_ds.map(decode_img, num_parallel_calls=AUTOTUNE)

    # Combines consecutive elements of this dataset into batches.1000 elements into 32 batches
    test_ds3 = test_ds2.batch(2)

    return test_ds3


def get_prediction():
    model_path = get_model_path()
    model = tf.keras.models.load_model(model_path)

    # data for prediction
    test_ds3 = preprocess_data()
    predictions = model.predict(test_ds3)

    diseases = get_disease()
    CLASS_NAMES = get_classes()

    print(type(diseases))
    init = 1000
    counter = 0
    for element in predictions:
        if counter % 2 == 0:
            final_table = merge_two(predictions[counter], predictions[counter + 1])
            id_value = np.argmax(final_table)
            print(id_value)
            predicted_class = {i for i in diseases if diseases[i] == CLASS_NAMES[id_value]}
            init += 1
        counter += 1
    return predicted_class

