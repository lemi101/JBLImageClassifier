from PIL import Image
import colorsys
import sys
import numpy
import glob
import cv2
import os


def get_filepath():
    return os.path.dirname(os.path.realpath(__file__))

def main():
    print(get_filepath())

if __name__ == "__main__":
    main()

'''
def hsv_color(img):
    if isinstance(img, Image.Image):
        r, g, b = img.split()
        h_result = []
        s_result = []
        v_result = []
        for rd, gn, bl in zip(r.getdata(), g.getdata(), b.getdata()):
            h, s, v = colorsys.rgb_to_hsv(rd / 255., gn / 255., bl / 255.)
            h_result.append(int(h * 255.))
            s_result.append(int(s * 255.))
            v_result.append(int(v * 255.))
        r.putdata(h_result)
        g.putdata(s_result)
        b.putdata(v_result)
        return Image.merge('RGB', (r, g, b))
    else:
        return None


def normalize_hist(hist, pixel_size):
    for v in hist:
        v = int(v / pixel_size)

    return hist


def load_image_from_dir_to_hist(path):
    img = Image.open(path)
    img = hsv_color(img)
    img = numpy.array(img)

    pixel_size = img.size

    hist_h = normalize_hist(cv2.calcHist([img], [0], None, [256], [0, 256]), pixel_size)
    hist_s = normalize_hist(cv2.calcHist([img], [1], None, [256], [0, 256]), pixel_size)
    hist_v = normalize_hist(cv2.calcHist([img], [2], None, [256], [0, 256]), pixel_size)

    print("%s %s %s" % (hist_h, hist_s, hist_v))

    return hist_h, hist_s, hist_v


def load_images_from_dir_to_hist(path):
    hist_list = []
    for filename in glob.glob(path + '/*.jpg'):
        img = Image.open(filename)
        img = hsv_color(img)
        img = numpy.array(img)

        pixel_size = img.size

        hist_h = normalize_hist(cv2.calcHist([img], [0], None, [256], [0, 256]), pixel_size)
        hist_s = normalize_hist(cv2.calcHist([img], [1], None, [256], [0, 256]), pixel_size)
        hist_v = normalize_hist(cv2.calcHist([img], [2], None, [256], [0, 256]), pixel_size)
        hist_list.append((hist_h, hist_s, hist_v))

    return hist_list


base_path = 'C:/Users/jmsrsd/PycharmProjects/JBLImageClassifier/Daun jeruk'

cendawanjelaga_list = load_images_from_dir_to_hist(base_path + '/cendawanjelaga')
cvpd_list = load_images_from_dir_to_hist(base_path + '/cvpd')
kekuranganzn_list = load_images_from_dir_to_hist(base_path + '/kekuranganzn')
mildew_list = load_images_from_dir_to_hist(base_path + '/mildew')
sehat_list = load_images_from_dir_to_hist(base_path + '/sehat')

class_label_list = ['cendawan jelaga', 'cvpd', 'kekurangan zn', 'mildew', 'sehat']

dataset = [cendawanjelaga_list, cvpd_list, kekuranganzn_list, mildew_list, sehat_list];


def knn(test):
    column = len(dataset[0])
    row = len(dataset)

    dataset_linear = []
    distance_list = []

    for i, r in enumerate(dataset):
        for j, c in enumerate(dataset[i]):
            dataset_linear.append(dataset[i][j])

    for i, item in enumerate(dataset_linear):
        distance = abs(test[0] - dataset_linear[i][0]) + abs(test[1] - dataset_linear[i][1]) + abs(
            test[2] - dataset_linear[i][2])
        distance_list.append(distance)

    print(distance_list)

    k = 3
    nearest_index = [-1, -1, -1]

    while k > 0:
        min_index = 0
        min_value = 256
        for i, item in enumerate(distance_list):
            if min_value >= item and i != nearest_index[0] and i != nearest_index[1] and i != nearest_index[2]:
                min_index = i
                min_value = item

        k -= 1
        nearest_index[k] = min_index

    class_score = [0, 0, 0]

    for i in range(len(class_score)):
        for j in range(len(class_score)):
            if int(nearest_index[i] / column) == int(nearest_index[j] / column):
                class_score[i] += 1

    max_index = 0
    max_value = 0
    for i in range(len(class_score)):
        if max_value <= class_score[i]:
            max_index = i
            max_value = class_score[i]

    return class_label_list[int(nearest_index[max_index] / column)]


if len(sys.argv) > 1:
    knn(load_image_from_dir_to_hist(sys.argv[1]))
'''
