import os

import imutils
import math

import numpy as np
from cv2 import cv2

from matplotlib import pyplot as plt

import Helper


def imageBGR2FeatureVector(image, size=(32, 32)):
    return cv2.resize(image, size).flatten()


def imageBGR2FlatHist(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)

    return hist.flatten()


def getDataset():
    datasetFeatureVectors = {}
    datasetHists = {}

    for root, directories, filenames in os.walk(os.path.join(Helper.getMainDirectoryPath(), 'dataset')):
        for directory in directories:
            for datasetRoot, datasetDirectories, datasetFilenames in os.walk(os.path.join(root, directory)):
                imageFeatureVectors = []
                imageHists = []

                for filename in datasetFilenames:
                    imagePath = os.path.join(datasetRoot, filename)

                    imageBGR = cv2.imread(imagePath)

                    imageFeatureVector = imageBGR2FeatureVector(imageBGR)
                    imageFeatureVectors.append(imageFeatureVector)

                    imageHist = imageBGR2FlatHist(imageBGR)
                    imageHists.append(imageHist)

                datasetFeatureVectors.update({directory: imageFeatureVectors})
                datasetHists.update({directory: imageHists})

    return datasetFeatureVectors, datasetHists


def classifyImageByHist(imagePath, datasetHists, showStep=False):
    imageName = os.path.basename(imagePath)

    imageBGR = cv2.imread(imagePath)

    imageFeatureVector = imageBGR2FeatureVector(imageBGR)
    imageHist = imageBGR2FlatHist(imageBGR)

    result = []
    resultLastIndex = 0

    if showStep:
        for key, item in datasetHists.items():
            for i in range(len(datasetHists[key])):
                imageHistFormatted = []
                print("\n" + key + "[" + str(i) + "]: ")
                for j in range(len(datasetHists[key][i])):
                    imageHistFormatted.append('{0:.5f}'.format(datasetHists[key][i][j]))

                for n in range(len(imageHistFormatted)):
                    # if n % 8 == 0:
                    # print('')

                    print(imageHistFormatted[n], end=' ')

                    if n == len(imageHistFormatted) - 1:
                        print()

        print("\nuji:")
        for n in range(len(imageHist)):
            print('{0:.5f}'.format(imageHist[n]), end=' ')

            if n == len(imageHist) - 1:
                print()

        for key, item in datasetHists.items():
            for i in range(len(datasetHists[key])):
                imageHistFormatted = []
                print("\nkomparasi uji dengan " + key + "[" + str(i) + "]: ")
                for j in range(len(datasetHists[key][i])):
                    buffer = min(imageHist[j], datasetHists[key][i][j])

                    imageHistFormatted.append('{0:.5f}'.format(buffer))

                for n in range(len(imageHistFormatted)):
                    print(imageHistFormatted[n], end=' ')

                    if n == len(imageHistFormatted) - 1:
                        print()

                imageHistFormatted = np.array(imageHistFormatted).astype(np.float);
                result.append((key, sum(imageHistFormatted)))

                # print("hasil sum: %f" % (result[resultLastIndex][1]))
                resultLastIndex += 1
    else:
        for key, value in datasetHists.items():
            for i in range(6):
                imageHistDataset = value[i]

                histValue = '{0:.5f}'.format(cv2.compareHist(imageHist, imageHistDataset, cv2.HISTCMP_INTERSECT))
                result.append((key, float(histValue)))

                # print(str(key) + "[" + str(i) + "]: " + str(result[resultLastIndex][1]))
                resultLastIndex += 1

    print("[ K O M P A R A S I   G A M B A R   %s   D E N G A N   D A T A S E T ]" % (imageName))
    result.sort(key=lambda tup: tup[1], reverse=True)
    for r in result:
        print(r)

    classes = []
    for key, value in datasetHists.items():
        classes.append((key, 0))

    k = 5
    print()
    print("[ P E M B E R I A N   R A N K I N G   D E N G A N   K   =   %d ]" % (k))
    for i in range(len(classes)):
        for j in range(k):
            if result[j][0] == classes[i][0]:
                classes[i] = (classes[i][0], classes[i][1] + 1)

    classes.sort(key=lambda tup: tup[1], reverse=True)
    for c in classes:
        print(c)

    print()
    print("[ H A S I L   K L A S I F I K A S I ]")
    print("Kelas gambar %s adalah %s" % (imageName, classes[0][0]))
    print("________________________________________________________________________________\n")


def main():
    datasetFeatureVectors, datasetHists = getDataset()
    print(Helper.getMainDirectoryPath())

    datatest = ['C:/Users/jmsrsd/PycharmProjects/JBLImageClassifier/dataset/sehat37-5.jpg',
                'C:/Users/jmsrsd/PycharmProjects/JBLImageClassifier/dataset/sehat38-5.jpg',
                'C:/Users/jmsrsd/PycharmProjects/JBLImageClassifier/dataset/sehat39-5.jpg',
                'C:/Users/jmsrsd/PycharmProjects/JBLImageClassifier/dataset/sehat40-5.jpg']

    for i in range(len(datatest)):
        classifyImageByHist(datatest[i], datasetHists,
                            showStep=False)


if __name__ == '__main__':
    main()
