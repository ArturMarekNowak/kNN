import math
import functools
import operator
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import time
import csv      
import pprint
import random
import contextlib
import sys, os
from sklearn import neighbors, datasets



def initHelper(features, chosenFeatures):
    retTup = []

    for ft in chosenFeatures:
        retTup.append(features[ft])

    return retTup


def getPoints(dim=3, mod=10000):
    pointsXD = {0: [], 1: []}
    chosenFeatures = [1, 6, 10, 13, 14, 15, 17, 19, 20, 21]
    chosenFeatures = chosenFeatures[:dim]

    # drive.mount('/content/drive/', force_remount=True)

    with open('\datasets\Malware dataset - Malware dataset.csv', 'r') as f:
        data = csv.reader(f, delimiter=",")

        itr = 1  # Skip first row with collumn names
        for row in data:
            if (itr % mod) == 0:
                if row[2] == 'malware':
                    rc = initHelper(row, chosenFeatures)
                    rc = [float(item) for item in rc]
                    pointsXD[1].append(rc)
                else:
                    rc = initHelper(row, chosenFeatures)
                    rc = [float(item) for item in rc]
                    pointsXD[0].append(rc)
            itr += 1

    print("success!!!!")
    # drive.flush_and_unmount()

    return pointsXD


points2D = {0: [[1, 12], [2, 5], [3, 6], [3, 10], [3.5, 8], [2, 11], [2, 9], [1, 7]],
            1: [[5, 3], [3, 2], [1.5, 9], [7, 2], [6, 1], [3.8, 1], [5.6, 4], [4, 2], [2, 5]]}

points3D = {0: [[1, 12, 4], [2, 5, 5], [3, 6, 5], [3, 10, 7], [3.5, 8, 7], [2, 11, 0], [2, 9, 8], [1, 7, 8]],
            1: [[5, 3, 3], [3, 2, 4], [1.5, 9, 4], [7, 2, 7], [6, 1, 4], [3.8, 1, 2], [5.6, 4, 1], [4, 2, 1],
                [2, 5, 8]]}

points4D = {
    0: [[1, 12, 4, 1], [2, 5, 5, 7], [3, 6, 5, 12], [3, 10, 7, 13], [3.5, 8, 7, 4.75], [2, 11, 0, 9.2], [2, 9, 8, 3],
        [1, 7, 8, 4]],
    1: [[5, 3, 3, 3], [3, 2, 4, 9], [1.5, 9, 4, 11], [7, 2, 7, 3.2], [6, 1, 4, 6], [3.8, 1, 2, 2], [5.6, 4, 1, 11],
        [4, 2, 1, 8], [2, 5, 8, 4]]}

points5D = {
    0: [[1, 12, 4, 1, 7], [2, 5, 5, 2, 7], [3, 6, 5, 9, 5], [3, 10, 7, 6, 3], [3.5, 8, 7, 2, 8], [2, 11, 0, 8, 2],
        [2, 9, 8, 1, 3], [1, 7, 8, 5, 3]],
    1: [[5, 3, 3, 1, 1], [3, 2, 4, 2, 2, ], [1.5, 9, 4, 3, 3], [7, 2, 7, 4, 4], [6, 1, 4, 5, 5], [3.8, 1, 2, 6, 6],
        [5.6, 4, 1, 7, 7], [4, 2, 1, 3, 5], [2, 5, 8, 12, 12]]}

pointsXD = getPoints()

p2D = [5, 3.5]
p3D = [1, 12, 4.5]
p4D = [3.5, 9, 7, 4.75]
p5D = [5, 3, 3, 1, 1.5]
p = [round(random.uniform(0, 1), 2) for i in range(len(pointsXD[0][0]))]

k1 = 1
k3 = 3
k4 = 5
k5 = 7


def normalization(pointsDict=pointsXD):
    # Simplest idea:
    maxValues = [0] * len(pointsDict[0][0])

    # Find max values in each column
    for key in points3D:
        for onePoint in pointsDict[key]:
            for feature in range(len(onePoint)):
                if maxValues[feature] < onePoint[feature]:
                    maxValues[feature] = onePoint[feature]

    # Normalize
    for key in pointsDict:
        for onePoint in pointsDict[key]:
            for feature in range(len(onePoint)):
                onePoint[feature] = (onePoint[feature] / maxValues[feature])
                onePoint[feature] = round(onePoint[feature], 2)

    return pointsDict


pprint.pprint(normalization())


def printNeighbours(distance, distance1, distance2, dim):
    print("Euclidean: ", end='')
    print(dim * "\t", end='')
    print("Mink/Cheb: ", end='')
    print(dim * "\t", end='')
    print("Chi2: ")

    for i in range(len(distance)):
        print(distance[i], end='')
        print("\t", end='')
        print(distance1[i], end='')
        print("\t", end='')
        print(distance2[i])


def distanceChiSquare(dataSet, pkt, dim):
    accumulator = 0
    for d in range(dim):
        accumulator += ((dataSet[d] - pkt[d]) ** 2) / (dataSet[d] + pkt[d])

    return 0.5 * accumulator


def distanceMinkovsky(dataSet, p, dim, power=10):
    accumulator = 0
    for d in range(dim):
        accumulator += (dataSet[d] - p[d]) ** power

    return accumulator ** (1 / float(power))


def distanceEuclidean(data_set, pkt, dim):
    accumulator = 0
    for d in range(dim):
        accumulator += (data_set[d] - pkt[d]) ** 2

    return math.sqrt(accumulator)


def result(distance):
    freq1 = 0
    freq2 = 0
    for d in distance:
        if d[2] == 0:
            freq1 += 1
        else:
            freq2 += 1

    return 0 if freq1 > freq2 else 1


def classifyAPoint(p, points=pointsXD, k=4, printNeighbouring="No", printAll="No"):
    dim = len(p)
    distanceEucl = []
    distanceMink = []
    distanceChi2 = []

    for group in points:
        for feature in points[group]:
            distanceE = distanceEuclidean(feature, p, dim)
            distanceE = round(distanceE, 2)
            distanceEucl.append((distanceE, feature, group))

            distanceM = distanceMinkovsky(feature, p, dim, 10)
            distanceM = round(distanceM, 2)
            distanceMink.append((distanceM, feature, group))

            distanceC = distanceChiSquare(feature, p, dim)
            distanceC = round(distanceC, 2)
            distanceChi2.append((distanceC, feature, group))

    if printAll == "Yes":
        print("Distance to all poits:")
        printNeighbours(distanceEucl, distanceMink, distanceChi2, dim)
        print(20 * "-")

    if printNeighbouring == "Yes":
        distanceEucl = sorted(distanceEucl)[:k]
        distanceMink = sorted(distanceMink)[:k]
        distanceChi2 = sorted(distanceChi2)[:k]

        print("Distance to k = %s nearest points: " % k)
        printNeighbours(distanceEucl, distanceMink, distanceChi2, dim)

    resultE = result(distanceEucl)
    resultM = result(distanceMink)
    resultC = result(distanceChi2)

    return "\n\nEuclid classified point %s as %d, Minkovsky classified as %d and Chi2 inventor classified as %d " % (
    p, resultE, resultM, resultC)


print(classifyAPoint(p, pointsXD, k5, "Yes", "Yes"))


def executionAnalysis(kEnd=50, mod=1000, dim=10):
    # Time vs k, fixed: dim = 10, points = every 1000
    kValues = [*range(1, kEnd + 1, 1)]
    execTimesK = []
    pointsXD = getPoints(dim, mod)
    p = [round(random.uniform(0, 1), 2) for i in range(len(pointsXD[0][0]))]

    for k in kValues:
        startTime = time.clock()
        classifyAPoint(p, pointsXD, k)
        timeX = time.clock() - startTime
        execTimesK.append(timeX)

    plt.subplot(131)
    plt.plot(kValues, execTimesK, 'o', color='red')

    # Dim vs Time, fixed: k = 7, points = every 1000
    dValues = [*range(1, dim + 1, 1)]
    execTimesDim = []
    dimensions = ["%dD" % (i + 1) for i in range(dim)]
    k = 7

    for d in dValues:
        pointsXD = getPoints(d, mod)
        p = [round(random.uniform(0, 1), 2) for i in range(len(pointsXD[0][0]))]

        start_time1 = time.clock()
        classifyAPoint(p, pointsXD, k)
        time1 = time.clock() - start_time1
        execTimesDim.append(time1)

    plt.subplot(132)
    plt.plot(dimensions, execTimesDim, 'o', color='green')

    # Points vs Time, fixed: k = 7, dim = 10
    mValues = [*range(0, 10000 + 11, 500)]
    mValues.pop(0)
    execTimesP = []
    k = 7

    for m in mValues:
        pointsXD = getPoints(dim, m)
        p = [round(random.uniform(0, 1), 2) for i in range(len(pointsXD[0][0]))]

        start_time1 = time.clock()
        classifyAPoint(p, pointsXD, k)
        time1 = time.clock() - start_time1
        execTimesP.append(time1)

    plt.subplot(133)
    plt.plot(mValues, execTimesP, 'o', color='blue')


executionAnalysis()

tempDict = getPoints(2, 10000)
tempDict = normalization(tempDict)

p = [0.9, 0.9]

rc = classifyAPoint(p, tempDict, 3)

x_0_2D=[tempDict[0][x][0] for x in range(len(tempDict[0]))]
x_1_2D=[tempDict[1][x][0] for x in range(len(tempDict[1]))]
y_0_2D=[tempDict[0][y][1] for y in range(len(tempDict[0]))]
y_1_2D=[tempDict[1][y][1] for y in range(len(tempDict[1]))]

if rc == 0:
    col = 'red'
elif rc == 1:
    col = 'green'

plt.subplot(121)
plt.plot(x_0_2D, y_0_2D, 'o', color='red')
plt.plot(x_1_2D, y_1_2D, 'o', color='green')
plt.plot(p[0], p[1],'o', color='blue' )
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.grid(True)


plt.subplot(122)
plt.plot(x_0_2D, y_0_2D, 'o', color='red')
plt.plot(x_1_2D, y_1_2D, 'o', color='green')
plt.plot(p[0], p[1],'o', color=col)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.grid(True)

plt.tight_layout(pad=3.0)

plt.show() 

#Comparison with scikitlearn

X = tempDict[0] + tempDict[1]
X = np.array([np.array(i) for i in X])
y = [0] * len(tempDict[0]) + [1] * len(tempDict[1])

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])


clf = neighbors.KNeighborsClassifier(3, weights = 'uniform')
clf.fit(X, y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("2-Class classification (k = %i, weights = '%s')"
              % (3, 'uniform'))

x_0_3D = []
x_1_3D = []
y_0_3D = []
y_1_3D = []
z_0_3D = []
z_1_3D = []

for onePoint in points3D:
    for feature in points3D[onePoint]:
        if (onePoint == 0):
            x_0_3D.append(feature[0])
            y_0_3D.append(feature[1])
            z_0_3D.append(feature[2])
        if (onePoint == 1):
            x_1_3D.append(feature[0])
            y_1_3D.append(feature[1])
            z_1_3D.append(feature[2])

if p3D == 0:
    col = 'red'
else:
    col = 'green'

fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.scatter3D(x_0_3D, y_0_3D, z_0_3D, c='red')
ax1.scatter3D(x_1_3D, y_1_3D, z_1_3D, c='green')
ax1.scatter3D(p3D[0], p3D[1], p3D[2], c='blue')

plt.show()

fig = plt.figure()
ax2 = plt.axes(projection='3d')
ax2.scatter3D(x_0_3D, y_0_3D, z_0_3D, c='red')
ax2.scatter3D(x_1_3D, y_1_3D, z_1_3D, c='green')
ax2.scatter3D(p3D[0], p3D[1], p3D[2], c=col)

plt.show()