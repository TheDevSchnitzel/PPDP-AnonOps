
from math import sqrt
import random
import numpy as np
import numbers

# As by some error some case attributes get safed in the first event of a trace -> Lift these up to become actual case attributes


def liftUniqueEventAttributesToCase(log, attributes):
    for attr in attributes:
        for case_index, case in enumerate(log):
            # Check whether the attribute is a unique event attribute (Only occuring once and in the first event)
            if(attr not in case.attributes.keys()):
                # Ensure the attribute exists, even if it is None
                case.attributes[attr] = None

                move = True
                for event_index, event in enumerate(case):
                    if ((event_index == 0 and attr not in event.keys()) or (event_index > 0 and attr in event.keys())):
                        move = False

                if(move):
                    case.attributes[attr] = case[0][attr]
    return log


def euclidDistCluster_Fit(values, k_clusters, weights, verboose=0):
    # Randomly select k cases to be the centroids of our clustering
    clusterCentroids = []
    try:
        for i in random.sample(range(0, len(values)), k_clusters):
            clusterCentroids.append(values[i])
    except ValueError:
        raise ValueError("Choose a suitable amount of clusters: k < " + str(len(values)))

    # When a selected attribute is not of a numeric type => One-Hot encode it
    oneHotEncodedDict = {}
    for i in range(len(values)):
        for j in range(len(values[i])):
            if not isinstance(values[i][j], numbers.Number) and j not in oneHotEncodedDict:
                oneHotEncodedDict[j] = {'Values': [], 'OneHotEncoded': [], 'OneHotNormalized': [], 'ValueToIndex': {}}

    for j in oneHotEncodedDict.keys():
        for i in range(len(values)):
            if values[i][j] not in oneHotEncodedDict[j]['Values']:
                oneHotEncodedDict[j]['Values'].append(values[i][j])
                oneHotEncodedDict[j]['ValueToIndex'][values[i][j]] = len(oneHotEncodedDict[j]['Values']) - 1

        for k in range(0, len(oneHotEncodedDict[j]['Values'])):
            # Create OneHot-Tuple
            t = [0, ]*len(oneHotEncodedDict[j]['Values'])
            t[k] = 1
            oneHotEncodedDict[j]['OneHotEncoded'].append(tuple(t))
            oneHotEncodedDict[j]['OneHotNormalized'].append(((k * 1.0) / len(oneHotEncodedDict[j]['Values'])))

    caseClusters = []
    for i in range(len(values)):
        val = [(oneHotEncodedDict[j]['OneHotNormalized'][oneHotEncodedDict[j]['ValueToIndex'][values[i][j]]] if j in oneHotEncodedDict.keys() else values[i][j]) for j in range(len(values[i]))]

        centroidDistance = []
        for centroid in clusterCentroids:
            centroidVals = [(oneHotEncodedDict[j]['OneHotNormalized'][oneHotEncodedDict[j]['ValueToIndex'][centroid[j]]] if j in oneHotEncodedDict.keys() else centroid[j]) for j in range(len(values[i]))]
            centroidDistance.append(euclidianDistance(weights, val, centroidVals))
        caseClusters.append(np.argmin(centroidDistance))

    return {"labels": caseClusters, "categories": [clusterCentroids[i][-1] for i in range(len(clusterCentroids))]}


def euclidianDistance(weights, attributesA, attributesB):
    if(len(weights) != len(attributesA) != len(attributesB)):
        raise NotImplementedError("This feature is only available for input arrays of identical length")

    sum = 0

    for i in range(len(weights)):
        sum += weights[i] * ((attributesA[i] - attributesB[i]) ** 2)

        #print(str(weights[i]) + " * " + "((" + str(attributesA[i]) + " - " + str(attributesB[i]) + ") ** 2)) = " + str(weights[i] * ((attributesA[i] - attributesB[i]) ** 2)))
    return sqrt(sum)


def oneHotEncodeNonNumericAttributes(namedAttributes, instanceAttribute2DArray):
    """
    This function takes a 2-dimensional array of instances and their attributes [[Attribute 1, Attribute 2], [Attribute 1, Attribute 2]]
    and checks, whether an attribute is non numerical.

    If a non numerical attribute is discovered it will get OneHot encoded over all available values of that attribute
    The OneHot-Vectors will then get normalized to a value in the interval [0, 1] by the mapping i-th vector to i/N for N-Vector dimensions

    This function returns an array in the same structure as it was given, just with categorical values replaces by OneHot-Values
    Secondly, this function returns a dictionary to reverse the OneHot encoding: oneHotInversionDict[ATTRIBUTE][ONEHOT-VALUE] = ORIGINAL-ATTRIBUTE
    """
    values = instanceAttribute2DArray
    attributes = len(values[0])
    performOneHot = []
    oneHotToValueDict = {}
    valueToOneHotDict = {}

    for i in range(attributes):
        performOneHot.append(not isinstance(values[0][i], numbers.Number))

    # When a selected attribute is not of a numeric type => One-Hot encode it
    oneHotEncodedDict = {}
    for i in range(attributes):
        if performOneHot[i]:
            oneHotEncodedDict[i] = {'Values': [], 'OneHotEncoded': [], 'OneHotNormalized': [], 'ValueToIndex': {}}
            oneHotToValueDict[namedAttributes[i]] = {}
            valueToOneHotDict[namedAttributes[i]] = {}

            for j in range(len(values)):
                if values[j][i] not in oneHotEncodedDict[i]['Values']:
                    oneHotEncodedDict[i]['Values'].append(values[j][i])
                    oneHotEncodedDict[i]['ValueToIndex'][values[j][i]] = len(oneHotEncodedDict[i]['Values']) - 1

            for k in range(0, len(oneHotEncodedDict[i]['Values'])):
                # Create OneHot-Tuple
                t = [0, ]*len(oneHotEncodedDict[i]['Values'])
                t[k] = 1
                oneHotEncodedDict[i]['OneHotEncoded'].append(tuple(t))

                oneHotValue = ((k * 1.0) / len(oneHotEncodedDict[i]['Values']))
                oneHotEncodedDict[i]['OneHotNormalized'].append(oneHotValue)
                oneHotToValueDict[namedAttributes[i]][oneHotValue] = oneHotEncodedDict[i]['Values'][k]
                valueToOneHotDict[namedAttributes[i]][oneHotEncodedDict[i]['Values'][k]] = oneHotValue

            for j in range(len(values)):
                idx = oneHotEncodedDict[i]['ValueToIndex'][values[j][i]]
                #values[j][i] = oneHotEncodedDict[i]['OneHotEncoded'][idx]
                values[j][i] = oneHotEncodedDict[i]['OneHotNormalized'][idx]

    return values, valueToOneHotDict, oneHotToValueDict

# Make sure all values provided are actually numeric


def checkNumericAttributes(values):
    numCheck = [x for x in values if not isinstance(x, numbers.Number)]
    if(len(numCheck) > 0):
        raise NotImplementedError("Use a numeric attribute")
    pass


def getValuesOfSensitiveAttributePerClusterAsDict(clusterLabels, values):
    valueToClusterDict = {}
    for i in range(len(clusterLabels)):
        # [-1] as the sensitive attribute value is always the last in the list
        if values[i][-1] not in valueToClusterDict.keys():
            valueToClusterDict[values[i][-1]] = clusterLabels[i]
    return valueToClusterDict
