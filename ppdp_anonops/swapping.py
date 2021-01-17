from .anonymizationOperationInterface import AnonymizationOperationInterface
import collections
import random

# k-means
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import numbers
from kmodes.kmodes import KModes

from ppdp_anonops.utils import euclidClusterHelper


class Swapping(AnonymizationOperationInterface):
    def __init__(self):
        super(Swapping, self).__init__()

    def SwapEventAttributeValuesBykMeanCluster(self, xesLog, sensitiveAttribute, descriptiveAttributes, k_clusters):
        allAttributes = descriptiveAttributes.copy()
        allAttributes.append(sensitiveAttribute)

        values = self._getEventMultipleAttributeValues(xesLog, allAttributes, distinct=True)
        values, valueToOneHotDict, oneHotToValueDict = euclidClusterHelper.oneHotEncodeNonNumericAttributes(allAttributes, values)

        kmeans = KMeans(n_clusters=k_clusters)
        kmeans.fit(values)

        # Get a dict with the value as key and the cluster it is assigned to as value
        valueToClusterDict = euclidClusterHelper.getValuesOfSensitiveAttributePerClusterAsDict(kmeans.labels_, values)

        # If OneHot encoding was used: Ensure the mapping dicts are working with the original values, not the OneHot values
        clusterToValuesDict = {}
        if(sensitiveAttribute in valueToOneHotDict.keys()):
            clusterToValuesDict = {k: [oneHotToValueDict[sensitiveAttribute][x] for x in valueToClusterDict.keys() if valueToClusterDict[x] == k] for k in range(k_clusters)}
            valueToClusterDict = {oneHotToValueDict[sensitiveAttribute][x]: valueToClusterDict[x] for x in valueToClusterDict.keys()}
        else:
            clusterToValuesDict = {k: [x for x in valueToClusterDict.keys() if valueToClusterDict[x] == k] for k in range(k_clusters)}

        # Choose random new value from clustered data
        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):
                if(sensitiveAttribute in event.keys()):
                    # Get possible values from current values cluster
                    listOfValues = clusterToValuesDict[valueToClusterDict[event[sensitiveAttribute]]]

                    # Generate new random index
                    rnd = random.randint(0, len(listOfValues) - 1)

                    # Overwrite old attribute value with new one
                    event[sensitiveAttribute] = listOfValues[rnd]

        self.AddExtension(xesLog, 'swa', 'event', sensitiveAttribute)
        return xesLog

    def SwapCaseAttributeValuesBykMeanCluster(self, xesLog, sensitiveAttribute, descriptiveAttributes, k_clusters):
        allAttributes = descriptiveAttributes.copy()
        allAttributes.append(sensitiveAttribute)

        values = self._getCaseMultipleAttributeValues(xesLog, allAttributes, distinct=True)
        values, valueToOneHotDict, oneHotToValueDict = euclidClusterHelper.oneHotEncodeNonNumericAttributes(allAttributes, values)

        kmeans = KMeans(n_clusters=k_clusters)
        kmeans.fit(values)

        # Get a dict with the value as key and the cluster it is assigned to as value
        valueToClusterDict = euclidClusterHelper.getValuesOfSensitiveAttributePerClusterAsDict(kmeans.labels_, values)

        # If OneHot encoding was used: Ensure the mapping dicts are working with the original values, not the OneHot values
        clusterToValuesDict = {}
        if(sensitiveAttribute in valueToOneHotDict.keys()):
            clusterToValuesDict = {k: [oneHotToValueDict[sensitiveAttribute][x] for x in valueToClusterDict.keys() if valueToClusterDict[x] == k] for k in range(k_clusters)}
            valueToClusterDict = {oneHotToValueDict[sensitiveAttribute][x]: valueToClusterDict[x] for x in valueToClusterDict.keys()}
        else:
            clusterToValuesDict = {k: [x for x in valueToClusterDict.keys() if valueToClusterDict[x] == k] for k in range(k_clusters)}

        # Choose random new value from clustered data
        for case_index, case in enumerate(xesLog):
            if(sensitiveAttribute in case.attributes.keys()):
                # Get possible values from current values cluster
                listOfValues = clusterToValuesDict[valueToClusterDict[case.attributes[sensitiveAttribute]]]

                # Generate new random index
                rnd = random.randint(0, len(listOfValues) - 1)

                # Overwrite old attribute value with new one
                case.attributes[sensitiveAttribute] = listOfValues[rnd]

        self.AddExtension(xesLog, 'swa', 'case', sensitiveAttribute)
        return xesLog

    def SwapEventAttributeBykModesClusterUsingMode(self, xesLog, sensitiveAttribute, descriptiveAttributes, k_clusters):
        # Make sure the sensitive attribute is last in line for later indexing
        allAttributes = descriptiveAttributes.copy()
        allAttributes.append(sensitiveAttribute)

        values = self._getEventMultipleAttributeValues(xesLog, allAttributes, distinct=True)
        km = KModes(n_clusters=k_clusters, init='random')
        clusters = km.fit_predict(values)

        # Get a dict with the value as key and the cluster it is assigned to as value
        valueToClusterDict = euclidClusterHelper.getValuesOfSensitiveAttributePerClusterAsDict(clusters, values)
        clusterToValuesDict = {k: [x for x in valueToClusterDict.keys() if valueToClusterDict[x] == k] for k in range(k_clusters)}

        # Choose random new value from clustered data
        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):
                if(sensitiveAttribute in event.keys()):
                    # Get possible values from current values cluster
                    listOfValues = clusterToValuesDict[valueToClusterDict[event[sensitiveAttribute]]]

                    # Generate new random index
                    rnd = random.randint(0, len(listOfValues) - 1)

                    # Overwrite old attribute value with new one
                    event[sensitiveAttribute] = listOfValues[rnd]

        return self.AddExtension(xesLog, 'swa', 'event', sensitiveAttribute)

    def SwapCaseAttributeBykModesClusterUsingMode(self, xesLog, sensitiveAttribute, descriptiveAttributes, k_clusters):
        # Make sure the sensitive attribute is last in line for later indexing
        allAttributes = descriptiveAttributes.copy()
        allAttributes.append(sensitiveAttribute)

        values = self._getCaseMultipleAttributeValues(xesLog, allAttributes, distinct=True)
        km = KModes(n_clusters=k_clusters, init='random')
        clusters = km.fit_predict(values)

        # Get a dict with the value as key and the cluster it is assigned to as value
        valueToClusterDict = euclidClusterHelper.getValuesOfSensitiveAttributePerClusterAsDict(clusters, values)
        clusterToValuesDict = {k: [x for x in valueToClusterDict.keys() if valueToClusterDict[x] == k] for k in range(k_clusters)}

        # Choose random new value from clustered data
        for case_index, case in enumerate(xesLog):
            if(sensitiveAttribute in case.attributes.keys()):
                # Get possible values from current values cluster
                listOfValues = clusterToValuesDict[valueToClusterDict[case.attributes[sensitiveAttribute]]]

                # Generate new random index
                rnd = random.randint(0, len(listOfValues) - 1)

                # Overwrite old attribute value with new one
                case.attributes[sensitiveAttribute] = listOfValues[rnd]

        return self.AddExtension(xesLog, 'swa', 'case', sensitiveAttribute)

    def SwapEventAttributeByEuclidianDistance(self, xesLog, sensitiveAttribute, descriptiveAttributes, weightDict, k_clusters):
        attributes = descriptiveAttributes
        attributes.append(sensitiveAttribute)
        weights = [weightDict[a] for a in attributes]

        values = self._getEventMultipleAttributeValues(xesLog, attributes, distinct=True)
        cluster = euclidClusterHelper.euclidDistCluster_Fit(values, k_clusters, weights)

        clusterToValuesDict = {i: [] for i in range(len(cluster['labels']))}
        valueToClusterDict = {}
        for i in range(len(cluster['labels'])):
            if(values[i][-1] not in clusterToValuesDict[cluster['labels'][i]]):
                clusterToValuesDict[cluster['labels'][i]].append(values[i][-1])
            if(values[i][-1] not in valueToClusterDict):
                valueToClusterDict[values[i][-1]] = cluster['labels'][i]

        # Choose random new value from clustered data
        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):
                if(sensitiveAttribute in event.keys()):
                    # Get possible values from current values cluster
                    listOfValues = clusterToValuesDict[valueToClusterDict[event[sensitiveAttribute]]]

                    # Generate new random index
                    rnd = random.randint(0, len(listOfValues) - 1)

                    # Overwrite old attribute value with new one
                    event[sensitiveAttribute] = listOfValues[rnd]

        return self.AddExtension(xesLog, 'swa', 'event', sensitiveAttribute)

    def SwapCaseAttributeByEuclidianDistance(self, xesLog, sensitiveAttribute, descriptiveAttributes, weightDict, k_clusters):
        # Make sure the sensitive attribute is last in line for later indexing
        allAttributes = descriptiveAttributes.copy()
        allAttributes.append(sensitiveAttribute)

        weights = [weightDict[a] for a in allAttributes]

        values = self._getCaseMultipleAttributeValues(xesLog, allAttributes, distinct=True)
        cluster = euclidClusterHelper.euclidDistCluster_Fit(values, k_clusters, weights)

        clusterToValuesDict = {i: [] for i in range(len(cluster['labels']))}
        valueToClusterDict = {}
        for i in range(len(cluster['labels'])):
            if(values[i][-1] not in clusterToValuesDict[cluster['labels'][i]]):
                clusterToValuesDict[cluster['labels'][i]].append(values[i][-1])
            if(values[i][-1] not in valueToClusterDict):
                valueToClusterDict[values[i][-1]] = cluster['labels'][i]

        # Choose random new value from clustered data
        for case_index, case in enumerate(xesLog):
            if(sensitiveAttribute in case.attributes.keys()):
                # Get possible values from current values cluster
                listOfValues = clusterToValuesDict[valueToClusterDict[case.attributes[sensitiveAttribute]]]

                # Generate new random index
                rnd = random.randint(0, len(listOfValues) - 1)

                # Overwrite old attribute value with new one
                case.attributes[sensitiveAttribute] = listOfValues[rnd]

        return self.AddExtension(xesLog, 'swa', 'case', sensitiveAttribute)
