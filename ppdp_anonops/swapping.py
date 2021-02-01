from .anonymizationOperation import AnonymizationOperation
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


class Swapping(AnonymizationOperation):
    def __init__(self):
        super(Swapping, self).__init__()

    def SwapEventAttributeValuesBykMeanCluster(self, xesLog, sensitiveAttribute, descriptiveAttributes, k_clusters):
        allAttributes = descriptiveAttributes.copy()
        allAttributes.append(sensitiveAttribute)

        # Extrace the values of descriptive and sensitive attributes from the event log, preserving their order of discovery (IMPORTANT!)
        values = self._getEventMultipleAttributeValues(xesLog, allAttributes, distinct=True)
        sensitiveValues = [x[-1] for x in values]
        descriptiveValues = [x[:-1] for x in values]

        values, valueToOneHotDict, oneHotToValueDict = euclidClusterHelper.oneHotEncodeNonNumericAttributes(allAttributes, values)
        descriptiveValuesEncoded = [x[:-1] for x in values]

        kmeans = KMeans(n_clusters=k_clusters)
        kmeans.fit(descriptiveValuesEncoded)

        # Get a dict with the sensitive attribute's value as key and the cluster it is assigned to as value
        # This is possible without using the descriptiveValuesEncoded, as all methods preserve the order of items passed to them, so no conversion between encoded and original is needed
        descrToClusterDict = self.valuesToCluster(kmeans.labels_, descriptiveValues)

        # Returns a dictionary x[cluster] = [sensitive values in the cluster]
        clusterToSensitiveValues = self.clusterToValues(kmeans.labels_, sensitiveValues, k_clusters)

        # Choose random new value from clustered data
        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):

                # Only proceed if the descriptive and sensitive attributes are present in the event
                if(all([a in event.keys() for a in allAttributes])):

                    # Discriminating values of the descriptive attributes as tuple
                    descriptiveTuple = tuple([event[d] for d in descriptiveAttributes])

                    event[sensitiveAttribute] = self.__assignNewValue(descriptiveTuple, descrToClusterDict, clusterToSensitiveValues)

        self.AddExtension(xesLog, 'swa', 'event', sensitiveAttribute)
        return xesLog

    def SwapCaseAttributeValuesBykMeanCluster(self, xesLog, sensitiveAttribute, descriptiveAttributes, k_clusters):
        allAttributes = descriptiveAttributes.copy()
        allAttributes.append(sensitiveAttribute)

        # Extrace the values of descriptive and sensitive attributes from the event log, preserving their order of discovery (IMPORTANT!)
        values = self._getCaseMultipleAttributeValues(xesLog, allAttributes, distinct=True)
        sensitiveValues = [x[-1] for x in values]
        descriptiveValues = [x[:-1] for x in values]

        values, valueToOneHotDict, oneHotToValueDict = euclidClusterHelper.oneHotEncodeNonNumericAttributes(allAttributes, values)
        descriptiveValuesEncoded = [x[:-1] for x in values]

        kmeans = KMeans(n_clusters=k_clusters)
        kmeans.fit(descriptiveValuesEncoded)

        # Get a dict with the sensitive attribute's value as key and the cluster it is assigned to as value
        # This is possible without using the descriptiveValuesEncoded, as all methods preserve the order of items passed to them, so no conversion between encoded and original is needed
        descrToClusterDict = self.valuesToCluster(kmeans.labels_, descriptiveValues)

        # Returns a dictionary x[cluster] = [sensitive values in the cluster]
        clusterToSensitiveValues = self.clusterToValues(kmeans.labels_, sensitiveValues, k_clusters)

        # Choose random new value from clustered data
        for case_index, case in enumerate(xesLog):

            # Only proceed if the descriptive and sensitive attributes are present in the case
            if(all([a in case.attributes.keys() for a in allAttributes])):

                # Discriminating values of the descriptive attributes as tuple
                descriptiveTuple = tuple([case.attributes[d] for d in descriptiveAttributes])

                # Overwrite old attribute value with new one
                case.attributes[sensitiveAttribute] = self.__assignNewValue(descriptiveTuple, descrToClusterDict, clusterToSensitiveValues)

        self.AddExtension(xesLog, 'swa', 'case', sensitiveAttribute)
        return xesLog

    def SwapEventAttributeBykModesClusterUsingMode(self, xesLog, sensitiveAttribute, descriptiveAttributes, k_clusters):
        # Make sure the sensitive attribute is last in line for later indexing
        allAttributes = descriptiveAttributes.copy()
        allAttributes.append(sensitiveAttribute)

        # Extract the values of descriptive and sensitive attributes from the event log, preserving their order of discovery (IMPORTANT!)
        values = self._getEventMultipleAttributeValues(xesLog, allAttributes, distinct=True)
        sensitiveValues = [x[-1] for x in values]
        descriptiveValues = [x[:-1] for x in values]

        km = KModes(n_clusters=k_clusters, init='random')
        clusters = km.fit_predict(descriptiveValues)

        # Get a dict with the value as key and the cluster it is assigned to as value
        descrToClusterDict = self.valuesToCluster(clusters, descriptiveValues)

        # Returns a dictionary x[cluster] = [sensitive values in the cluster]
        clusterToValuesDict = self.clusterToValues(clusters, sensitiveValues, k_clusters)

        # Choose random new value from clustered data
        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):

                # Only proceed if the descriptive and sensitive attributes are present in the event
                if(all([a in event.keys() for a in allAttributes])):

                    # Discriminating values of the descriptive attributes as tuple
                    descriptiveTuple = tuple([event[d] for d in descriptiveAttributes])

                    event[sensitiveAttribute] = self.__assignNewValue(descriptiveTuple, descrToClusterDict, clusterToValuesDict)

        return self.AddExtension(xesLog, 'swa', 'event', sensitiveAttribute)

    def SwapCaseAttributeBykModesClusterUsingMode(self, xesLog, sensitiveAttribute, descriptiveAttributes, k_clusters):
        # Make sure the sensitive attribute is last in line for later indexing
        allAttributes = descriptiveAttributes.copy()
        allAttributes.append(sensitiveAttribute)

        # Extract the values of descriptive and sensitive attributes from the event log, preserving their order of discovery (IMPORTANT!)
        values = self._getCaseMultipleAttributeValues(xesLog, allAttributes, distinct=True)
        sensitiveValues = [x[-1] for x in values]
        descriptiveValues = [x[:-1] for x in values]

        km = KModes(n_clusters=k_clusters, init='random')
        clusters = km.fit_predict(descriptiveValues)

        # Get a dict with the value as key and the cluster it is assigned to as value
        descrToClusterDict = self.valuesToCluster(clusters, descriptiveValues)

        # Returns a dictionary x[cluster] = [sensitive values in the cluster]
        clusterToValuesDict = self.clusterToValues(clusters, sensitiveValues, k_clusters)

        # Choose random new value from clustered data
        for case_index, case in enumerate(xesLog):

            # Only proceed if the descriptive and sensitive attributes are present in the case
            if(all([a in case.attributes.keys() for a in allAttributes])):

                # Discriminating values of the descriptive attributes as tuple
                descriptiveTuple = tuple([case.attributes[d] for d in descriptiveAttributes])

                # Overwrite old attribute value with new one
                case.attributes[sensitiveAttribute] = self.__assignNewValue(descriptiveTuple, descrToClusterDict, clusterToValuesDict)

        return self.AddExtension(xesLog, 'swa', 'case', sensitiveAttribute)

    def SwapEventAttributeByEuclidianDistance(self, xesLog, sensitiveAttribute, descriptiveAttributes, weightDict, k_clusters):
        allAttributes = descriptiveAttributes.copy()
        allAttributes.append(sensitiveAttribute)
        weights = [weightDict[a] for a in allAttributes]

        values = self._getEventMultipleAttributeValues(xesLog, allAttributes, distinct=True)

        # Get sensitive values here, before oneHotEncoding changes the value array
        sensitiveValues = [x[-1] for x in values]

        # Clustering over all attributes (including sensitive one) is valid, as the sensitive one is weighted as well
        cluster = euclidClusterHelper.euclidDistCluster_Fit(values, k_clusters, weights)

        # Get a dict with the value as key and the cluster it is assigned to as value (Including the sensitive attribute, as is is used to discriminate as well)
        descrToClusterDict = self.valuesToCluster(cluster['labels'], values)

        # Returns a dictionary x[cluster] = [sensitive values in the cluster]
        clusterToValuesDict = self.clusterToValues(cluster['labels'], sensitiveValues, k_clusters)

        # Choose random new value from clustered data
        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):

                # Only proceed if the descriptive and sensitive attributes are present in the event
                if(all([a in event.keys() for a in allAttributes])):

                    # Discriminating values of the descriptive attributes (including the sensitive one!) as tuple
                    descriptiveTuple = tuple([event[d] for d in allAttributes])

                    event[sensitiveAttribute] = self.__assignNewValue(descriptiveTuple, descrToClusterDict, clusterToValuesDict)

        return self.AddExtension(xesLog, 'swa', 'event', sensitiveAttribute)

    def SwapCaseAttributeByEuclidianDistance(self, xesLog, sensitiveAttribute, descriptiveAttributes, weightDict, k_clusters):
        # Make sure the sensitive attribute is last in line for later indexing
        allAttributes = descriptiveAttributes.copy()
        allAttributes.append(sensitiveAttribute)
        weights = [weightDict[a] for a in allAttributes]

        values = self._getCaseMultipleAttributeValues(xesLog, allAttributes, distinct=True)

        # Clustering over all attributes (including sensitive one) is valid, as the sensitive one is weighted as well
        cluster = euclidClusterHelper.euclidDistCluster_Fit(values, k_clusters, weights)

        # Get a dict with the value as key and the cluster it is assigned to as value (Including the sensitive attribute, as is is used to discriminate as well)
        descrToClusterDict = self.valuesToCluster(cluster['labels'], values)

        # Returns a dictionary x[cluster] = [sensitive values in the cluster]
        sensitiveValues = [x[-1] for x in values]
        clusterToValuesDict = self.clusterToValues(cluster['labels'], sensitiveValues, k_clusters)

        # Choose random new value from clustered data
        for case_index, case in enumerate(xesLog):
            # Only proceed if the descriptive and sensitive attributes are present in the case
            if(all([a in case.attributes.keys() for a in allAttributes])):

                # Discriminating values of the descriptive attributes (including the sensitive one!) as tuple
                descriptiveTuple = tuple([case.attributes[d] for d in allAttributes])

                # Overwrite old attribute value with new one
                case.attributes[sensitiveAttribute] = self.__assignNewValue(descriptiveTuple, descrToClusterDict, clusterToValuesDict)

        return self.AddExtension(xesLog, 'swa', 'case', sensitiveAttribute)

    def valuesToCluster(self, clusterLabels, values):
        valueToClusterDict = {}
        for i in range(len(clusterLabels)):
            # [-1] as the sensitive attribute value is always the last in the list
            if tuple(values[i]) not in valueToClusterDict.keys():
                valueToClusterDict[tuple(values[i])] = clusterLabels[i]
        return valueToClusterDict

    def clusterToValues(self, clusterLabels, values, k_clusters):
        clusterToValueDict = {k: [] for k in range(k_clusters)}
        for i in range(len(clusterLabels)):
            # [-1] as the sensitive attribute value is always the last in the list
            if values[i] not in clusterToValueDict[clusterLabels[i]]:
                clusterToValueDict[clusterLabels[i]].append(values[i])
        return clusterToValueDict

    def __assignNewValue(self, descriptiveTuple, descrToClusterDict, clusterToSensitiveValues):
        # Cluster the sensitive attribute value belongs to, discriminated by the descriptive attributes
        cluster = descrToClusterDict[descriptiveTuple]

        # Get possible values from current values cluster
        listOfValues = clusterToSensitiveValues[cluster]

        # Generate new random index
        rnd = random.randint(0, len(listOfValues) - 1)

        # Overwrite old attribute value with new one
        return listOfValues[rnd]
