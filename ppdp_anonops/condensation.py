from .anonymizationOperation import AnonymizationOperation
import collections

# k-means
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from kmodes.kmodes import KModes
import numbers
from ppdp_anonops.utils import euclidClusterHelper
import statistics


class Condensation(AnonymizationOperation):

    def __init__(self):
        super(Condensation, self).__init__()

    def CondenseEventAttributeBykMeanCluster(self, xesLog, sensitiveAttribute, descriptiveAttributes, k_clusters, condensationFunction):
        allAttributes = descriptiveAttributes.copy()
        allAttributes.append(sensitiveAttribute)

        # Extract the values of descriptive and sensitive attributes from the event log, preserving their order of discovery (IMPORTANT!)
        values = self._getEventMultipleAttributeValues(xesLog, allAttributes)
        sensitiveValues = [x[-1] for x in values]
        descriptiveValues = [x[:-1] for x in values]

        values, valueToOneHotDict, oneHotToValueDict = euclidClusterHelper.oneHotEncodeNonNumericAttributes(allAttributes, values)
        descriptiveValuesEncoded = [x[:-1] for x in values]

        kmeans = KMeans(n_clusters=k_clusters)
        kmeans.fit(descriptiveValuesEncoded)

        # Get a dict with the sensitive attribute's value as key and the cluster it is assigned to as value
        # This is possible without using the descriptiveValuesEncoded, as all methods preserve the order of items passed to them, so no conversion between encoded and original is needed
        descrToClusterDict = self.valuesToCluster(kmeans.labels_, descriptiveValues)

        # Condense the data in the clusters
        clusterValueDict = self.__condenseClusterData(kmeans.labels_, sensitiveValues, k_clusters, condensationFunction)

        # Apply clustered data mode to log
        xesLog = self.__applyCondensedDataOnEvents(xesLog, allAttributes, descrToClusterDict, clusterValueDict)

        return self.AddExtension(xesLog, 'con', 'event', sensitiveAttribute)

    def CondenseCaseAttributeBykMeanCluster(self, xesLog, sensitiveAttribute, descriptiveAttributes, k_clusters, condensationFunction):
        allAttributes = descriptiveAttributes.copy()
        allAttributes.append(sensitiveAttribute)

        # Extract the values of descriptive and sensitive attributes from the event log, preserving their order of discovery (IMPORTANT!)
        values = self._getCaseMultipleAttributeValues(xesLog, allAttributes)
        sensitiveValues = [x[-1] for x in values]
        descriptiveValues = [x[:-1] for x in values]

        values, valueToOneHotDict, oneHotToValueDict = euclidClusterHelper.oneHotEncodeNonNumericAttributes(allAttributes, values)
        descriptiveValuesEncoded = [x[:-1] for x in values]

        kmeans = KMeans(n_clusters=k_clusters)
        kmeans.fit(descriptiveValuesEncoded)

        # Get a dict with the sensitive attribute's value as key and the cluster it is assigned to as value
        # This is possible without using the descriptiveValuesEncoded, as all methods preserve the order of items passed to them, so no conversion between encoded and original is needed
        descrToClusterDict = self.valuesToCluster(kmeans.labels_, descriptiveValues)

        # Condense the data in the clusters
        clusterValueDict = self.__condenseClusterData(kmeans.labels_, sensitiveValues, k_clusters, condensationFunction)

        # Apply clustered data mode to log
        xesLog = self.__applyCondensedDataOnCases(xesLog, allAttributes, descrToClusterDict, clusterValueDict)

        return self.AddExtension(xesLog, 'con', 'case', sensitiveAttribute)

    def CondenseEventAttributeBykModesCluster(self, xesLog, sensitiveAttribute, descriptiveAttributes, k_clusters, condensationFunction):
        # Make sure the sensitive attribute is last in line for later indexing
        allAttributes = descriptiveAttributes.copy()
        allAttributes.append(sensitiveAttribute)

        # Extract the values of descriptive and sensitive attributes from the event log, preserving their order of discovery (IMPORTANT!)
        values = self._getEventMultipleAttributeValues(xesLog, allAttributes)
        sensitiveValues = [x[-1] for x in values]
        descriptiveValues = [x[:-1] for x in values]

        km = KModes(n_clusters=k_clusters, init='random')
        clusters = km.fit_predict(descriptiveValues)

        # Get a dict with the value as key and the cluster it is assigned to as value
        descrToClusterDict = self.valuesToCluster(clusters, descriptiveValues)

        # Condense the data in the clusters
        clusterValueDict = self.__condenseClusterData(clusters, sensitiveValues, k_clusters, condensationFunction)

        # Apply clustered data mode to log
        xesLog = self.__applyCondensedDataOnEvents(xesLog, allAttributes, descrToClusterDict, clusterValueDict)

        return self.AddExtension(xesLog, 'con', 'event', sensitiveAttribute)

    def CondenseCaseAttributeBykModesCluster(self, xesLog, sensitiveAttribute, descriptiveAttributes, k_clusters, condensationFunction):
        # Make sure the sensitive attribute is last in line for later indexing
        allAttributes = descriptiveAttributes.copy()
        allAttributes.append(sensitiveAttribute)

        # Extract the values of descriptive and sensitive attributes from the event log, preserving their order of discovery (IMPORTANT!)
        values = self._getCaseMultipleAttributeValues(xesLog, allAttributes)
        sensitiveValues = [x[-1] for x in values]
        descriptiveValues = [x[:-1] for x in values]

        km = KModes(n_clusters=k_clusters, init='random')
        clusters = km.fit_predict(descriptiveValues)

        # Get a dict with the sensitive attribute's value as key and the cluster it is assigned to as value
        descrToClusterDict = self.valuesToCluster(clusters, descriptiveValues)

        # Condense the data in the clusters
        clusterValueDict = self.__condenseClusterData(clusters, sensitiveValues, k_clusters, condensationFunction)

        # Apply clustered data mode to log
        xesLog = self.__applyCondensedDataOnCases(xesLog, allAttributes, descrToClusterDict, clusterValueDict)

        return self.AddExtension(xesLog, 'con', 'case', sensitiveAttribute)

    def CondenseEventAttributeByEuclidianDistance(self, xesLog, sensitiveAttribute, descriptiveAttributes, weightDict, k_clusters, condensationFunction):
        attributes = descriptiveAttributes
        attributes.append(sensitiveAttribute)
        weights = [weightDict[a] for a in attributes]

        values = []
        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):
                eventValues = []
                for attr in attributes:
                    eventValues.append(event[attr])
                values.append(eventValues)

        cluster = euclidClusterHelper.euclidDistCluster_Fit(values, k_clusters, weights)

        clusterValueDict = {}
        if condensationFunction.lower() == "mode":
            clusterValueDict = self.__getModeOfSensitiveAttributePerCluster(cluster['labels'], values, k_clusters)
        elif condensationFunction.lower() == "mean":
            clusterValueDict = self.__getMeanOfSensitiveAttributePerCluster(cluster['labels'], values, k_clusters)
        elif condensationFunction.lower() == "median":
            clusterValueDict = self.__getMedianOfSensitiveAttributePerCluster(cluster['labels'], values, k_clusters)

        i = 0
        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):
                event[sensitiveAttribute] = clusterValueDict[cluster['labels'][i]]
                i = i + 1

        return self.AddExtension(xesLog, 'con', 'event', sensitiveAttribute)

    def CondenseCaseAttributeByEuclidianDistance(self, xesLog, sensitiveAttribute, descriptiveAttributes, weightDict, k_clusters, condensationFunction):
        # Move Unique-Event-Attributes up to trace attributes
        attributes = descriptiveAttributes
        attributes.append(sensitiveAttribute)
        weights = [weightDict[a] for a in attributes]

        values = []
        for case_index, case in enumerate(xesLog):
            caseValues = []
            for attr in attributes:

                # Check whether the attribute is a unique event attribute (Only occuring once and in the first event)
                if(attr not in case.attributes.keys()):
                    # Ensure the attribute exists, even if it is None
                    val = None

                    unique = True
                    for event_index, event in enumerate(case):
                        if ((event_index == 0 and attr not in event.keys()) or (event_index > 0 and attr in event.keys())):
                            unique = False

                    if(unique):
                        val = case[0][attr]
                    caseValues.append(val)
                else:
                    caseValues.append(case.attributes[attr])
            values.append(caseValues)

        cluster = euclidClusterHelper.euclidDistCluster_Fit(values, k_clusters, weights)
        clusterValueDict = {}
        if condensationFunction.lower() == "mode":
            clusterValueDict = self.__getModeOfSensitiveAttributePerCluster(cluster['labels'], values, k_clusters)
        elif condensationFunction.lower() == "mean":
            clusterValueDict = self.__getMeanOfSensitiveAttributePerCluster(cluster['labels'], values, k_clusters)
        elif condensationFunction.lower() == "median":
            clusterValueDict = self.__getMedianOfSensitiveAttributePerCluster(cluster['labels'], values, k_clusters)

        i = 0
        for case_index, case in enumerate(xesLog):
            case.attributes[sensitiveAttribute] = clusterValueDict[cluster['labels'][i]]
            i = i + 1

        return self.AddExtension(xesLog, 'con', 'case', sensitiveAttribute)

    def __getMode(self, valueList):
        if len(valueList) == 0:
            return 0

        s = {}
        for v in valueList:
            if v in s:
                s[v] += 1
            else:
                s[v] = 1

        # Sort dict by value
        s = {k: v for k, v in sorted(s.items(), key=lambda item: item[1])}
        return next(iter(s.keys()))

    def __getModeOfSensitiveAttributePerCluster(self, clusterLabels, values, k_clusters):
        clusterValues = {k: [] for k in range(k_clusters)}
        for i in range(len(clusterLabels)):
            clusterValues[clusterLabels[i]].append(values[i])

        modeDict = {k: 0 for k in range(k_clusters)}
        for k in range(k_clusters):
            modeDict[k] = self.__getMode(clusterValues[k])

        return modeDict

    def __getMedianOfSensitiveAttributePerCluster(self, clusterLabels, values, k_clusters):
        clusterValues = {k: [] for k in range(k_clusters)}
        for i in range(len(clusterLabels)):
            clusterValues[clusterLabels[i]].append(values[i])

        medianDict = {k: 0 for k in range(k_clusters)}
        for k in range(k_clusters):
            medianDict[k] = statistics.median(clusterValues[k])

        return medianDict

    def __getMeanOfSensitiveAttributePerCluster(self, clusterLabels, values, k_clusters):
        clusterValues = {k: [] for k in range(k_clusters)}
        for i in range(len(clusterLabels)):
            clusterValues[clusterLabels[i]].append(values[i])

        avgDict = {k: 0 for k in range(k_clusters)}
        for k in range(k_clusters):
            avgDict[k] = (sum(clusterValues[k]) * 1.0) / len(clusterValues[k])

        return avgDict

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

    def __condenseClusterData(self, clusters, sensitiveValues, k_clusters, condensationFunction):
        clusterValueDict = {}
        if condensationFunction.lower() == "mode":
            clusterValueDict = self.__getModeOfSensitiveAttributePerCluster(clusters, sensitiveValues, k_clusters)
        elif condensationFunction.lower() == "mean":
            clusterValueDict = self.__getMeanOfSensitiveAttributePerCluster(clusters, sensitiveValues, k_clusters)
        elif condensationFunction.lower() == "median":
            clusterValueDict = self.__getMedianOfSensitiveAttributePerCluster(clusters, sensitiveValues, k_clusters)
        return clusterValueDict

    def __applyCondensedDataOnCases(self, xesLog, allAttributes, descrToClusterDict, clusterValueDict):
        print(descrToClusterDict)
        print(clusterValueDict)

        # Apply clustered data mode to log
        for case_index, case in enumerate(xesLog):
            # Only proceed if the descriptive and sensitive attributes are present in the case
            if(all([a in case.attributes.keys() for a in allAttributes])):

                # Discriminating values of the descriptive attributes as tuple
                descriptiveTuple = tuple([case.attributes[d] for d in allAttributes[:-1]])

                # Cluster the sensitive attribute value belongs to, discriminated by the descriptive attributes
                cluster = descrToClusterDict[descriptiveTuple]

                # Assign new value to cluster
                case.attributes[allAttributes[-1]] = clusterValueDict[cluster]
        return xesLog

    def __applyCondensedDataOnEvents(self, xesLog, allAttributes, descrToClusterDict, clusterValueDict):
        print(descrToClusterDict)
        print(clusterValueDict)

        # Apply clustered data mode to log
        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):
                # Only proceed if the descriptive and sensitive attributes are present in the event
                if(all([a in event.keys() for a in allAttributes])):

                    # Discriminating values of the descriptive attributes as tuple
                    descriptiveTuple = tuple([event[d] for d in allAttributes[:-1]])

                    # Cluster the sensitive attribute value belongs to, discriminated by the descriptive attributes
                    cluster = descrToClusterDict[descriptiveTuple]

                    # Assign new value to cluster
                    event[allAttributes[-1]] = clusterValueDict[cluster]
        return xesLog
