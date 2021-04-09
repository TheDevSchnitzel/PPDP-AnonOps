from unittest import TestCase
import os
from ppdp_anonops import Condensation
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects.log.exporter.xes import factory as xes_exporter


class TestCondensation(TestCase):
    def getTestXesLog(self):
        xesPath = os.path.join(os.path.dirname(__file__), 'resources', 'Sepsis Cases - Event Log.xes')  # 'Sepsis Cases - Event Log.xes')
        return xes_importer.apply(xesPath)

    def test_01_eventLevelCondensation(self):
        # for clusters in range(4, 9):

        log = self.getTestXesLog()
        clusters = self.__getNumberOfDistinctEventAttributeValues(log, 'Diagnose')
        s = Condensation()
        # log = s.CondenseEventAttributeBykMeanCluster(log, 'Age', ["Age"], clusters, "mean")
        log = s.CondenseEventAttributeBykModesCluster(log, 'Age', ["Diagnose"], clusters, "mode")

        xes_exporter.export_log(log, "Sepsis_Condensed_" + str(clusters) + ".xes")
        # self.assertEqual(self.__getNumberOfDistinctEventAttributeValues(log, 'Diagnose'), clusters)
        raise Exception()

    def test_02_eee(self):
        log = xes_importer.apply(os.path.join(os.path.dirname(__file__), 'resources', 'running_exampleWithCostsAsInt.xes'))
        s = Condensation()

        # Needs to be a numeric attribute
        matchAttribute = "Costs"

        log = s.CondenseEventAttributeBykMeanCluster(log, matchAttribute, ["Activity", "Resource"], 4, "mean")

        self.assertEqual(self.__getNumberOfDistinctEventAttributeValues(log, matchAttribute), 4)

    def __getNumberOfDistinctEventAttributeValues(self, xesLog, attribute):
        values = []

        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):
                if(attribute in event.keys() and event[attribute] not in values):
                    values.append(event[attribute])
        print(values)
        return len(values)
