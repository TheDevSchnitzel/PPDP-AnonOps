from unittest import TestCase
import os
from ppdp_anonops import Swapping
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects.log.exporter.xes import factory as xes_exporter


class TestSwapping(TestCase):
    def getTestXesLog(self):
        xesPath = os.path.join(os.path.dirname(__file__), 'resources', 'Sepsis Cases - Event Log.xes')  # 'Sepsis Cases - Event Log.xes')
        return xes_importer.apply(xesPath)

    def test_01_EventLevelSwapping(self):
        # for clusters in range(4, 9):
        log = self.getTestXesLog()
        clusters = self.__getNumberOfDistinctEventAttributeValues(log, 'Diagnose')
        s = Swapping()
        log = s.SwapEventAttributeValuesBykMeanCluster(log, 'Age', ["Diagnose"], clusters)
        xes_exporter.export_log(log, "Sepsis_Swapped_" + str(clusters) + ".xes")
        raise Exception()

    def __getNumberOfDistinctEventAttributeValues(self, xesLog, attribute):
        values = []

        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):
                if(attribute in event.keys() and event[attribute] not in values):
                    values.append(event[attribute])

        return len(values)
