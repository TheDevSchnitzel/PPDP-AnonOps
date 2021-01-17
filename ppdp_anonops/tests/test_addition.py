from unittest import TestCase
import os
from ppdp_anonops import Addition
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects.log.exporter.xes import factory as xes_exporter


class TestAddition(TestCase):
    def getTestXesLog(self):
        xesPath = os.path.join(os.path.dirname(__file__), 'resources', 'running_example.xes')
        return xes_importer.apply(xesPath)

    def getMatchLambda(self, attribute, value):  # Match last event in trace
        return (lambda c, e: len(c) > 0 and attribute in c[-1].keys() and c[-1][attribute] == value)

    def getTestEventTemplate(self):
        return [
            {'Name': 'concept:name', 'Value': 'Another Test'},
            {'Name': 'org:resource', 'Value': 'TestResource'},
            {'Name': 'Activity', 'Value': 'Just a test entry'},
            {'Name': 'Resource', 'Value': 'Test'},
            {'Name': 'Costs', 'Value': 123}
        ]

    def test_01_additionEventAtTraceEndConditioned(self):
        log = self.getTestXesLog()

        s = Addition()

        matchAttribute = "org:resource"
        matchAttributeValue = "Ellen"

        self.__checkLogProperties(log, noCases=6, noEvents=42)

        eventTemplate = self.getTestEventTemplate()
        log = s.AddEventLastInTrace(log, eventTemplate,  (lambda c, e: len(c) > 0 and matchAttribute in c[-1].keys() and c[-1][matchAttribute] == matchAttributeValue), doNotUseRandomlyGeneratedTimestamp=False)

        self.__checkTimestampContinuity(log)
        self.__checkLogProperties(log, noCases=6, noEvents=45)

        for case_index, case in enumerate(log):
            if(case[len(case) - 2][matchAttribute] == matchAttributeValue):
                self.assertTrue(case[len(case) - 1]['concept:name'] == "Another Test", "Wrong event name set")
            else:
                self.assertTrue(case[len(case) - 1][matchAttribute] != matchAttributeValue, "New event not added, though match was found")

    def test_02_additionEventAsFirstInTraceConditioned(self):
        log = self.getTestXesLog()

        s = Addition()

        matchAttribute = "org:resource"
        matchAttributeValue = "Ellen"

        self.__checkLogProperties(log, noCases=6, noEvents=42)

        eventTemplate = self.getTestEventTemplate()
        log = s.AddEventFirstInTrace(log, eventTemplate, (lambda c, e: len(c) > 0 and matchAttribute in c[-1].keys() and c[-1][matchAttribute] == matchAttributeValue), doNotUseRandomlyGeneratedTimestamp=False)

        self.__checkTimestampContinuity(log)
        self.__checkLogProperties(log, noCases=6, noEvents=45)

        for case_index, case in enumerate(log):
            if(case[len(case) - 1][matchAttribute] == matchAttributeValue):
                self.assertTrue(case[0]['concept:name'] == "Another Test", "Wrong event name set: " + case[0]['concept:name'])
            else:
                self.assertFalse(case[0]['concept:name'] == "Another Test", "New event added, though no match was found")

    def test_03_additionEventRandomPosition(self):
        log = self.getTestXesLog()

        s = Addition()

        no_traces = len(log)
        no_events = sum([len(trace) for trace in log])
        self.assertEqual(no_traces, 6)
        self.assertEqual(no_events, 42)

        eventTemplate = self.getTestEventTemplate()
        log = s.AddEventAtRandomPlaceInTrace(log, eventTemplate, doNotUseRandomlyGeneratedTimestamp=False)

        self.__checkTimestampContinuity(log)

        no_traces = len(log)
        no_events = sum([len(trace) for trace in log])
        self.assertEqual(no_traces, 6)
        self.assertEqual(no_events, 48)  # 42 original events + 1 for each case (6)

    def __checkTimestampContinuity(self, log):
        for case_index, case in enumerate(log):
            timestamp = None

            for event_index, event in enumerate(case):
                if(timestamp is None):
                    timestamp = event['time:timestamp']
                else:
                    if(timestamp > event['time:timestamp']):
                        self.fail("Timestamp out of sync")
                    else:
                        timestamp = event['time:timestamp']

    def __checkLogProperties(self, log, noCases, noEvents):
        no_traces = len(log)
        no_events = sum([len(trace) for trace in log])
        self.assertEqual(no_traces, noCases)
        self.assertEqual(no_events, noEvents)
