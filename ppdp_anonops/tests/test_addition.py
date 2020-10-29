from unittest import TestCase
import os
from ppdp_anonops.addition import addition


class TestAddition(TestCase):
    def getTestXesPath(self):
        return os.path.join(os.path.dirname(__file__), 'resources', 'running_example.xes')

    def test_additionEventAtTraceEnd(self):
        s = addition(self.getTestXesPath())

        matchAttribute = "org:resource"
        matchAttributeValue = "Ellen"

        no_traces = len(s.xesLog)
        no_events = sum([len(trace) for trace in s.xesLog])
        self.assertEqual(no_traces, 6)
        self.assertEqual(no_events, 42)

        s.addEvent(matchAttribute, matchAttributeValue)

        no_traces = len(s.xesLog)
        no_events = sum([len(trace) for trace in s.xesLog])
        self.assertEqual(no_traces, 6)
        self.assertEqual(no_events, 45)
