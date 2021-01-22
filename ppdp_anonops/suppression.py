from .anonymizationOperation import AnonymizationOperation
from pm4py.objects.log.importer.xes import factory as xes_importer_factory
import hashlib


class Suppression(AnonymizationOperation):
    """Replace a """

    def __init__(self):
        super(Suppression, self).__init__()

    def SuppressEvent(self, xesLog, conditional):
        for t_idx, trace in enumerate(xesLog):
            # filter out all the events with matching attribute values - matchAttribute "concept:name" at event level typically represents the performed activity
            trace[:] = [event for event in trace if (not conditional(trace, event))]
        return self.AddExtension(xesLog, 'sup', 'event', 'event')

    def SuppressEventAttribute(self, xesLog, suppressedAttribute, conditional=None):
        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):
                isMatch = conditional is None or conditional(case, event)

                if (isMatch):
                    # Only supress resource if activity value is a match
                    event[suppressedAttribute] = None
        return self.AddExtension(xesLog, 'sup', 'event', suppressedAttribute)

    def SuppressCase(self, xesLog, conditional):
        # filter out all the cases with matching attribute values
        xesLog[:] = [case for idx, case in enumerate(xesLog) if (not conditional(case, None))]

        return self.AddExtension(xesLog, 'sup', 'case', 'case')

    def SuppressCaseAttribute(self, xesLog, suppressedAttribute, conditional=None):
        for case_index, case in enumerate(xesLog):
            isMatch = conditional is None or conditional(case, None)

            if (isMatch):
                # Only supress resource if activity value is a match
                case.attributes[suppressedAttribute] = None
        return self.AddExtension(xesLog, 'sup', 'case', suppressedAttribute)
