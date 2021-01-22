from .anonymizationOperation import AnonymizationOperation
from .utils import TaxonomyTree


class Generalization(AnonymizationOperation):

    def __init__(self):
        super(Generalization, self).__init__()

    def GeneralizeEventAttributeByTaxonomyTreeDepth(self, xesLog, sensitiveAttribute, taxonomyTree, depth):
        depth = int(depth) + 1
        taxDict = taxonomyTree.GetPathDict_NodeNamesUntilLeaf()

        # Replace all attribute values below 'depth' in the taxTree with their generalized parental value
        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):
                if sensitiveAttribute in event.keys():
                    if event[sensitiveAttribute] in taxDict.keys():
                        idx = -depth
                        if(depth >= len(taxDict[event[sensitiveAttribute]])):
                            idx = 0

                        event[sensitiveAttribute] = taxDict[event[sensitiveAttribute]][idx]

        return self.AddExtension(xesLog, 'gen', 'event', sensitiveAttribute)

    def GeneralizeCaseAttributeByTaxonomyTreeDepth(self, xesLog, sensitiveAttribute, taxonomyTree, depth):
        depth = int(depth) + 1
        taxDict = taxonomyTree.GetPathDict_NodeNamesUntilLeaf()

        # Replace all attribute values below 'depth' in the taxTree with their generalized parental value
        for case_index, case in enumerate(xesLog):
            if sensitiveAttribute in case.attributes.keys():
                if case.attributes[sensitiveAttribute] in taxDict.keys():
                    idx = -depth
                    if(depth >= len(taxDict[case.attributes[sensitiveAttribute]])):
                        idx = 0

                    case.attributes[sensitiveAttribute] = taxDict[case.attributes[sensitiveAttribute]][idx]

        return self.AddExtension(xesLog, 'gen', 'case', sensitiveAttribute)

    def GeneralizeEventTimeAttribute(self, xesLog, dateTimeAttribute, generalizationLevel):
        level = generalizationLevel.lower()

        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):
                if(dateTimeAttribute in event.keys()):
                    event[dateTimeAttribute] = self.__GeneralizeTime(event[dateTimeAttribute], level)
        return self.AddExtension(xesLog, 'gen', 'event', dateTimeAttribute)

    def GeneralizeCaseTimeAttribute(self, xesLog, dateTimeAttribute, generalizationLevel):
        level = generalizationLevel.lower()

        for case_index, case in enumerate(xesLog):
            if(dateTimeAttribute in case.attributes.keys()):
                case.attributes[dateTimeAttribute] = self.__GeneralizeTime(case.attributes[dateTimeAttribute], level)
        return self.AddExtension(xesLog, 'gen', 'case', dateTimeAttribute)

    def __GeneralizeTime(self, time, level):
        val = time

        if(level == "seconds"):
            val = val.replace(microsecond=0)

        if(level == "minutes"):
            val = val.replace(microsecond=0, second=0)

        if(level == "hours"):
            val = val.replace(microsecond=0, second=0, minute=0)

        if(level == "days"):
            val = val.replace(microsecond=0, second=0, minute=0, hour=0)

        if(level == "months"):
            val = val.replace(microsecond=0, second=0, minute=0, hour=0, day=1)

        if(level == "years"):
            val = val.replace(microsecond=0, second=0, minute=0, hour=0, day=1, month=1)
        return val
