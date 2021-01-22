from .anonymizationOperation import AnonymizationOperation
import random


class Substitution(AnonymizationOperation):

    def __init__(self):
        super(Substitution, self).__init__()

    def SubstituteEventAttributeValue(self, xesLog, targetAttribute, sensitiveAttributeValues, substituteValues):
        insensitiveAttributeValues = substituteValues

        if(len(insensitiveAttributeValues) == 0 or (len(insensitiveAttributeValues) == 1 and insensitiveAttributeValues[0] == '')):
            insensitiveAttributeValues = []
            for case_index, case in enumerate(xesLog):
                for event_index, event in enumerate(case):
                    if (targetAttribute in event.keys() and event[targetAttribute] not in insensitiveAttributeValues and event[targetAttribute] not in sensitiveAttributeValues):
                        insensitiveAttributeValues.append(event[targetAttribute])

        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):
                if (targetAttribute in event.keys() and event[targetAttribute] in sensitiveAttributeValues):
                    event[targetAttribute] = insensitiveAttributeValues[random.randint(0, len(insensitiveAttributeValues) - 1)]

        return self.AddExtension(xesLog, 'sub', 'event', targetAttribute)

    def SubstituteCaseAttributeValue(self, xesLog, targetAttribute, sensitiveAttributeValues, substituteValues):
        insensitiveAttributeValues = substituteValues

        if(len(insensitiveAttributeValues) == 0 or (len(insensitiveAttributeValues) == 1 and insensitiveAttributeValues[0] == '')):
            insensitiveAttributeValues = []
            for case_index, case in enumerate(xesLog):
                if (targetAttribute in case.attributes.keys() and case.attributes[targetAttribute] not in insensitiveAttributeValues and case.attributes[targetAttribute] not in sensitiveAttributeValues):
                    insensitiveAttributeValues.append(case.attributes[targetAttribute])

        for case_index, case in enumerate(xesLog):
            if (targetAttribute in case.attributes.keys() and case.attributes[targetAttribute] in sensitiveAttributeValues):
                case.attributes[targetAttribute] = insensitiveAttributeValues[random.randint(0, len(insensitiveAttributeValues) - 1)]

        return self.AddExtension(xesLog, 'sub', 'case', targetAttribute)
