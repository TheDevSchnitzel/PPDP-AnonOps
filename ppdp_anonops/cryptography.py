from .anonymizationOperation import AnonymizationOperation
from pm4py.objects.log.importer.xes import factory as xes_importer_factory
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
from base64 import b64encode

import time


class Cryptography(AnonymizationOperation):
    """Extract text from a PDF."""

    def __init__(self):
        super(Cryptography, self).__init__()
        self.cryptoKey = b'Sixteen byte key'
        self.cryptoIV = b'I am IV for AES!'
        self.cryptoSalt = "a.9_Oq1S*23xLgB"

    def HashEventAttribute(self, xesLog, targetedAttribute, conditional=None,  hashAlgo='ripemd160'):
        dict = {}

        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):
                # Only hash if value is a match or no mathcing is required
                isMatch = conditional in (None, '') or conditional(case, event)

                if(isMatch and targetedAttribute in event.keys()):
                    # Perform time consuming operations only once
                    if(event[targetedAttribute] in dict.keys()):
                        event[targetedAttribute] = dict[event[targetedAttribute]]
                    else:
                        h = hashlib.new(hashAlgo)
                        h.update((self.cryptoSalt + str(event[targetedAttribute])).encode('utf-8'))

                        hVal = h.hexdigest()
                        dict[event[targetedAttribute]] = hVal
                        event[targetedAttribute] = hVal

        return self.AddExtension(xesLog, 'cry', 'event', targetedAttribute)

    def EncryptEventAttribute(self, xesLog, targetedAttribute, conditional=None):
        dict = {}

        for case_index, case in enumerate(xesLog):
            for event_index, event in enumerate(case):
                # Only crypt if value is a match or no mathcing is required
                isMatch = conditional in (None, '') or conditional(case, event)

                if(isMatch and targetedAttribute in event.keys()):
                    # Perform time consuming operations only once
                    if(event[targetedAttribute] in dict.keys()):
                        event[targetedAttribute] = dict[event[targetedAttribute]]
                    else:
                        cipher = AES.new(self.cryptoKey, AES.MODE_CBC, iv=self.cryptoIV)
                        hVal = cipher.encrypt(pad(str(event[targetedAttribute]).encode('utf-8'), AES.block_size)).hex()

                        dict[event[targetedAttribute]] = hVal
                        event[targetedAttribute] = hVal

        return self.AddExtension(xesLog, 'cry', 'event', targetedAttribute)

    def HashCaseAttribute(self, xesLog, targetedAttribute, conditional=None, hashAlgo='ripemd160'):
        h = hashlib.new(hashAlgo)
        dict = {}

        for case_index, case in enumerate(xesLog):
            # Only hash if value is a match or no mathcing is required
            isMatch = conditional in (None, '') or conditional(case, None)

            if(isMatch and targetedAttribute in case.attributes.keys()):
                    # Perform time consuming operations only once
                if(case.attributes[targetedAttribute] in dict.keys()):
                    case.attributes[targetedAttribute] = dict[case.attributes[targetedAttribute]]
                else:
                    h.update((self.cryptoSalt + str(case.attributes[targetedAttribute])).encode('utf-8'))
                    hVal = h.hexdigest()

                    dict[case.attributes[targetedAttribute]] = hVal
                    case.attributes[targetedAttribute] = hVal

        return self.AddExtension(xesLog, 'cry', 'case', targetedAttribute)

    def EncryptCaseAttribute(self, xesLog, targetedAttribute, conditional=None):
        dict = {}

        for case_index, case in enumerate(xesLog):
            # Only crypt if value is a match or no mathcing is required
            isMatch = conditional in (None, '') or conditional(case, None)

            if(isMatch and targetedAttribute in case.attributes.keys()):
                    # Perform time consuming operations only once
                if(case.attributes[targetedAttribute] in dict.keys()):
                    case.attributes[targetedAttribute] = dict[case.attributes[targetedAttribute]]
                else:
                    cipher = AES.new(self.cryptoKey, AES.MODE_CBC, iv=self.cryptoIV)
                    hVal = cipher.encrypt(pad(str(case.attributes[targetedAttribute]).encode('utf-8'), AES.block_size)).hex()

                    dict[case.attributes[targetedAttribute]] = hVal
                    case.attributes[targetedAttribute] = hVal

        return self.AddExtension(xesLog, 'cry', 'case', targetedAttribute)
