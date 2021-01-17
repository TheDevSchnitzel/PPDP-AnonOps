import csv
import os.path
import time
import sys
from ppdp_anonops import *
from ppdp_anonops.utils import *
from pm4py.objects.log.importer.xes import factory as xes_importer
from datetime import datetime, timedelta
from copy import deepcopy
import pickle
from joblib import Parallel, delayed
import multiprocessing as mp
import ujson
import json
entries = []


def getIterations():
    return 1000


def loadLog(path):
    return xes_importer.apply(path)


def prepareLog(log):
    val = datetime.today()
    for case_index, case in enumerate(log):
        for event_index, event in enumerate(case):
            if 'time:timestamp' not in event.keys():
                event['time:timestamp'] = val + timedelta(hours=event_index+1, minutes=event_index+1, seconds=event_index+1)
    return log


def performBenchmark(logPath, begin, end):
    log = loadLog(logPath)
    logName = os.path.basename(logPath)

    start_time = time.time()
    log = prepareLog(log)
    print("Preparation took: " + str(time.time() - start_time))
    print("Entries " + str(begin) + " - " + str(end))

    for i in range((end - begin) + 1):
        j = i + begin

        start_Time = time.time()
        tLog = pickle.loads(pickle.dumps(log, -1))
        print("    Deepcopy took: " + str(time.time() - start_Time))
        afterTime = time.time()

        # Create T1-T6 entries
        benchmarkSubstitution(tLog, logName, j)

        # Remove T1-T6 entries
        benchmarkSuppression(tLog, logName, j)

        # Hash the None values from suppression
        benchmarkCryptography(tLog, logName, j)

        benchmarkSwapping(tLog, logName, j)

        benchmarkGeneralization(tLog, logName, j)

        benchmarkCondensation(tLog, logName, j)

        benchmarkAddition(tLog, logName, j)

        print("    Processing took: " + str(time.time() - afterTime))
        print("Iteration " + str(j) + " took: " + str(time.time() - start_Time))
    global entries
    writeEntries(entries)


def benchmarkCopyStrategies(log, logPath):
    start_time = time.time()
    tLog = pickle.loads(pickle.dumps(log, -1))
    print('Pickle', (time.time() - start_time))

    start_time = time.time()
    tLog = deepcopy(log)
    print('DEEPCOPY', (time.time() - start_time))

    start_time = time.time()
    tLog = loadLog(logPath)
    print('Reload', (time.time() - start_time))


def benchmarkAddition(log, logName, i):
    start_time = time.time()

    a = Addition()
    a.AddEventAtRandomPlaceInTrace(log, [{"Name": "concept:name", "Value": "RandomEvent"}],  (lambda c, e: len(c) <= 40 and c[0]["concept:name"] == "b_1"))

    saveBenchmark(logName, 'Addition - Random Placing', (time.time() - start_time), i)


def benchmarkCondensation(log, logName, i):
    start_time = time.time()

    c = Condensation()
    c.CondenseEventAttributeBykModesCluster(log, "concept:name", [], 4, "mode")

    saveBenchmark(logName, 'Condensation - Name', (time.time() - start_time), i)


def benchmarkGeneralization(log, logName, i):
    start_time = time.time()

    g = Generalization()
    g.GeneralizeEventTimeAttribute(log, "time:timestamp", "hours")

    saveBenchmark(logName, 'Generalization - Timestamp', (time.time() - start_time), i)


def benchmarkSwapping(log, logName, i):
    start_time = time.time()

    s = Swapping()
    s.SwapEventAttributeBykModesClusterUsingMode(log, "concept:name", [], 5)

    saveBenchmark(logName, 'Swapping - Event', (time.time() - start_time), i)


def benchmarkSuppression(log, logName, i):
    start_time = time.time()

    s = Suppression()
    s.SuppressEventAttribute(log, "concept:name", (lambda c, e: e["concept:name"] in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]))

    saveBenchmark(logName, 'Suppression - Event', (time.time() - start_time), i)


def benchmarkSubstitution(log, logName, i):
    start_time = time.time()

    s = Substitution()
    s.SubstituteEventAttributeValue(log, "concept:name", ["d_2", "a_1", "e_7", "b_2", "c_1", "a_2", "c_2"], ["T1", "T2", "T3", "T4", "T5", "T6", "T7"])

    saveBenchmark(logName, 'Substitution - Event', (time.time() - start_time), i)


def benchmarkCryptography(log, logName, i):
    start_time = time.time()

    c = Cryptography()
    c.EncryptEventAttribute(log, "concept:name")

    saveBenchmark(logName, 'Cryptography - Hashing', (time.time() - start_time), i)
    # return [logName, 'Cryptography - Hashing', (time.time() - start_time), i]


def saveBenchmark(logName, operation, duration, run):
    global entries
    entries.append([logName, operation, duration, run])

    if(len(entries) > 1):
        writeEntries(entries)
        entries = []


def writeEntries(entries):
    if(not os.path.isfile('benchmark.csv')):
        with open('benchmark.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(['Log', 'Operation', 'Duration', 'Run'])

    with open('benchmark.csv', 'a+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for row in entries:
            spamwriter.writerow(row)


def main(args):
    _, _, filenames = next(os.walk(args[0]))
    for file in filenames:
        #file = filenames[-1]
        performBenchmark(os.path.join(args[0], file), int(args[1]), int(args[2]))


if __name__ == "__main__":
    # execute only if run as a script
    os.chdir('G:\\OneDrive\\Dokumente\\Studium\\B.Sc. - Thesis\\Repo\\PPDP-AnonOps')
    main(sys.argv[1:])
