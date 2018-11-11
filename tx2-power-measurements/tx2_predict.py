"""
Convient power measurement script for the Jetson TX2/Tegra X2.
"""
import os
from Two_stream_model import two_stream_model
import numpy as np
from Data import *
import tensorflow as tf
from keras import backend as K

# descr, i2c-addr, channel
_nodes = [('module/main', '0041', '0'),
          ('module/cpu', '0041', '1'),
          ('module/ddr', '0041', '2'),
          ('module/gpu', '0040', '0'),
          ('module/soc', '0040', '1'),
          ('module/wifi', '0040', '2'),

          ('board/main', '0042', '0'),
          ('board/5v0-io-sys', '0042', '1'),
          ('board/3v3-sys', '0042', '2'),
          ('board/3v3-io-sleep', '0043', '0'),
          ('board/1v8-io', '0043', '1'),
          ('board/3v3-m.2', '0043', '2'),
          ]

_valTypes = ['power', 'voltage', 'current']
_valTypesFull = ['power [mW]', 'voltage [mV]', 'current [mA]']


def getNodes():
    """Returns a list of all power measurement nodes, each a
    tuple of format (name, i2d-addr, channel)"""
    return _nodes


def getNodesByName(nameList=['module/main']):
    return [_nodes[[n[0] for n in _nodes].index(name)] for name in nameList]


def powerSensorsPresent():
    """Check whether we are on the TX2 platform/whether the sensors are present"""
    return os.path.isdir('/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/')


def getPowerMode():
    return os.popen("nvpmodel -q | grep 'Power Mode'").read()[15:-1]


def readValue(i2cAddr='0041', channel='0', valType='power'):
    """Reads a single value from the sensor"""
    fname = '/sys/bus/i2c/drivers/ina3221x/0-%s/iio_device/in_%s%s_input' % (i2cAddr, valType, channel)
    with open(fname, 'r') as f:
        return f.read()


def getModulePower():
    """Returns the current power consumption of the entire module in mW."""
    return float(readValue(i2cAddr='0041', channel='0', valType='power'))


def getAllValues(nodes=_nodes):
    """Returns all values (power, voltage, current) for a specific set of nodes."""
    return [[float(readValue(i2cAddr=node[1], channel=node[2], valType=valType))
             for valType in _valTypes]
            for node in nodes]


def printFullReport():
    """Prints a full report, i.e. (power,voltage,current) for all measurement nodes."""
    from tabulate import tabulate
    header = []
    header.append('description')
    for vt in _valTypesFull:
        header.append(vt)

    resultTable = []
    for descr, i2dAddr, channel in _nodes:
        row = []
        row.append(descr)
        for valType in _valTypes:
            row.append(readValue(i2cAddr=i2dAddr, channel=channel, valType=valType))
        resultTable.append(row)
    print(tabulate(resultTable, header))


import threading
import time


class PowerLogger:
    """This is an asynchronous power logger.
    Logging can be controlled using start(), stop().
    Special events can be marked using recordEvent().
    Results can be accessed through
    """

    def __init__(self, interval=0.01, nodes=_nodes):
        """Constructs the power logger and sets a sampling interval (default: 0.01s)
        and fixes which nodes are sampled (default: all of them)"""
        self.interval = interval
        self._startTime = -1
        self.eventLog = []
        self.dataLog = []
        self._nodes = nodes

    def start(self):
        "Starts the logging activity"""

        # define the inner function called regularly by the thread to log the data
        def threadFun():
            # start next timer
            self.start()
            # log data
            t = self._getTime() - self._startTime
            self.dataLog.append((t, getAllValues(self._nodes)))
            # ensure long enough sampling interval
            t2 = self._getTime() - self._startTime
            assert (t2 - t < self.interval)

        # setup the timer and launch it
        self._tmr = threading.Timer(self.interval, threadFun)
        self._tmr.start()
        if self._startTime < 0:
            self._startTime = self._getTime()

    def _getTime(self):
        # return time.clock_gettime(time.CLOCK_REALTIME)
        return time.time()

    def recordEvent(self, name):
        """Records a marker a specific event (with name)"""
        t = self._getTime() - self._startTime
        self.eventLog.append((t, name))

    def stop(self):
        """Stops the logging activity"""
        self._tmr.cancel()

    def getDataTrace(self, nodeName='module/main', valType='power'):
        """Return a list of sample values and time stamps for a specific measurement node and type"""
        pwrVals = [itm[1][[n[0] for n in self._nodes].index(nodeName)][_valTypes.index(valType)]
                   for itm in self.dataLog]
        timeVals = [itm[0] for itm in self.dataLog]
        return timeVals, pwrVals

    def showDataTraces(self, names=None, valType='power', showEvents=True):
        """creates a PyPlot figure showing all the measured power traces and event markers"""
        if names == None:
            names = [name for name, _, _ in self._nodes]

        # prepare data to display
        TPs = [self.getDataTrace(nodeName=name, valType=valType) for name in names]
        Ts, _ = TPs[0]
        Ps = [p for _, p in TPs]
        energies = [self.getTotalEnergy(nodeName=nodeName) for nodeName in names]
        Ps = list(map(list, zip(*Ps)))  # transpose list of lists

        # draw figure
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.plot(Ts, Ps)
        plt.xlabel('time [s]')
        plt.ylabel(_valTypesFull[_valTypes.index(valType)])
        plt.grid(True)
        plt.legend(['%s (%.2f J)' % (name, enrgy / 1e3) for name, enrgy in zip(names, energies)])
        plt.title('power trace (NVPModel: %s)' % (os.popen("nvpmodel -q | grep 'Power Mode'").read()[15:-1],))
        if showEvents:
            for t, _ in self.eventLog:
                plt.axvline(x=t, color='black')
        plt.show()

    def showMostCommonPowerValue(self, nodeName='module/main', valType='power', numBins=100):
        """computes a histogram of power values and print most frequent bin"""
        import numpy as np
        _, pwrData = np.array(self.getDataTrace(nodeName=nodeName, valType=valType))
        count, center = np.histogram(pwrData, bins=numBins)
        # import matplotlib.pyplot as plt
        # plt.bar((center[:-1]+center[1:])/2.0, count, align='center')
        maxProbVal = center[np.argmax(count)]  # 0.5*(center[np.argmax(count)] + center[np.argmax(count)+1])
        print('max frequent power bin value [mW]: %f' % (maxProbVal,))

    def getTotalEnergy(self, nodeName='module/main', valType='power'):
        """Integrate the power consumption over time."""
        timeVals, dataVals = self.getDataTrace(nodeName=nodeName, valType=valType)
        assert (len(timeVals) == len(dataVals))
        tPrev, wgtdSum = 0.0, 0.0
        for t, d in zip(timeVals, dataVals):
            wgtdSum += d * (t - tPrev)
            tPrev = t
        return wgtdSum

    def getAveragePower(self, nodeName='module/main', valType='power'):
        energy = self.getTotalEnergy(nodeName=nodeName, valType=valType)
        timeVals, _ = self.getDataTrace(nodeName=nodeName, valType=valType)
        return energy / timeVals[-1]


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    K.set_session(tf.Session(config=config))
    
    printFullReport()
    #    print(getModulePower())
    #    pl = PowerLogger(interval=0.05, nodes=getNodesByName(['module/main', 'board/main']))
    model = two_stream_model()
    test_data = get_static_frame_and_stacked_opt_flows()
    pl = PowerLogger(interval=0.05, nodes=list(filter(lambda n: n[0].startswith('module/'), getNodes())))
    pl.start()
    #time.sleep(1)
    pl.recordEvent('run model!')

    for i in range(50):
        result = model.predict([np.expand_dims(test_data[0], axis=0),
                                np.expand_dims(test_data[1], axis=0)])

    time.sleep(1.5)
    pl.stop()
    pl.showDataTraces()
