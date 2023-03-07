from .derivFuncs1 import derivFuncList1
from .derivFuncs1 import gaussian_func as gaussian_func1
from .derivFuncs2 import derivFuncList2
from .derivFuncs2 import gaussian_func as gaussian_func2
from .derivFuncs3 import derivFuncList3
from .derivFuncs3 import gaussian_func as gaussian_func3
from .deconvoluteFuncs import funcList
from .deconvoluteFuncs import gaussian_func as gaussian_func0
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import numpy as np

@dataclass
class Gaussian():
    center: float
    width: float
    amplitude: float
    
def sortPeaks(peaks: list[Gaussian]) -> list[Gaussian]:
    centerList = [gauss.center for gauss in peaks]
    sortList = sorted(centerList)
    outGaussList = []
    for center in sortList:
        for gauss in peaks:
            if gauss.center == center:
                outGaussList += [gauss]
    return outGaussList

def evToNm(eV: float | list[float], error: float = 0.0) -> float | tuple[float, float, float]:
    '''Converts eV values to nm'''
    def ev2nm(eV):
        h = 4.135667e-15
        c = 2.99792e8
        return np.divide(np.multiply(h, c), np.multiply(eV, 1e-9))

    if type(eV) == float:
        nm = ev2nm([eV, eV + error, eV - error])
        if error == 0:
            return nm[0]
        elif error != 0:
            return (nm[0], nm[1], nm[2])

    else:
        nm = ev2nm(eV)
        return nm

def nmToEv(nm, error: float = 0.0):
    '''Converts eV values to nm. Is just a wrapper for evToNm, since the maths is the same'''
    return evToNm(nm, error)

def baseLineCorrect(y, shift: bool, level: bool) -> list[float]:
    if shift:
        y = np.subtract(y, min(y))
        shiftAmount = [min(y) for i in range(len(y))]
    if level:
        y = np.subtract(y, min(y))
        maxLoc = list(y).index(max(y))
        lower = y[0:maxLoc]
        lowerLoc = list(lower).index(min(lower))
        upper = y[maxLoc:-1]
        upperLoc = list(upper).index(min(upper))

        innerRange = (upperLoc + len(lower)) - lowerLoc
        interval = ((min(lower) - min(upper)) / innerRange)
        shiftAmount = [i * interval for i in range(1, len(y) + 1)]
        shiftAmount = np.subtract(shiftAmount, (len(y) - upperLoc) * interval)
        y = np.add(y, shiftAmount)
        y = np.subtract(y, min(y))
    return y

def clipFunc(x: list[float], y: list[float], units: str = 'eV') -> tuple[list[float], list[float]]:
    maxLoc = list(y).index(max(y))
    lower = y[0:maxLoc]
    lowerLoc = list(lower).index(min(lower))
    upper = y[maxLoc:-1]
    upperLoc = list(upper).index(min(upper))

    less = np.less(x, x[upperLoc + len(lower)])
    more = np.greater(x, x[lowerLoc])
    if units == 'eV':
        print(f'Clipped from {nmToEv(x[upperLoc+len(lower)]):.3f} eV to {nmToEv(x[lowerLoc]):.3f} eV')
    if units == 'nm':
        print(f'Clipped from {x[upperLoc+len(lower)]:.0f} nm to {x[lowerLoc]:.0f} nm')
    xor = np.logical_xor(less, more)
    x = np.delete(x, xor)
    y = np.delete(y, xor)
    return x, y

def process(x: list[float], y: list[float], ngauss: int, amp: list[float, float],
            sigma: list[int, int], maxIter: int, derivLevel:int) -> tuple[list[float], float, list[float], list[float], bool]:
    if derivLevel == 0:
        derivFuncList = funcList
        gaussian_func = gaussian_func0
    elif derivLevel == 1:
        derivFuncList = derivFuncList1
        gaussian_func = gaussian_func1
    elif derivLevel == 2:
        derivFuncList = derivFuncList2
        gaussian_func = gaussian_func2
    elif derivLevel == 3:
        derivFuncList = derivFuncList3
        gaussian_func = gaussian_func3

    minbounds = []
    maxbounds = []
    for i in range(ngauss):
        minbounds += [amp[0], min(x), sigma[0]]
        maxbounds += [amp[1], max(x), sigma[1]]

    try:
        popt_gauss, pcov_gauss = curve_fit(derivFuncList[ngauss], x, y, bounds=(minbounds, maxbounds), maxfev=maxIter)
        error = False
    except RuntimeError as e:
        print(e)
        print('Your input spectra may need clipping')
        return [0.0], 0.0, [0.0], [0.0], True

    amps = []
    for count, i in enumerate(popt_gauss):
        if count % 3 == 0:
            amps += [i]

    residual = np.abs(np.sum(np.subtract(y, derivFuncList[ngauss](x, *popt_gauss))))
    return amps, residual, popt_gauss, pcov_gauss, error

def backProcess(x: list[float], y: list[float], amp: list[float, float], sigma: list[int, int],
                locList: list[float], ampList: list[float], maxIter: int, ampCutoff: tuple[float, float]) -> tuple[list[float], float, list[float], list[float]]:

    culledLocList = []
    for i in zip(locList, ampList):
        # print(i[0], i[1])
        if i[1] <= 1 * (10**(- ampCutoff[0])) and i[1] >= 1 * (10**(- ampCutoff[1])):
            culledLocList += [i[0]]

    ngauss = len(culledLocList)
    minbounds = []
    maxbounds = []
    for i in culledLocList:
        minbounds += [amp[0], i - 0.001, sigma[0]]
        maxbounds += [amp[1], i, sigma[1]]

    popt_gauss, pcov_gauss = curve_fit(funcList[ngauss], x, y, bounds=(minbounds, maxbounds), maxfev=maxIter)

    amps = []
    for count, i in enumerate(popt_gauss):
        if count % 3 == 0:
            amps += [i]

    residual = np.abs(np.sum(np.subtract(y, funcList[ngauss](x, *popt_gauss))))
    return amps, residual, popt_gauss, pcov_gauss, ngauss

def smoothing(algorithm: str, y: list[float], level: int, savgolPoly: int, savgolDeriv: int) -> list[float]:
    if algorithm == 'Gaussian':
        y = gaussian_filter1d(y, level)
    elif algorithm == 'Savitzky–Golay':
        y = savgol_filter(y, level, savgolPoly, deriv=savgolDeriv)
    return y
