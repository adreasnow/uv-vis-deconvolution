from .gaussian0 import gaussian0FuncList, gaussian0_func
from .gaussian0 import gaussian0FuncList as funcList
from .gaussian1 import gaussian1FuncList, gaussian1_func
from .gaussian2 import gaussian2FuncList, gaussian2_func
from .gaussian3 import gaussian3FuncList, gaussian3_func
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import numpy as np
import os

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
            sigma: list[int, int], maxIter: int, derivLevel: int) -> tuple[list[float], float, list[float], list[float], bool]:
    if derivLevel == 0:
        derivFuncList = gaussian0FuncList
        gaussian_func = gaussian0_func
    elif derivLevel == 1:
        derivFuncList = gaussian1FuncList
        gaussian_func = gaussian1_func
    elif derivLevel == 2:
        derivFuncList = gaussian2FuncList
        gaussian_func = gaussian2_func
    elif derivLevel == 3:
        derivFuncList = gaussian3FuncList
        gaussian_func = gaussian3_func

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

def saveSpectrum(filename: str, filepath: str, baseLine: str, clip: tuple[float, float],
                 autoClip: bool, derivLevel: int, amp: tuple[float, float], width: tuple[float, float],
                 convergence: float, gaussRange: tuple[int, int], maxIter: int, ampCutoff: float,
                 smoothingRaw: int, preSmoothingSigma: int, smoothingSigma: int, smoothingType: str,
                 savgolPoly: int, savgolDeriv: int,
                 derivPeaks: list[Gaussian], nonDerivPeaks: list[Gaussian],
                 x: list[float], yRaw: list[float], yDeriv: list[float]) -> None:
    name = filename.split(".")[0]
    path = f'{filepath}/{name}'
    settingsFile = f'{path}/settings.toml'
    derivGaussFile = f'{path}/derivGaussians.csv'
    nonDerivGaussFile = f'{path}/nonDerivGaussians.csv'
    rawSpectrumFile = f'{path}/rawSpectrum.csv'
    derivSpectrumFile = f'{path}/derivSpectrum.csv'
    fileList = [settingsFile, derivGaussFile, nonDerivGaussFile, rawSpectrumFile, derivSpectrumFile]

    try:
        os.mkdir(path)
    except FileExistsError:
        for file in fileList:
            try:
                os.remove(file)
                print(f'Deleted old {file}')
            except FileNotFoundError:
                pass

    tomlString = f'#### Input Parameters for {name} ####\n'
    tomlString += '[input]\n'
    tomlString += f'baseline = "{baseLine}"\n'
    tomlString += f'spectraRangeLow = [{clip[0]:.2f}, {clip[1]:.2f}]\n'
    tomlString += f'autoClip = {autoClip}\n\n'
    tomlString += '[fitting]\n'
    tomlString += f'derivLevel = {derivLevel}\n'
    tomlString += f'ampRange = [{amp[0]:.2f}, {amp[1]:.2f}]\n'
    tomlString += f'widthRange = [{width[0]:.2f}, {width[1]:.2f}]\n'
    tomlString += f'convergence = {convergence:.2f}\n'
    tomlString += f'gaussRange = [{gaussRange[0]}, {gaussRange[1]}]\n'
    tomlString += f'maxFittingIter = {maxIter}\n\n'
    tomlString += '[refitting]\n'
    tomlString += f'refitAmpCutoff = [{ampCutoff[0]:.2f}, {ampCutoff[1]:.2f}]\n\n'
    tomlString += '[smoothing]\n'
    tomlString += f'rawSmoothingLevel = {smoothingRaw}\n'
    tomlString += f'nonDerivSmoothingLevel = {preSmoothingSigma}\n'
    tomlString += f'derivSmoothingLevel = {smoothingSigma}\n'
    tomlString += f'smoothingAlgorithm = "{smoothingType}"\n'
    tomlString += f'SavGolPolyOrder = {savgolPoly}\n'
    tomlString += f'SavGolDerivLevel = {savgolDeriv}\n\n'

    with open(settingsFile, 'a+') as f:
        f.write(tomlString)

    with open(derivGaussFile, 'a+') as f:
        f.write('location,width,amplitude\n')
        for gauss in derivPeaks:
            f.write(f'{gauss.center:.3f},{gauss.width:.4f},{gauss.amplitude:.4f}\n')

    with open(nonDerivGaussFile, 'a+') as f:
        f.write('location,width,amplitude\n')
        for gauss in nonDerivPeaks:
            f.write(f'{gauss.center:.3f},{gauss.width:.4f},{gauss.amplitude:.4f}\n')

    with open(rawSpectrumFile, 'a+') as f:
        f.write('X (eV),Y (AU)\n')
        for (xpoint, ypoint) in zip(x, yRaw):
            f.write(f'{xpoint:.3f},{ypoint:.4f}\n')

    with open(derivSpectrumFile, 'a+') as f:
        f.write('X (eV),Y (AU)\n')
        for (xpoint, ypoint) in zip(x, yDeriv):
            f.write(f'{xpoint:.3f},{ypoint:.4f}\n')
    return
