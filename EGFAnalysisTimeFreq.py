'''
This software is used for extract group and phase velocity dispersion curves from surface wave empirical Green’s function (EGF) or cross-correlation function (CF) from ambient noise.	
'''

import numpy as np
import pandas as pd
from enum import Enum
from scipy import signal,interpolate
import math
from cmath import inf
import os

from scipy.signal import hilbert, windows
from scipy.fftpack import fft,ifft
from geopy import distance

from matplotlib import pyplot as plt
import seaborn as sns

import logging

log_name = 'EGFAnalysisTimeFreq'
logger = logging.getLogger(log_name)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(
    f"{log_name}.log", mode='w', encoding='utf-8')  # overwrite old files
console_handler.setFormatter(logging.Formatter(
    '%(levelname)s - %(message)s'))
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# setting all log levels
logger.setLevel(logging.DEBUG)

console_handler.setLevel('INFO')
file_handler.setLevel('DEBUG')


class gfcn_analysis:
    class GreenFcnObjectsType(Enum):
        A_to_B = 1,
        B_to_A = 2,
        A_add_B = 3
        
    def __init__(self, DataFileName , isEGF = True,
                 StartT=5, EndT=50, DeltaT=0.1,
                 StartV=2, EndV=5.2, DeltaV=0.002,
                 GreenFcnObjects=GreenFcnObjectsType.A_add_B,
                 WinAlpha=0.1,  NoiseTime=150,
                 MinSNR=5.0,
                 ):
        '''
        FilePath: path of the data file
        isEGF: True for EGF, False for CF
        StartT: start time of the analysis
        EndT: end time of the analysis
        DeltaT: time interval of the analysis
        StartV: start velocity of the analysis
        EndV: end velocity of the analysis
        DeltaV: velocity interval of the analysis
        GreenFcnObjectsType: the type of Green’s function objects
        WinAlpha: the proportion of cosine part to the whole window
        NoiseTime: the time of noise sampling
        '''
        self.DataFileName = DataFileName
        self.StartT = StartT
        self.EndT = EndT
        self.DeltaT = DeltaT
        self.StartV = StartV
        self.EndV = EndV
        self.DeltaV = DeltaV
        self.GreenFcnObjects = GreenFcnObjects
        self.MinSNR = MinSNR
        try:
            with open(DataFileName, 'r') as f:
                # read the first line
                data_header = list(map(float, f.readline().split()))
                # read the second line
                data_header2 = list(map(float, f.readline().split()))
                # read the rest
                self.RawData = np.loadtxt(f)
        except:
            logger.error(f'Fail to load data from {DataFileName}')
            raise

        # station longitude , latitude and altitude
        self.Longitude_A = data_header[0]
        self.Latitude_A = data_header[1]
        if len(data_header) == 3:
            self.Altitude_A = data_header[2]
        else:
            self.Altitude_A = 0.0
        self.Longitude_B = data_header2[0]
        self.Latitude_B = data_header2[1]
        if len(data_header2) == 3:
            self.Altitude_B = data_header2[2]
        else:
            self.Altitude_B = 0.0
        if self.Longitude_A < 0:
            self.Longitude_A += 360
        if self.Longitude_B < 0:
            self.Longitude_B += 360
        logger.debug(f'Longitude_A: {self.Longitude_A}, Latitude_A: {self.Latitude_A}, Altitude_A: {self.Altitude_A}')
        logger.debug(f'Longitude_B: {self.Longitude_B}, Latitude_B: {self.Latitude_B}, Altitude_B: {self.Altitude_B}')

        # calculate great circle distance
        circleDist = distance.great_circle(
            (self.Latitude_A, self.Longitude_A), (self.Latitude_B, self.Longitude_B)).km
        staElevDiff = abs(self.Altitude_A - self.Altitude_B)/1000
        if np.isnan(staElevDiff):
            staElevDiff = 0
        logger.debug(f'circleDist: {circleDist}, staElevDiff: {staElevDiff}')
        # correct station distance due to elevation difference
        self.StaDist = np.sqrt(circleDist**2 + staElevDiff**2)
        logger.info('Station distance: {} km'.format(self.StaDist))
      
        self.PtNum = self.RawData.shape[0]
        self.Time = self.RawData[:, 0]
        self.Green_AB = self.RawData[:, 1]
        self.Green_BA = self.RawData[:, 2]

        maxamp = max(max(self.Green_AB), max(self.Green_BA))
        if maxamp > 0:
            self.Green_AB /= maxamp
            self.Green_BA /= maxamp

        # using hilbert tranform to obtain EGF from CF if reading CF
        if isEGF == False:
            self.Green_AB = np.imag(hilbert(self.Green_AB))
            self.Green_BA = np.imag(hilbert(self.Green_BA))

        # select function object
        if GreenFcnObjects == GreenFcnObjects.A_to_B:
            self.GreenFcn = self.Green_AB
        elif GreenFcnObjects == GreenFcnObjects.B_to_A:
            self.GreenFcn = self.Green_BA
        elif GreenFcnObjects == GreenFcnObjects.A_add_B:
            self.GreenFcn = (self.Green_AB + self.Green_BA) / 2.0

        self.SampleT = self.Time[1] - self.Time[0]
        self.SampleF = 1 / self.SampleT

        # calculate the typical value of the time difference from the typical value of the velocity as the width of the window function
        self.StartWin = round(self.SampleF * self.StaDist / self.EndV)
        self.EndWin = round(self.SampleF * self.StaDist / self.StartV)
        if self.EndWin >= self.PtNum:
            self.EndWin = self.PtNum - 1
            self.StartV = np.ceil(10 * self.StaDist/self.Time[-1])/10
            logger.warning(f'Min velocity reset to {self.StartV}')

        # the number of time points
        self.NumCtrT = round((self.EndT - self.StartT) / self.DeltaT) + 1
        self.TPoint = np.linspace(self.StartT, self.EndT, self.NumCtrT)

        # the number of velocity points
        self.NumCtrV = round((self.EndV - self.StartV) / self.DeltaV) + 1
        self.VPoint = np.linspace(self.EndV, self.StartV, self.NumCtrV)

        # calculate the window function
        Window, TaperLen = self.GenerateSignalWindow(
            self.StartWin, self.EndWin, self.PtNum, WinAlpha)
        WinWave = self.GreenFcn * Window

        # extract noise window after the windowed surface wave
        self.NoisePt = round(NoiseTime/self.SampleT)
        self.NoiseStartIndex = self.EndWin + TaperLen
        if (self.NoiseStartIndex + self.NoisePt) < self.PtNum:
            self.NoiseWinWave = self.GreenFcn[self.NoiseStartIndex:self.NoiseStartIndex + self.NoisePt]
        else:
            self.NoiseWinWave = self.GreenFcn[self.NoiseStartIndex:]
            self.NoisePt = len(self.NoiseWinWave)
            logger.warning(
                f'Noise window length of {self.NoiseWinWave.shape[0]}, not long enough')

        self.WaveClipPt = min((self.EndWin + TaperLen), self.PtNum)
        self.WinWaveClip = WinWave[:self.WaveClipPt]

    def GroupVelocityImgCalculate(self):
        '''
        calculate group velocity dispersion curve
        args:
        return:
        '''
        # calculate envelope images for signal and noise and estimate SNR
        # SNR(T) =  max(signal envelope at period T)/mean(noise envelope at period T)
        EnvelopeImageSignal = self.EnvelopeImageCalculation(
            self.WinWaveClip, self.SampleF, self.TPoint, self.StaDist)
        AmpS_T = np.max(EnvelopeImageSignal, axis=1)
        EnvelopeImageNoise = self.EnvelopeImageCalculation(
            self.NoiseWinWave * windows.tukey(self.NoisePt, 0.2), self.SampleF, self.TPoint, self.StaDist)
        self.SNR_T = AmpS_T / np.mean(EnvelopeImageNoise, axis=1)

        self.HighSNRIndex = np.where(self.SNR_T > self.MinSNR)
        self.SNRIndex = np.zeros(self.NumCtrT)
        self.SNRIndex[self.HighSNRIndex] = 1
        # for those not so bad pts, if they are in the middle of good pts, accept them
        for ii in range(1, self.NumCtrT-1):
            if self.SNRIndex[ii] == 0:
                if self.SNR_T[ii] > self.MinSNR / 2 and self.SNRIndex[ii -1] == 1 and self.SNRIndex[ii + 1] == 1:
                    self.SNRIndex[ii] = 1

        # calculate the velocity
        TravPtV = self.StaDist / (np.asarray(range(self.StartWin - 1, self.EndWin + 1)) * self.SampleT)
        GroupVelocityImg = []
        for i in range(self.NumCtrT):
            # GroupVelocityImg.append(np.interp(
            #     self.VPoint, TravPtV[::-1], (EnvelopeImageSignal[i, self.StartWin:self.EndWin+1]/AmpS_T[i])[::-1]))
            GroupVelocityImg.append(interpolate.interp1d
                                    (TravPtV[::-1], (EnvelopeImageSignal[i, self.StartWin-1:self.EndWin+1]/AmpS_T[i])[::-1], kind='cubic', bounds_error=False,fill_value=0)(self.VPoint))
        GroupVelocityImg = np.transpose(np.array(GroupVelocityImg))
        # reverse
        GroupVelocityImg = GroupVelocityImg[::-1]

        self.GroupVelocityImg = GroupVelocityImg
        return GroupVelocityImg

    def GroupVelocityImgPlot(self):
        # packaged data
        self.VImgData = pd.DataFrame(self.GroupVelocityImg[::-1])
        self.VImgData.columns = np.round(self.TPoint)
        self.VImgData.index = np.round(self.VPoint, 2)

        fig = plt.figure(num='Group Velocity Image', figsize=(10, 6))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        # plot signal (blue) and noise (red)
        ax = fig.add_subplot(321)
        ax.plot(self.Time[:self.WaveClipPt], self.WinWaveClip, 'b-', label='Windowed Signal')
        ax.plot(self.Time[self.NoiseStartIndex:self.NoiseStartIndex+self.NoisePt],
                self.NoiseWinWave, 'r-', label='Noise')
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Amplitude')
        ax.legend()

        # plot SNR
        ax = fig.add_subplot(322)
        ax.set_yscale('log')
        ax.plot(self.TPoint, self.SNR_T, 'b-', label='SNR')
        ax.set_xlabel('Period(s)')
        ax.set_ylabel('SNR')
        ax.grid(which='both')
        # flag values greater than MinSNR
        ax.plot(self.TPoint[self.HighSNRIndex], self.SNR_T[self.HighSNRIndex],
                'r*', label='SNR > ' + str(self.MinSNR))
        ax.legend()

        # plot velocity image
        fig.add_subplot(3,1,(2,3))
        # ax = sns.heatmap(df, cmap="RdBu_r")
        # ax = sns.heatmap(df, cmap="Spectral_r")
        ax = sns.heatmap(self.VImgData, cmap="RdYlBu_r")

        ax.set_xlabel('Period(s)')
        ax.set_ylabel('Group Velocity(km/s)')

        if hasattr(self, 'GroupDisperCurve'):
            ax.plot(range(self.NumCtrT), self.NumCtrV * (self.EndV - self.GroupDisperCurve) /
                    (self.EndV - self.StartV), 'blue', label='Dispersion Curve')

        plt.show()

    def AutoGroupDisperPick(self,minlamdaRatio=2):
        '''
        function for automatic pick of group velocity dispersion curve
        '''
        self.minlamdaRatio = minlamdaRatio
        dc = self.DeltaV
        dT = self.DeltaT

        if not hasattr(self, 'refgroupdisp'):
            logger.error('No reference group dispersion curve, please run LoadRefGroupDisper() first')
            return None
        gRef_low = np.interp(self.TPoint, self.refgroupdisp[:,0], self.refgroupdisp[:,1])
        gRef_high = np.interp(self.TPoint, self.refgroupdisp[:,0], self.refgroupdisp[:,2])
        gRef = (gRef_low + gRef_high) / 2

        # find the approximate maximum period for dispersion analysis which satisfies interstation_distance > minlamdaRatio*c*T
        lamda = gRef * self.TPoint
        II = np.where(lamda * minlamdaRatio >= self.StaDist)
        if len(II[0]) > 0:
            nMaxT = II[0][0]
        else:
            nMaxT = self.NumCtrT

        # set tial T and c for searching dispersion curves
        T_try_index = np.round(np.arange(10)/10*(self.TPoint[nMaxT-1] - self.TPoint[0])/dT).astype(int)
        T_try = self.TPoint[0] + T_try_index*dT  # trial T for searching dispersion index

        g_try_index = np.argmax(self.GroupVelocityImg[:, T_try_index], axis=0)
       
        # search the dispersion curves
        GroupVDisp_try = np.zeros((len(T_try),self.NumCtrT))
        k = 0
        for Initialg, InitialT in zip(g_try_index,T_try_index):
            DispPt = self.AutoSearch(Initialg,InitialT,self.GroupVelocityImg)
            if k == 0:
                GroupVDisp_try[k,:] = self.VPoint[-1] + DispPt*dc
                k = k + 1
            else:
                tempDisp = self.VPoint[-1] + DispPt*dc
                SameDisperIndex = 0
                for nn in range(k):
                    if sum(abs(GroupVDisp_try[nn,:] - tempDisp)) < 1e-4:
                        SameDisperIndex = 1
                if SameDisperIndex == 0:
                    GroupVDisp_try[k,:] = tempDisp
                    k = k + 1
        # determine the quality of the dispersion curve by looking at how many
        # disperion points fall in the reference dispersion range
        NumDispCurve = k
        InRangePt = np.zeros(NumDispCurve)
        for i in range(NumDispCurve):
            GoodIndex = np.sign(
                GroupVDisp_try[i, :] - gRef_low) + np.sign(gRef_high - GroupVDisp_try[i, :])
            II = np.where(GoodIndex == 2)
            InRangePt[i] = len(II[0])
        maxpt = np.max(InRangePt)
        meanpt = np.mean(InRangePt)
        # find the best several dispersion curves within reference range
        II = np.where(InRangePt >= (2 * maxpt + meanpt) / 3)
        if len(II[0]) == 1:
            GroupVDisp = GroupVDisp_try[II[0], :]
        else:
            RefObsDispDiff = np.zeros(len(II[0]))
            ObsSumAbsDiff = np.zeros(len(II[0]))
            for i in range(len(II[0])):
                RefObsDispDiff[i] = sum(abs(GroupVDisp_try[II[0][i], :] - ( gRef_low + gRef_high)/2))
                ObsSumAbsDiff[i] = sum(abs(np.diff(GroupVDisp_try[II[0][i], :])))
            # find lowest difference dispersion curve with respect to reference
            mindiff = np.min(RefObsDispDiff)
            index1 = np.where(RefObsDispDiff == mindiff)
            # find smoothest dispersion curve
            minabs = np.min(ObsSumAbsDiff)
            index2 = np.where(ObsSumAbsDiff == minabs)
            if index1 == index2:
                GroupVDisp = GroupVDisp_try[II[0][index1], :]
            else:
                BestTwoDiff = abs(GroupVDisp_try[II[0][index1], :] - GroupVDisp_try[II[0][index2], :])
                # 2/3 of two best dispersion curves are overlapping
                if len(np.where(BestTwoDiff < 1e-3)) > 2/3 * nMaxT:
                    # choose the smoothest one
                    GroupVDisp = GroupVDisp_try[II[0][index2], :]
                else:
                    # choose the smaller difference one if 2/3 dispersion are different
                    GroupVDisp = GroupVDisp_try[II[0][index1], :]

        NewDisper = np.stack((self.TPoint, GroupVDisp[0],np.ones(self.NumCtrT))).T

        # whether NewDisper falls in Group V range or not
        for ii in range(self.NumCtrT):
            if NewDisper[ii,1] > gRef_high[ii] or NewDisper[ii,1] < gRef_low[ii]:
                NewDisper[ii,2] = 0
        
        # find the group velocity corresponding to the maximum amplitude of the
        # envelope at each period
        GroupVMaxAmp = np.max(self.GroupVelocityImg, axis=0)
        JJ = abs(GroupVMaxAmp - GroupVDisp[0]) < 0.01
        # if the picked dispersion is very different from the dispersion
        # corresponding to the maximum amplitude of the envelope (only 1/10 points
        # are overlapping), not save the dispersion curve and will not pick phase
        # velocity dispersion curve by setting IsDispGood = False
        if len(JJ) < 0.1 * self.NumCtrT:
            IsDispGood = False
        else:
            # find reasonable dispersion points with high SNR
            GoodIndex = NewDisper[:,2] + self.SNRIndex
            II = np.where(GoodIndex == 2)

            # save dispersion data when having at least 4 or 0.1*self.NumCtrT good points
            if len(II[0]) >= 4 or len(II[0]) >= 0.1 * self.NumCtrT:
                IsDispGood = True
            else:
                IsDispGood = False
        self.GroupDisperCurve = GroupVDisp[0]
        return IsDispGood, self.GroupDisperCurve

    class TimeVariableFilterType(Enum):
        no = 1,
        obs = 2

    def PhaseVelocityImgCalculate(self,
                         TimeVariableFilter=TimeVariableFilterType.no,
                         WindnumT=5,
                         Winmintime=25,
                         ):
        '''
        calculate phase velocity dispersion curve
        args:
            BandWidth: 
            WindnumT : number of window period
            Winmintime : minimum of window time
        return:
        '''       
        BandWidth = self.DeltaT
        filter_num = int(2 ** math.ceil(np.log2(1024*self.SampleF)))

        #TODO:How to determine this value
        if(filter_num > 8192):
            filter_num=8192

        filter_KaiserPara = 6
        HalfFilterNum = int(filter_num / 2)
        WinWave = np.concatenate((np.copy(self.WinWaveClip), np.zeros(HalfFilterNum)))

        # No GroupVDisp for time-variable filtering analysis
        if not hasattr(self, 'GroupDisperCurve'):
            TimeVariableFilter = self.TimeVariableFilterType.no
            logger.warning(
                'No GroupDisperCurve for time-variable filtering analysis')

        if TimeVariableFilter == self.TimeVariableFilterType.obs:
            GroupTime = self.StaDist/self.GroupDisperCurve

            III = np.where(GroupTime == np.inf)
            GroupTime[III] = self.StaDist/self.StartV

            GroupVwinMin = self.StaDist/(GroupTime + np.maximum(WindnumT/2*self.TPoint,Winmintime))
            GroupVwinMax = self.StaDist/(GroupTime - np.maximum(WindnumT/2*self.TPoint, Winmintime))
            III = np.where(GroupVwinMax <= 0)
            GroupVwinMax[III] = self.EndWin *2

            # plot time-variable group velocity window at different periods
            # reset period-dependent group v winodw: has to be less than
            # 0.98*gfcn.WinMaxV or larger than 1.02*gfcn.WinMinV
            pWinMinV = np.maximum(1.02*self.StartV, GroupVwinMin)
            pWinMaxV = np.minimum(0.98*self.EndV, GroupVwinMax)
            GroupVwinMin = pWinMinV
            GroupVwinMax = pWinMaxV
            # hold(h2, 'on');plot(self.TPoint, pWinMinV, 'w--', 'LineWidth', 2);
            # hold(h2, 'on'); plot(self.TPoint, pWinMaxV,'w--', 'LineWidth', 2);    

        PhaseImg = []
        for numt in range(self.NumCtrT):
            CtrT = self.StartT + numt * self.DeltaT
            CtrF = (2 / self.SampleF) / CtrT
            LowF = (2 / self.SampleF) / (CtrT + 0.5 * BandWidth)
            HighF = (2 / self.SampleF) / (CtrT - 0.5 * BandWidth)
            filter_data = signal.firwin(filter_num + 1, [LowF, HighF], pass_zero=False,window=('kaiser', filter_KaiserPara))
            if TimeVariableFilter == self.TimeVariableFilterType.obs:
                winpt = np.round(np.maximum(WindnumT * CtrT,Winmintime)*self.SampleF)
                # to ensure winpt is even number
                if winpt % 2 == 1:
                    winpt = winpt + 1
                wintukey = signal.windows.tukey(int(winpt), 0.2) 
                grouppt = winpt + round(GroupTime[numt]*self.SampleF + 1)
                
                tmpWave = np.concatenate((np.zeros(int(winpt)),WinWave[:self.WaveClipPt],np.zeros(int(winpt))))
                tmpWave[int(grouppt-winpt//2):int(grouppt+winpt//2)] = tmpWave[int(grouppt-winpt//2):int(grouppt+winpt//2)]*wintukey
                tmpWave[:int(grouppt-winpt//2)] = 0
                tmpWave[int(grouppt+winpt//2):] = 0
                NewWinWave = np.zeros(self.WaveClipPt + HalfFilterNum);
                NewWinWave[:self.WaveClipPt] = tmpWave[int(winpt):int(winpt+self.WaveClipPt)]
                FilteredWave = signal.lfilter(filter_data , 1, NewWinWave)
            else:
                # filtering
                FilteredWave = signal.lfilter(filter_data, 1, WinWave)
            # inverse order
            FilteredWave = FilteredWave[::-1]
            # filtering
            FilteredWave = signal.lfilter(filter_data, 1, FilteredWave)
            # inverse order
            FilteredWave = FilteredWave[::-1]
            # clip
            FilteredWave = (FilteredWave[:self.WaveClipPt])
            # normalization
            FilteredWave = FilteredWave / np.max(np.abs(FilteredWave))
            PhaseImg.append(FilteredWave)

        timeptnum = np.array(range(self.StartWin, self.EndWin))
        time = timeptnum * self.SampleT
        PhaseVelocityImg = []
        for i in range(self.NumCtrT):
            CenterT = self.StartT + i * self.DeltaT
            TravPtV = self.StaDist/(time - CenterT/8)

            # time - CenterT/8 maybe zero
            TravPtV[TravPtV == inf] = 100

            # PhaseVelocityImg.append(np.interp(
            #         self.VPoint, TravPtV[::-1], (PhaseImg[i][self.StartWin:self.EndWin])[::-1]))
            PhaseVelocityImg.append(interpolate.interp1d
                                    (TravPtV[::-1], (PhaseImg[i][self.StartWin:self.EndWin])[::-1], kind='cubic', bounds_error=False, fill_value=0)(self.VPoint))
        PhaseVelocityImg = np.transpose(np.array(PhaseVelocityImg))
        # reverse
        PhaseVelocityImg = PhaseVelocityImg[::-1]

        self.PhaseVelocityImg = PhaseVelocityImg
        return PhaseVelocityImg

    def PhaseVelocityImgPlot(self):
        # packaged data
        self.VImgData = pd.DataFrame(self.PhaseVelocityImg[::-1])
        self.VImgData.columns = np.round(self.TPoint)
        self.VImgData.index = np.round(self.VPoint, 2)

        fig = plt.figure(num='Phase Velocity Image', figsize=(10, 6))
        fig.subplots_adjust(hspace=0.5)

        # plot velocity image
        # ax = sns.heatmap(df, cmap="RdBu_r")
        # ax = sns.heatmap(df, cmap="Spectral_r")
        ax = sns.heatmap(self.VImgData, cmap="RdYlBu_r")

        ax.set_xlabel('Period(s)')
        ax.set_ylabel('Phase Velocity(km/s)')

        if hasattr(self, 'PhaseDisperCurve'):
            ax.plot(range(self.NumCtrT), self.NumCtrV * (self.EndV - self.PhaseDisperCurve) / (self.EndV - self.StartV), 'blue', label='Dispersion Curve')

        plt.show()

    def AutoPhaseDisperPick(self):
        '''
        function for automatic pick of phase velocity dispersion curve
        '''
        minlamdaRatio = 2

        dc = self.DeltaV
        dT = self.DeltaT

        if not hasattr(self, 'refphasedisp'):
            logger.error('No reference phase dispersion curve, please run LoadRefPhaseDisper() first')
            return None
        cRef_low = np.interp(self.TPoint, self.refphasedisp[:, 0], self.refphasedisp[:, 1])
        cRef_high = np.interp(self.TPoint, self.refphasedisp[:, 0], self.refphasedisp[:, 2])
        cRef = (cRef_low + cRef_high) / 2

        # find the approximate maximum period for dispersion analysis which satisfies interstation_distance > minlamdaRatio*c*T
        lamda = cRef * self.TPoint
        II = np.where(lamda * minlamdaRatio >= self.StaDist)
        if len(II[0]) > 0:
            nMaxT = II[0][0]
        else:
            nMaxT = self.NumCtrT

        # max c in the ref. disper
        cmax_ref = max(cRef_high)
        # min c in the ref. disper
        cmin_ref = min(cRef_low)

        # set tial T and c for searching dispersion curves
        T_try_index = np.round(np.array(range(5))/10 *
                               (self.TPoint[nMaxT-1] - self.TPoint[0])/dT).astype(int)
        # trial T for searching dispersion index
        T_try = self.TPoint[0] + T_try_index*dT
        c_try_index = np.round(((cmin_ref - self.VPoint[-1]) + (cmax_ref - cmin_ref)*range(9)/10)/dc).astype(int)
        c_try = self.VPoint[-1] + c_try_index*dc

        # search the dispersion curves
        PhaseVDisp_try = np.zeros((len(T_try)*len(c_try), self.NumCtrT))
        k = 0
        for i in range(len(T_try)):
            for j in range(len(c_try)):
                Initialc = c_try_index[j]
                InitialT = T_try_index[i]
                DispPt = self.AutoSearch(Initialc, InitialT, self.PhaseVelocityImg)
                if k == 0:
                    PhaseVDisp_try[k, :] = self.VPoint[-1] + DispPt*dc
                    k = k + 1
                else:
                    tempDisp = self.VPoint[-1] + DispPt*dc
                    SameDisperIndex = 0
                    for nn in range(k):
                        if sum(abs(PhaseVDisp_try[nn, :] - tempDisp)) < 1e-4:
                            SameDisperIndex = 1
                    if SameDisperIndex == 0:
                        PhaseVDisp_try[k, :] = tempDisp
                        k = k + 1
        # determine the quality of the dispersion curve by looking at how many
        # disperion points fall in the reference dispersion range
        NumDispCurve = k
        InRangePt = np.zeros(NumDispCurve)
        for i in range(NumDispCurve):
            GoodIndex = np.sign(
                PhaseVDisp_try[i, :] - cRef_low) + np.sign(cRef_high - PhaseVDisp_try[i, :])
            II = np.where(GoodIndex == 2)
            InRangePt[i] = len(II[0])
        maxpt = np.max(InRangePt)
        meanpt = np.mean(InRangePt)
        # find the best several dispersion curves within reference range
        II = np.where(InRangePt >= (2 * maxpt + meanpt) / 3)
        if len(II[0]) == 1:
            PhaseVDisp = PhaseVDisp_try[II[0], :]
        else:
            RefObsDispDiff = np.zeros(len(II[0]))
            ObsSumAbsDiff = np.zeros(len(II[0]))
            for i in range(len(II[0])):
                RefObsDispDiff[i] = sum(
                    abs(PhaseVDisp_try[II[0][i], :] - (cRef_low + cRef_high)/2))
                ObsSumAbsDiff[i] = sum(
                    abs(np.diff(PhaseVDisp_try[II[0][i], :])))
            # find lowest difference dispersion curve with respect to reference
            mindiff = np.min(RefObsDispDiff)
            index1 = np.where(RefObsDispDiff == mindiff)
            # find smoothest dispersion curve
            minabs = np.min(ObsSumAbsDiff)
            index2 = np.where(ObsSumAbsDiff == minabs)
            if (index1[0] == index2[0]).all():
                PhaseVDisp = PhaseVDisp_try[II[0][index1], :]
            else:
                BestTwoDiff = abs(
                    PhaseVDisp_try[II[0][index1], :] - PhaseVDisp_try[II[0][index2], :])
                # 2/3 of two best dispersion curves are overlapping
                if len(np.where(BestTwoDiff < 1e-3)) > 2/3 * nMaxT:
                    # choose the smoothest one
                    PhaseVDisp = PhaseVDisp_try[II[0][index2], :]
                else:
                    # choose the smaller difference one if 2/3 dispersion are different
                    PhaseVDisp = PhaseVDisp_try[II[0][index1], :]

        NewDisper = np.stack((self.TPoint, PhaseVDisp[0], np.ones(self.NumCtrT))).T

        # whether NewDisper falls in Phase V range or not
        for ii in range(self.NumCtrT):
            if NewDisper[ii, 1] > cRef_high[ii] or NewDisper[ii, 1] < cRef_low[ii]:
                NewDisper[ii, 2] = 0

        # find the phase velocity corresponding to the maximum amplitude of the envelope at each period
        PhaseVMaxAmp = np.max(self.PhaseVelocityImg, axis=0)
        JJ = abs(PhaseVMaxAmp - PhaseVDisp[0]) < 0.01
        # if the picked dispersion is very different from the dispersion corresponding to the maximum amplitude of the envelope (only 1/10 points are overlapping), not save the dispersion curve and will not pick phase velocity dispersion curve by setting IsDispGood = False
        if len(JJ) < 0.1 * self.NumCtrT:
            IsDispGood = False
        else:
            # No SNRIndex, skip SNR check
            if not hasattr(self, 'SNRIndex'):
                IsDispGood = True
                logger.warning('No SNRIndex, skip SNR check')
            else:
                # find reasonable dispersion points with high SNR
                GoodIndex = NewDisper[:, 2] + self.SNRIndex
                II = np.where(GoodIndex == 2)

                # save dispersion data when having at least 4 or 0.1*self.NumCtrT good points
                if len(II[0]) >= 4 or len(II[0]) >= 0.1 * self.NumCtrT:
                    # save dispersion data
                    IsDispGood = True
                else:
                    IsDispGood = False
        self.PhaseDisperCurve = PhaseVDisp[0]
        return IsDispGood, self.PhaseDisperCurve

    def SaveGroupDisper(self, save_path=r'./Disper'):
        '''
        save dispersion data
        '''
        if not hasattr(self, 'GroupDisperCurve'):
            logger.warning('No dispersion curve to save')
            return
        # Separate original path and file name
        origin_path, origin_filename = os.path.split(self.DataFileName)
        save_name = 'GDisp.' + origin_filename
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        FileName = os.path.join(save_path, save_name)
        with open(FileName, 'w') as f:
            f.write(f'{self.Longitude_A}    {self.Latitude_A}\n')
            f.write(f'{self.Longitude_B}    {self.Latitude_B}\n')
            for i in range(self.NumCtrT):
                wavelength = self.GroupDisperCurve[i] * self.TPoint[i]
                # TODO:第三列信噪比
                if self.StaDist >= self.minlamdaRatio * wavelength:
                    f.write(f'{self.TPoint[i]:.1f}    {self.GroupDisperCurve[i]:.3f}    {0:.3f}    {1:d}\n')
                else:
                    f.write(f'{self.TPoint[i]:.1f}    {self.GroupDisperCurve[i]:.3f}    {0:.3f}    {0:d}\n')


    def SavePhaseDisper(self, save_path=r'./Disper'):
        '''
        save dispersion data
        '''
        if not hasattr(self, 'PhaseDisperCurve'):
            logger.warning('No dispersion curve to save')
            return
        # Separate original path and file name
        origin_path, origin_filename = os.path.split(self.DataFileName)
        save_name = 'CDisp.' + origin_filename
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        FileName = os.path.join(save_path, save_name)
        with open(FileName, 'w') as f:
            f.write(f'{self.Longitude_A}    {self.Latitude_A}\n')
            f.write(f'{self.Longitude_B}    {self.Latitude_B}\n')
            for i in range(self.NumCtrT):
                wavelength = self.PhaseDisperCurve[i] * self.TPoint[i]
                if self.StaDist >= self.minlamdaRatio * wavelength:
                    f.write(
                        f'{self.TPoint[i]:.1f}    {self.PhaseDisperCurve[i]:.3f}    {0:.3f}    {1:d}\n')
                else:
                    f.write(
                        f'{self.TPoint[i]:.1f}    {self.PhaseDisperCurve[i]:.3f}    {0:.3f}    {0:d}\n')


    @staticmethod
    def GenerateSignalWindow(StartWin, EndWin, PtNum, Alpha=0.1):
        '''
        generate window function, scaling to length of PtNum

        args:
            StartWin: start index for the value of 1 in the window function
            EndWin: end index for the value of 1 in the window function
            PtNum: length of the original data
            Alpha: the proportion of cosine part to the whole window

        return:
            Window: window function
            TaperLen: the width of one side of the cosine function
        '''
        # window length
        win_len = int((EndWin - StartWin)/(1-Alpha)) + 1
        # generate window function
        Window = windows.tukey(win_len, Alpha)
        TaperLen = round(win_len * Alpha / 2)
        # crop or add the left side
        pad_left_len = StartWin - TaperLen
        if pad_left_len > 0:
            Window = np.pad(Window, (pad_left_len, 0), 'constant')
        else:
            Window = Window[-pad_left_len:]
        # crop or add the right side
        if Window.shape[0] < PtNum:
            Window = np.pad(
                Window, (0, PtNum - Window.shape[0]), 'constant')
        else:
            Window = Window[:PtNum]
        
        return Window, TaperLen

    @staticmethod
    def EnvelopeImageCalculation(WinWave, fs, TPoint, StaDist):
        '''
        calculate envelope image, i.e., to obtain envelope at each T
        new code for group velocity analysis using frequency domain Gaussian filter
        '''
        # linear interpolation
        alfa_x = [0,100,250,500,1000,2000,4000,20000]
        alfa_y = [5, 8, 12, 20, 25, 35, 50, 75]
        guassalfa = np.interp(StaDist, alfa_x, alfa_y)
        logger.debug(f'guassalfa: {guassalfa}')

        NumCtrT = TPoint.shape[0]
        PtNum = WinWave.shape[0]

        nfft = int(2 ** math.ceil(np.log2(max(PtNum, 1024*fs))))
        xxfft = fft(WinWave, nfft)
        fxx = np.asarray(range(nfft // 2 + 1)) / float(nfft) * fs

        EnvelopeImage = np.zeros((NumCtrT, PtNum))
        for i in range(NumCtrT):
            CtrT = TPoint[i]
            fc = 1/CtrT
            Hf = np.exp(-guassalfa*(fxx - fc) ** 2 / fc ** 2)
            yyfft = xxfft[:nfft // 2 + 1] * Hf
            yyfft = np.append(yyfft, np.conj(yyfft[-2:0:-1]))

            yy = np.real(ifft(yyfft, nfft))
            filtwave = abs(hilbert(yy))
            EnvelopeImage[i,:] = filtwave[0:PtNum]
        return EnvelopeImage

    @staticmethod
    def AutoSearch(InitialY, InitialX, ImageData):
        '''
        Automatically search arrival time line on a image
        '''
        InitialY = int(InitialY)
        InitialX = int(InitialX)
        YSize = ImageData.shape[0]
        XSize = ImageData.shape[1]
        ArrPt = np.zeros(XSize)
        # Center_T search up
        step = 3
        # step = 1
        point_left = int(0)
        point_right = int(0)
        for i in range(InitialX,XSize):
            index1 = 0
            index2 = 0
            point_left = int(InitialY)
            point_right = int(InitialY)
            while index1 == 0:
                point_left_new = max(0, point_left - step)
                if ImageData[point_left, i] < ImageData[point_left_new, i]:
                    point_left = point_left_new
                else:
                    index1 = 1
                    point_left = point_left_new
            while index2 == 0:
                point_right_new = min(point_right + step, YSize - 1)
                if ImageData[point_right, i] < ImageData[point_right_new, i]:
                    point_right = point_right_new
                else:
                    index2 = 1
                    point_right = point_right_new
            index_max = np.argmax(ImageData[point_left:point_right, i])
            ArrPt[i] = index_max + point_left
            InitialY = ArrPt[i]
        # Center_T search down
        InitialY = ArrPt[InitialX]
        for i in range(InitialX)[::-1]:
            index1 = 0
            index2 = 0
            point_left = int(InitialY)
            point_right = int(InitialY)
            while index1 == 0:
                point_left_new = max(0, point_left - step)
                if ImageData[point_left, i] < ImageData[point_left_new, i]:
                    point_left = point_left_new
                else:
                    index1 = 1
                    point_left = point_left_new
            while index2 == 0:
                point_right_new = min(point_right + step, YSize - 1)
                if ImageData[point_right, i] < ImageData[point_right_new, i]:
                    point_right = point_right_new
                else:
                    index2 = 1
                    point_right = point_right_new
            index_max = np.argmax(ImageData[point_left:point_right, i])
            ArrPt[i] = index_max + point_left
            InitialY = ArrPt[i]
        return ArrPt

    def LoadRefGroupDisper(self,path,deltaPhaseV=0.5):
        # load group velocity reference dispersion and ranges
        try:
            raw = np.loadtxt(path)
            refgdisp = np.copy(raw)
            if raw.shape[1] == 3:
                # Second column minus third column
                refgdisp[:, 1] = raw[:, 1] - raw[:, 2]
                # Second column plus third column
                refgdisp[:, 2] = raw[:, 1] + raw[:, 2]
            elif raw.shape[1] == 2:
                refgdisp[:, 1] = raw[:, 1] - deltaPhaseV
                refgdisp[:, 2] = raw[:, 1] + deltaPhaseV
            else:
                logger.error(
                    'group velocity reference should have 2 or 3 columns')
        except:
            logger.error(f'Fail to load data from {FilePath}')
            raise
        self.refgroupdisp = refgdisp

    def LoadRefPhaseDisper(self, path, deltaPhaseV=0.5):
        # load phase velocity reference dispersion and ranges
        try:
            raw = np.loadtxt(path)
            refgdisp = np.copy(raw)
            if raw.shape[1] == 3:
                # Second column minus third column
                refgdisp[:, 1] = raw[:, 1] - raw[:, 2]
                # Second column plus third column
                refgdisp[:, 2] = raw[:, 1] + raw[:, 2]
            elif raw.shape[1] == 2:
                refgdisp[:, 1] = raw[:, 1] - deltaPhaseV
                refgdisp[:, 2] = raw[:, 1] + deltaPhaseV
            else:
                logger.error('phase velocity reference should have 2 or 3 columns')
        except:
            logger.error(f'Fail to load data from {FilePath}')
            raise
        self.refphasedisp = refgdisp

if __name__ == '__main__':
    # FilePath = "CF.dat"
    # gfcn = gfcn_analysis(FilePath, isEGF=False, DeltaV=0.005, DeltaT=1)
    
    # FilePath = "ZZ_0101-2901_82d.dat"
    FilePath = "ZZ_0101-2917_81d.dat"
    gfcn = gfcn_analysis(FilePath, isEGF=False,StartV=1,EndV=4,StartT=0.6,EndT=10,DeltaT=0.1, DeltaV=0.02)

    # FilePath = "FD01_FD16.dat"
    # gfcn = gfcn_analysis(FilePath, isEGF=False,StartV=0.5,EndV=4,StartT=0.2,EndT=5,DeltaT=0.1, DeltaV=0.005)

    gfcn.LoadRefGroupDisper('ref_RayGroupVdisp.txt')
    gfcn.LoadRefPhaseDisper('ref_RayPhaseVdisp.txt')

    gfcn.GroupVelocityImgCalculate()
    gfcn.AutoGroupDisperPick()
    gfcn.SaveGroupDisper()

    gfcn.PhaseVelocityImgCalculate(TimeVariableFilter=gfcn.TimeVariableFilterType.obs)
    gfcn.AutoPhaseDisperPick()
    gfcn.SavePhaseDisper()

    gfcn.GroupVelocityImgPlot()
    gfcn.PhaseVelocityImgPlot()
