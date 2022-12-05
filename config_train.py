import os
from EGFAnalysisTimeFreq import gfcn_analysis
GreenFcnObjectsType = gfcn_analysis.GreenFcnObjectsType
class Config(object):
    """ EGFAnalysisTimeFreq configuration.

    Attributes:
        isEGF: True for EGF, False for CF.
        StartT: Start time of the analysis.
        EndT: End time of the analysis.
        DeltaT: Time interval of the analysis.
        StartV: Start velocity of the analysis.
        EndV: End velocity of the analysis.
        DeltaV: Velocity interval of the analysis (fixed at 0.005 km/s).
        GreenFcnObjectsType: the type of Green’s function objects. 
                             (e.g., A_to_B, B_to_A, A_add_B).
        WinAlpha: the proportion of cosine part to the whole window. 
        NoiseTime: the time of noise sampling
        WinPeriodNum : number of window period
        WinMinTime : minimum of window time
        FilterKaiserPara : shape factor of Kaiser window
        MaxFilterLengthLog : base 2 logarithm of maximum value of fft point when using freq. domain, too high value will cause slow calculation
    Path:
        root: The path of EGFAnalysisTimeFreq.
        waveform_path： CF/EGF file path.
        result_path: Dispersion path.
    """


    """ DisperPicker configuration.
    
    Abbreviations:
        G: group velocity
        C: phase velocity
        T: period
        V: velocity

    Attributes:
        dT: Time interval of the analysis.
        dV: Velocity interval of the analysis (fixed at 0.005 km/s).
        range_T: Period range [start, end, num]
        range_V: Velocity range [start, end, num]
        input_size (int): Input size, auto calculated.
    
    Picking thresholds:
        confidence_G (float): Accept the points if G probability (output of DPNet) value is 
            larger than this parameter.
        mean_confidence_C (float): Accept the C curve if average C probability value is  
            larger than this parameter.
        confidence_C (float): Accept the C points if C probability value is larger than this.
        random_plot_ratio: Plot part of (e.g. 1%) the pick results. This is time consuming. 
            Please do not plot too much if it is not necessary. 

    Another config:
        min_len (int): Accept the dispersion curves if it's length (number of points) is  
            larger than this parameter.
        ref_T (int): Find the local maximum points in this column of C dispersion image. 
            This para can be set to 'None' to use the default value.
        ref_T2 ([int, int]): Use these columns to calculate the average probability of C curves.   
            This para can be set to [] to use the default value.
        train (bool): train DisperPicker or not.
        training_step (int): Training step.
        learning_rate (float): Learning rate.
        damping (float): Avoid over-fitting.
        test (bool): If you want to test the performance of DisperPicker, you can set  
            this para to True. You have to place the label (group_velocity and phase_velocity)
            in 'test_data_path' to test the DisperPicker, and when you run the pick.py, 
            DisperPicker will compare the reault with the label. 
            If you only want to use DisperPicker to pick dispersion curves, this should be False.

    Data path:
        test_data_path: Test data path.
        result_path: Result path.
        training_data_path: Training data path.
        validation_data_path: Validation data path.

    """

    def __init__(self):
        # =========    EGFAnalysisTimeFreq config    ========= #

        self.root = os.getcwd()
        # self.root = '/home/yang/Projects/EGFAnalysisTimeFreq_py/Feidong_test'
        self.waveform_path = self.root + '/FeidongCFs'
        # self.waveform_path = self.root + '/AllCFs'
        self.result_path = self.root + '/DisperPicker/result'
        self.isEGF = False
        self.StartV = 1
        self.EndV = 4
        self.StartT = 0.1
        self.EndT = 6
        self.DeltaT = 0.1
        self.DeltaV = 0.005
        self.GreenFcnObjects = GreenFcnObjectsType.A_add_B
        self.WinAlpha = 0.1
        self.NoiseTime = 150
        self.MinSNR = 5.0
        self.WinPeriodNum = 5
        self.WinMinTime = 25
        self.FilterKaiserPara = 6
        self.MaxFilterLengthLog = 14

        # ============    DisperPicker config    ============ #
        
        self.dT = self.DeltaT
        self.dV = self.DeltaV
        self.range_T = [self.StartT, self.EndT, 
                        round((self.EndT - self.StartT)/self.dT) + 1]    # [start, end, num]
        self.range_V = [self.StartV, self.EndV, 
                        round((self.EndV - self.StartV)/self.dV) + 1]    # [start, end, num]
        self.input_size = [self.range_V[2], self.range_T[2], 2]
        self.batch_size = 16
        
        # Picking thresholds (need fine-tuning)
        self.confidence_G = 0.6 
        self.mean_confidence_C = 0.4 
        self.confidence_C = 0 
        self.random_plot_ratio = 1 

        # Another config
        self.min_len = round(self.range_T[2]/5) 
        self.ref_T = None
        self.ref_T2 = [] 
        self.test = False  
        self.test_data_path = self.root + '/FDDisp'
        self.training_data_path = self.root + '/DisperPicker/data/TrainingData' 
        self.validation_data_path = self.root + '/DisperPicker/data/ValidationData' 

        # Training config
        self.train = True
        self.training_step = 1000
        self.learning_rate = 1e-3
        self.damping = 0.0
        self.radius = 20
