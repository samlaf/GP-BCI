function [DataAllF] = EmgAnalysisBasic(BlockData)
%EmgAnalysis Filter and analyse the EMG response in a datablock.mat file,
% then save them to a matlab file to be loaded into python pandas.
%
%    Input : Structure obtained from the block of the TDT
%
%    Output: Structure containing:
%    - Time of stimulation
%    - Channel of stimulation
%    - Time series of EMG response 200ms after stimulation for each channel
%    (1xnum_channels cell)

%% Initialisation

Freq_Stim = BlockData.streams.STM_.fs;
Freq_Emg = BlockData.streams.EMGr.fs;

Data_Stim = BlockData.streams.STM_.data(3,:);
Data_Emg_Raw = BlockData.streams.EMGr.data([1 2 3 5 6 7 9 10],:);

Onset_Stim = BlockData.epocs.Ch1_.onset';
ChannelOfStim = BlockData.epocs.Ch1_.data';

NUM_CHANS = size(Data_Emg_Raw,1);
NUM_STIMS = size(Onset_Stim,2);

%Filter parameters

[B,A] = butter(5,[40 500]/(0.5*Freq_Emg),'bandpass');
[B60,A60] = butter(2,[59.9 60.1]/(0.5*Freq_Emg),'stop');

%Filter Application
WINDOW_LENGTH=100;
OVERLAP=50;
DELTA = WINDOW_LENGTH - OVERLAP;
for chan =  1:NUM_CHANS
    DataEMG_F405(chan,:) = filtfilt(B,A,double(Data_Emg_Raw(chan,:)));
    DataEMG_F405(chan,:) = filtfilt(B60,A60,DataEMG_F405(chan,:));
    Data_Filt(chan,:) = rms(abs(DataEMG_F405(chan,:)),WINDOW_LENGTH,OVERLAP,0);
    Freq_Rms(chan) = Freq_Emg/OVERLAP;
end

%% Separation by Stimulation

% Save time and channel of stimulation
DataAllF.Time = Onset_Stim;
DataAllF.StimChan = ChannelOfStim;
    
for StimNum = 1:NUM_STIMS
    for chan = 1:8
        
        % Data acquisition for the wanted timeframe.        
        % The timeframe for the response is 50ms after the beginning of the
        % stimulation and end 100ms later
        % We instead keep a window of [-150,150]ms around the stimulation time
        
        %TimeWindow = [Onset_Stim(StimNum)-0.150 Onset_Stim(StimNum)+0.150];
        %EmgWindow = (floor(TimeWindow(1)*Freq_Rms(chan)):ceil(TimeWindow(2)*Freq_Rms(chan)));
        %RelevantEmg = Data_Filt(chan,EmgWindow);
        
        % The frequency of the rms filtered channel is roughly 100
        % (97.65..)
        % So we find the onset stim, and keep 15 points before and 15
        % points after that (which will be roughly [-150,150]ms
        OnsetIdx = floor(Onset_Stim(StimNum)*Freq_Rms(chan));
        EmgWindow = [OnsetIdx-15:OnsetIdx+15];
        RelevantEmg = Data_Filt(chan,EmgWindow);
        
        % Save the data
        DataAllF.(sprintf('chan%g',chan)){StimNum} = RelevantEmg;
        
    end
    
end