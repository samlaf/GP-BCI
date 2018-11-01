%% Obtaining EMG responses

NumOfBlock = 1; % How many blocks do you want to open?

for block = 1:NumOfBlock
    base = sprintf('FRQNT_ICMS_Aigu/FRQNT_1/Block--%g',block);
    DataBlock = TDTbin2mat(base);
    
    % Removing bad data (first 2 impulses at 0.2 and 0.7 seconds)
    % Real spikes are usually intertwined by same period (roughly 1s)
    
    MeanTimeStim = mean(diff(DataBlock.epocs.Ch1_.onset));
    while any(diff(DataBlock.epocs.Ch1_.onset) > 2*MeanTimeStim)
        DataBlock.epocs.Ch1_.onset = DataBlock.epocs.Ch1_.onset(2:end);
        DataBlock.epocs.Ch1_.offset = DataBlock.epocs.Ch1_.offset(2:end);
        DataBlock.epocs.Ch1_.data = DataBlock.epocs.Ch1_.data(2:end);
        DataBlock.epocs.Amp1.data = DataBlock.epocs.Amp1.data(2:end);
    end
    
    save(sprintf('DataBlock_%g.mat',block),'DataBlock')
    
    EmgResponse{block}=EmgAnalysisBasic(DataBlock);
end

save('EmgResponse.mat', 'EmgResponse');