  
AllBB=[1:3];
load('pairedBurst_pilotdata.mat')
%load('mKrash181029_protocolISI_data.mat')


emg_lim=[ 3e-4 1e-4 3e-5 3e-4 2e-4 2.5e-4 2e-4];
close all
%which_EMG=5;
raw_resp=cell(3,7,9);
for bb=1:numel(AllBB) 
    listcond=unique(P3{AllBB(bb)});
    for which_EMG=1:7
        for a_i=1:numel(listcond)
            raw_resp{bb,which_EMG,a_i}=KT{AllBB(bb),which_EMG}(find(P3{AllBB(bb)}==listcond(a_i)),505:1237);
        end
    end
    
    figure
    for a_i=1:numel(listcond)
        subplot(3,3,a_i)
        plot(mean(raw_resp{bb, which_EMG, a_i},1))
        xticks([1 245 489 732])
        xticklabels([-0.05:0.05:0.1])
        ylim([0 emg_lim(which_EMG)])
    end
end

%-------------------------------------%
%Filter parameters

% From marco's email
sampling_freq = 4882.8;
[B,A] = butter(5,[40 500]/(0.5*sampling_freq),'bandpass');
[B60,A60] = butter(2,[59.9 60.1]/(0.5*sampling_freq),'stop');

%Filter Application
WINDOW_LENGTH=100;
OVERLAP=50;
DELTA = WINDOW_LENGTH - OVERLAP;
for block =  1:3
    for muscle = 1:7
        for cond = 1:9
            for stim = 1:20
                ts = raw_resp{block,muscle,cond};
                ts(stim,:) = filtfilt(B,A,double(ts(stim,:)));
                ts(stim,:) = filtfilt(B60,A60,double(ts(stim,:)));
                filtered(stim,:) = rms(abs(ts(stim,:)),WINDOW_LENGTH,OVERLAP,0);
            end
            raw_resp{block,muscle,cond} = ts;
            filt_resp{block,muscle,cond} = filtered;
        end
    end
end

%-----------------------------------------%

save('RawMonkeyEmgResponse.mat', 'raw_resp');
save('FilteredMonkeyEmgResponse.mat', 'filt_resp');