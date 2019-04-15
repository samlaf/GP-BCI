% load paired trains
load('sKrash_190207_protocolPairedTrainsDiag')

% We first change PARA to get what we want
% 1) si PARA{n}(2,t)==0, le test t est un single train de 40ms.
% 2) si PARA{n}(2,t)==PARA{n}(3,t), les trains A et B sont fait 
% sur le même channel (disponible seulement avec délais >=40).
%%% Le point 1) est inutile... on devrait juste mettre
%%% PARA{n}(2,t)==PARA{n}(3,t) comme le point 2) puisque le dt=0 va nous
%%% indiquer que c'est un single train
% On fait donc ce changement ici:
EXPS = [3 4 6 7 8 9 11 12 13 14];
for exp = EXPS
    idxs = find(PARA{exp}(2,:) == 0);
    PARA{exp}(2,idxs) = PARA{exp}(3,idxs);
end

close all
N_EMGS=7;
N_CHS=10;
DT=[0, 10, 20, 40, 60, 80, 100];
N_DT=length(DT);
CHS = [2 6 9 10 13 14 17 18 21 22];
raw_resp=cell(N_EMGS,N_CHS,N_CHS,N_DT);
for expidx=1:length(EXPS)
    for trial=1:645
        exp = EXPS(expidx);
        dt = PARA{exp}(1,trial);
        ch1 = PARA{exp}(2,trial);
        ch2 = PARA{exp}(3,trial);
        ch1idx = find(ch1==CHS);
        ch2idx = find(ch2==CHS);
        dtidx = find(dt==DT);
        for emg=1:N_EMGS
            resp = KT{exp,emg}(trial,:);
            raw_resp{emg,ch1idx,ch2idx,dtidx} = [raw_resp{emg,ch1idx,ch2idx,dtidx}; resp];
        end
    end
end
%%
%-------------------------------------%
%%Filter parameters

% From marco's email
sampling_freq = 4882.8;

[B,A] = butter(5,[40 500]/(0.5*sampling_freq),'bandpass');
[B60,A60] = butter(2,[59.9 60.1]/(0.5*sampling_freq),'stop');

%Filter Application
WINDOW_LENGTH=100;
OVERLAP=50;
DELTA = WINDOW_LENGTH - OVERLAP;
for emg=1:N_EMGS
    for ch1=1:N_CHS
        for ch2=1:N_CHS
            for dt=1:N_DT
                ts = raw_resp{emg,ch1,ch2,dt};
                for stim = 1:size(ts,1)                  
                    ts(stim,:) = filtfilt(B,A,double(ts(stim,:)));
                    ts(stim,:) = filtfilt(B60,A60,double(ts(stim,:)));
                    filtered(stim,:) = rms(abs(ts(stim,:)),WINDOW_LENGTH,OVERLAP,0);
                    gfiltered(stim,:) = filter(gausswin(WINDOW_LENGTH),1,abs(ts(stim,:)));
                end
                raw_resp{emg,ch1,ch2,dt} = ts;
                filt_resp{emg,ch1,ch2,dt} = filtered;
                gfilt_resp{emg,ch1,ch2,dt} = gfiltered;
            end
        end
    end
end
%%
%-----------------------------------------%

%save('RawCebusEmgResponse.mat', 'raw_resp');
save('FilteredPairedTrains.mat', 'gfilt_resp');
