%% preparation

MaxEspectedResp=[0.02]; %set this to max expected response
MaxTrials=96; %no of algo runs

% dbgen=[63 84 15 63 84 15 1 2 3 4 5:32];

kappa=MaxEspectedResp*6;%1.3*10^-4; %(2 or 3 or 6 times the maximum expected response)
NoiseEstim=1e-04; %I thought it was not important
NumElectrodes=96;
rho=4; %fixed for array topology
%Krash
map = [[100 96:-1:89 100]' (88:-1:79)' (78:-1:69)' (68:-1:59)' (58:-1:49)'...
    (48:-1:39)' (38:-1:29)' (28:-1:19)' (18:-1:9)' [100 8:-1:1 100]' ];
map2 = [[1 96:-1:89 1]' (88:-1:79)' (78:-1:69)' (68:-1:59)' (58:-1:49)'...
    (48:-1:39)' (38:-1:29)' (28:-1:19)' (18:-1:9)' [1 8:-1:1 1]' ];
%Babko
% map = [[100 96:-1:89 81]' [88:-1:82 100 80 79]' [78:-1:69]'...
%     [68:-1:59]' [58 57 56 100 54:-1:49]' [48:-1:39]' [38:-1:29]' [28:-1:19]' [18:-1:9]' [55 8:-1:4 100 2 1 3]'] ;
% map2 = [[1 96:-1:89 81]' [88:-1:82 1 80 79]' [78:-1:69]'...
%     [68:-1:59]' [58 57 56 1 54:-1:49]' [48:-1:39]' [38:-1:29]' [28:-1:19]' [18:-1:9]' [55 8:-1:4 1 2 1 3]'] ;
% mappazzone1=[31:-2:25 32:-2:18 23:-2:9 16:-2:2 7:-2:1];
% mappazzone2=[32:-2:2 31:-2:1];
% mappazzone3=[32:-2:2 31:-2:1];
mappazzone1=[1:2:7 2:2:16 9:2:15 17:2:23 18:2:32 25:2:31];
mappazzone2=[1:2:31 2:2:32];
mappazzone3=[1:2:31 2:2:32];
mappazzone=[mappazzone1 mappazzone2+32 mappazzone3+64]; %some remapping due to connectors
test_respo=0;
mat_turn=0;

%calc kernel
dist = zeros(96);
distKern = dist;
for x = 1:96
    for y = 1:96        
        [x1,y1] = find(map == x);
        [x2,y2] = find(map == y);        
        dif1 = (x1-x2)^2;
        dif2 = (y1-y2)^2;        
        dist(x,y) = sqrt(dif1 + dif2);        
        distKern(x,y) = (1+(sqrt(5)*dist(x,y))/rho + (5*(dist(x,y))^2)/(3*rho^2))*exp(-sqrt(5)*dist(x,y)/rho);
    end
end

Ccolor{1}='blue';
Ccolor{2}='purple';
Ccolor{3}='red';

%%
obj = TDEV(); %connect to TDT

%% running Ctrl + Enter
clear P_test
clear P_respo
obj.write('NoTrials', MaxTrials,'DEVICE','EMG_'); %synchro with TDT
obj.write('TDTTurn', 0,'DEVICE','EMG_'); %synchro with TDT
q=1;
% sampFreq=obj.FS{1}; 

while q <= MaxTrials    
    
    mat_turn=obj.read('MatlabTurn','DEVICE','EMG_');
        if mat_turn==q    
        if q>1
            AcquisitionMap = MapPrediction + kappa.*real(sqrt(VarianceMap));    
            Next_Elec = find(ismember(AcquisitionMap, max(AcquisitionMap))); 
            if length(Next_Elec) > 1
                Next_Elec = Next_Elec(randi(numel(Next_Elec)));
            end
            P_test(q,1) = Next_Elec; 
        else
            P_test(q,1) = randi(NumElectrodes); 
            K_maj=[];
        end
        
        % few next lines just sending TDT channel and reading response
        
%         x = input(['will test channel ', num2str(P_test(q,1)), ', cable: ', num2str(ceil(P_test(q,1)/32)), ', section: ', num2str((mod(mappazzone(P_test(q,1)),32)>16)+1), ' (TDT ch ',num2str(mappazzone(P_test(q,1))),') , press Enter']);
        disp(['will test channel ', num2str(P_test(q,1)), ', cable: ', num2str(ceil(P_test(q,1)/32)), ', section: ', num2str((mod(mappazzone(P_test(q,1)),32)>16)+1), ' (TDT ch ',num2str(mappazzone(P_test(q,1))),') , press Enter']);
%         obj.write('MatlabOutput', 33+mod(q,2)*32,'DEVICE','EMG_');
%         obj.write('MatlabOutput', dbgen(q),'DEVICE','EMG_');
        obj.write('MatlabOutput', mappazzone(P_test(q,1)),'DEVICE','EMG_');
        this_pulse=obj.read('PulseCnt2','DEVICE','EMG_');
        this_pulse2=obj.read('PulseCnt2','DEVICE','EMG_');
        while this_pulse2==this_pulse;
        this_pulse2=obj.read('PulseCnt2','DEVICE','EMG_');
        end   
        
        obj.write('TDTTurn', q,'DEVICE','EMG_');  
        pause(0.5) 
        test_respo=obj.read('MatlabMonitor','DEVICE','EMG_');
        disp(['response: ', num2str(test_respo)])
        
        % done reading response
        
        P_respo(q,:)=test_respo;
        P_test(q,2)=test_respo;
        [MapPrediction,VarianceMap,K_maj] = CalcPrediction(P_test,NoiseEstim,distKern,K_maj,NumElectrodes);  
        Good_Elec=find(ismember(MapPrediction, max(MapPrediction)));
        if length(Good_Elec) > 1
            Good_Elec = Good_Elec(randi(numel(Good_Elec)));
        end
        P_max(q)= Good_Elec; 
    q=q+1;      
    end 
    pause(0.010) 
end

while mat_turn~=0
    mat_turn=obj.read('MatlabTurn','DEVICE','EMG_');
end
obj.write('TDTTurn', 0,'DEVICE','EMG_');


save(['search_output_', date, '_', datestr(now,'HHMMSS'), '.mat' ])

%% Inner function

function [NEWMEAN,NEWVAR,K_maj] = CalcPrediction(PERFORMANCE,NoiseEstim,distKern,K_maj,NUMELMAP)
% Takes the performance matrix, the prior prediction and the prior
% variance map and finds the new prediction and variance matrices.
    t = size(PERFORMANCE,1);
    AddedRow = distKern(PERFORMANCE(end,1),PERFORMANCE(1:(end),1));
    K_maj(t,1:t) = AddedRow;
    K_maj(1:t,t) = AddedRow;
    K_maj_n = K_maj + eye(t)*NoiseEstim;
    KInv = K_maj_n^-1;
    NEWMEAN = zeros(1,NUMELMAP);
    NEWVAR = NEWMEAN;
    for l = 1:NUMELMAP    
        k_min = distKern(l,PERFORMANCE(:,1));    
        NEWMEAN(l) = k_min*KInv*PERFORMANCE(:,2);
        NEWVAR(l) = 1 - k_min*KInv*(k_min)';
    end
end