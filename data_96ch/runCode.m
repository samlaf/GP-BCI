load('randomRuns_Krash.mat')
MaxTrials=96; %no of algo runs

% dbgen=[63 84 15 63 84 15 1 2 3 4 5:32];

NumElectrodes=96;
rho=4; %fixed for array topology
%Krash
map = [[100 96:-1:89 100]' (88:-1:79)' (78:-1:69)' (68:-1:59)' (58:-1:49)'...
    (48:-1:39)' (38:-1:29)' (28:-1:19)' (18:-1:9)' [100 8:-1:1 100]' ];
map2 = [[1 96:-1:89 1]' (88:-1:79)' (78:-1:69)' (68:-1:59)' (58:-1:49)'...
    (48:-1:39)' (38:-1:29)' (28:-1:19)' (18:-1:9)' [1 8:-1:1 1]' ];
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

clear MP
syn=2;
for i=1:10
    MP(ALL_PTEST{syn,i}(:,1),i)=ALL_PTEST{syn,i}(:,2);
    Pmaxrand(i,1)=ALL_PTEST{syn,i}(1,1);
    randmax=ALL_PTEST{syn,i}(1,2);
    for t=2:96
        if ALL_PTEST{syn,i}(t,2)>randmax
            randmax=ALL_PTEST{syn,i}(t,2);
            Pmaxrand(i,t)=ALL_PTEST{syn,i}(t,1);
        else
            Pmaxrand(i,t)=Pmaxrand(i,t-1);
        end
    end        
end
MPm=mean(MP')';
mMpm=max(MPm);

nRep=50;


%%
NoiseEstim=[1e-8 0.05:0.05:0.5];

perf=cell(numel(NoiseEstim),1);
for n_i=1:numel(NoiseEstim)

    for rep_i=1:nRep
    %%
        MaxSeenResp=0;
        clear P_test
        clear P_respo
        q=1;

        while q <= MaxTrials      
                if q>1
                    MaxSeenResp=max(P_test(:,2));
                    kappa=abs(MaxSeenResp)*5;
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
                r_i=randi(10);
                test_respo=MP(P_test(q,1),r_i);
                    % done reading response
                P_respo(q,:)=test_respo;
                P_test(q,2)=test_respo;
                [MapPrediction,VarianceMap,K_maj] = CalcPrediction(P_test,NoiseEstim(n_i),distKern,K_maj,NumElectrodes);
                % Why use these 3 next lines instead of the commented
                % below? We only test for gp predictions at electrodes that
                % we had queried
                Tested=unique(sort(P_test(:,1)));
                MapPredictionTested=MapPrediction(Tested);
                Good_Elec=Tested(find(ismember(MapPredictionTested, max(MapPredictionTested))));
                %Good_Elec=find(ismember(MapPrediction, max(MapPrediction)));
                if length(Good_Elec) > 1
                    Good_Elec = Good_Elec(randi(numel(Good_Elec)));
                end
                P_max(q)= Good_Elec; 
            q=q+1; 
        end
        
       perf{n_i}(rep_i,:)=MPm(P_max)/mMpm;

    end

end
%%
hold on
for i=1:numel(NoiseEstim)
    plot(mean(perf{i}), 'Color', [(i-1)/(numel(NoiseEstim)-1) 0 (numel(NoiseEstim)-i)/(numel(NoiseEstim)-1)])
end
% plot(mean(MPm(Pmaxrand))/mMpm,'k')




%%
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