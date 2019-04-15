% We build MP from ALL_PTEST, which will be a 96x10 array
% 96 channels with 10 responses each
load('randomRuns_Krash.mat')

% maps from 2d position to channel number and vice versa
% Notice we put '1' on the corners (this is just for ease of plotting
% later)
xy2ch = [[1 96:-1:89 1]' (88:-1:79)' (78:-1:69)' (68:-1:59)' (58:-1:49)'...
    (48:-1:39)' (38:-1:29)' (28:-1:19)' (18:-1:9)' [1 8:-1:1 1]' ];
xy2ch2 = [[100 96:-1:89 100]' (88:-1:79)' (78:-1:69)' (68:-1:59)' (58:-1:49)'...
    (48:-1:39)' (38:-1:29)' (28:-1:19)' (18:-1:9)' [100 8:-1:1 100]' ];
ch2xy = zeros(96,2);
for i = 1:96
    [y,x] = find(xy2ch2==i);
    ch2xy(i,:) = [x,y];
end

%%
% Here we create the datastructure MP
clear MP
SYN=3;
for i=1:10
    MP(ALL_PTEST{SYN,i}(:,1),i)=ALL_PTEST{SYN,i}(:,2);      
end
MPm=mean(MP')';
mMPm=max(MPm);

%%
% We create the kernel
infm = @infGaussLik;
meanf = [];
covf = @covSEard;
likf = @likGauss;
hyp = struct('mean', [], 'cov', log([1 1 1]), 'lik', log(0.01));
% We also learn the kernel in advance to compare to sequential learning
y = MP(:);
x = repmat(ch2xy,10,1);
hypoSEard = minimize(hyp, @gp, -100, infm, meanf, covf, likf, x, y);

%%
% Then we run the sequential optimization
nRep=20;
MaxTrials = 96;
perf=[];
NumElectrodes = 96;
P_test = cell(nRep,1);
kappas = logspace(1,-2,96);

for rep_i=1:nRep
    
    MaxSeenResp=0;
    q=1;

    while q <= MaxTrials      
        if q>1
            % Set kappa dynamically
            %MaxSeenResp=max(P_test{rep_i}(:,2));
            %kappa=abs(MaxSeenResp)*5;
            kappa = kappas(q);
            % Find next point (max of acquisition function)
            AcquisitionMap = ymu + kappa.*real(sqrt(ys2));    
            Next_Elec = find(ismember(AcquisitionMap, max(AcquisitionMap))); 
            if length(Next_Elec) > 1
                Next_Elec = Next_Elec(randi(numel(Next_Elec)));
            end
            P_test{rep_i}(q,1) = Next_Elec; 
        else 
            P_test{rep_i}(q,1) = randi(NumElectrodes); 
            K_maj=[];
        end        
        r_i=randi(10);
        test_respo=MP(P_test{rep_i}(q,1),r_i);
        % done reading response
        P_test{rep_i}(q,2)=test_respo;
        x = ch2xy(P_test{rep_i}(:,1),:);
        y = P_test{rep_i}(:,2);
        [ymu ys2 fmu fs2] = gp(hypoSEard, infm, meanf, covf, likf, x, y, ch2xy);
        % Why use these 3 next lines instead of the commented
        % below? We only test for gp predictions at electrodes that
        % we had queried
        Tested=unique(sort(P_test{rep_i}(:,1)));
        MapPredictionTested=ymu(Tested);
        Good_Elec=Tested(find(ismember(MapPredictionTested, max(MapPredictionTested))));
        %Good_Elec=find(ismember(MapPrediction, max(MapPrediction)));
        if length(Good_Elec) > 1
            Good_Elec = Good_Elec(randi(numel(Good_Elec)));
        end
        P_max(q)= Good_Elec; 
        q=q+1; 
    end

    perf(rep_i,:)=MPm(P_max)/mMPm;

end

%%
plot(perf.');
figure();
plotSeq(MPm, P_test{2}, hypoSEard, kappas);
% plot(mean(MPm(Pmaxrand))/mMPm,'k')

%% Scripting
for i=1:20
    P_test{i}(end,2);
end