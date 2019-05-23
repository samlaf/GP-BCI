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
% Here we create the datastructure MP (Map Predictions)
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
mMPm=max(MPm);

%%
% We create the kernel
% We put a box prior on the two lengthscale params
% note that params are in log scale, so we need (log(1), log(3))
priorbox = {@priorSmoothBox1,log(1),log(3),100};
prior.cov = {priorbox; priorbox; []};
% We also
prior.lik = {{@priorSmoothBox1, log(1e-7), log(1e-2), 100}};
infprior = {@infPrior,@infGaussLik,prior};
infm = @infGaussLik;
meanf = @meanConst;
covf = {@covMaternard, 5};
likf = @likGauss;
hyp = struct('mean', [], 'cov', log([1 1 0.005]), 'lik', log(0.005));
% We also learn the kernel in advance to compare to sequential learning
y = MP(:);
x = repmat(ch2xy,10,1);
hypoMaternard = minimize(hyp, @gp, -50, infprior, [], covf, likf, x, y);

%%
% Then we run the sequential optimization
nRep=10;
MaxTrials = 96;
nrnd = 10; % #of rnd pts to ask at the beginning
perf=[];
NumElectrodes = 96;
P_test = cell(nRep,1);
dynamicKappa = false;
%kappas = logspace(1,-2,96);
kappas = ones(1,96)*3;
hyperparams = cell(nRep,MaxTrials);

for rep_i=1:nRep
    
    MaxSeenResp=0;
    q=1;

    while q <= MaxTrials      
        if q>nrnd
            if dynamicKappa
                MaxSeenResp=max(P_test{rep_i}(:,2));
                kappa=abs(MaxSeenResp)*5;
                kappas(q) = kappa;
            else
                kappa = kappas(q);
            end
            % Find next point (max of acquisition function)
            % I believe the equivalent to Marco's code is using fmu and
            % fs2, and not ymu and ys2
            AcquisitionMap = fmu + kappa.*real(sqrt(fs2));    
            Next_Elec = find(ismember(AcquisitionMap, max(AcquisitionMap))); 
            if length(Next_Elec) > 1
                Next_Elec = Next_Elec(randi(numel(Next_Elec)));
            end
            P_test{rep_i}(q,1) = Next_Elec; 
        else 
            P_test{rep_i}(q,1) = randi(NumElectrodes); 
        end        
        r_i=randi(10);
        query_elec = P_test{rep_i}(q,1);
        test_respo=MP(query_elec,r_i);
        % done reading response
        P_test{rep_i}(q,2)=test_respo;
        x = ch2xy(P_test{rep_i}(:,1),:);
        y = P_test{rep_i}(:,2);
        hyp = minimize(hyp, @gp, -20, infprior, [], covf, likf, x, y);
        hyperparams{rep_i,q} = hyp;
        [ymu ys2 fmu fs2] = gp(hyp, infm, [], covf, likf, x, y, ch2xy);
        % We only test for gp predictions at electrodes that
        % we had queried (presumable we only want to return an
        % electrode that we have already queried... though this is
        % debatable. (so we comment the next line and replace with
        % these 3)
        Tested=unique(sort(P_test{rep_i}(:,1)));
        MapPredictionTested=ymu(Tested);
        Good_Elec=Tested(find(ismember(MapPredictionTested, max(MapPredictionTested))));
        % Good_Elec=find(ismember(MapPrediction, max(MapPrediction)));
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
plot(mean(perf));
hold on;
plot(mean(MPm(Pmaxrand))/mMPm,'k');
figure();
plotSeq(P_test{1}, MPm, hyperparams(1,:), kappas);


%% Scripting
for i=1:nRep
    disp(P_test{i}(end,1));
end