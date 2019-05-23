%% DATA LOADING + HELPER DATA STRUCTURE CREATION
% randomRuns_Krash.mat contains ALL_PTEST and map2
% ALL_Ptest (3x10 cell)
% .   3 synergies
%     10 "trials" (runs through all chs stimulating)
% map2 (10x10 array) -- electrode position on 10x10 array
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

%% BUILD MAIN DATA STRUCTURE + PLOTS
% We build MP (Map Predictions, 96x10 array) from ALL_PTEST
% 96 channels with 10 responses each
clear MP
SYN=2;
for i=1:10
    runi = ALL_PTEST{SYN,i};
    idxs = runi(:,1);
    values = runi(:,2);
    MP(idxs,i) =  values;  
end
% We normalize MP (can't normalize if we are going to do
% sequential-optimization since we haven't seen all the data)
%MP = (MP - mean(MP(:))) / std(MP(:));

% We plot mean responses
figure();
means = mean(MP,2);
subplot(1,2,1);
imagesc(means(xy2ch))
colorbar
title('Data Means')
% and standard deviations
stds = std(MP,0,2);
subplot(1,2,2);
imagesc(stds(xy2ch))
colorbar
title('Data Standard Deviations')


%% LEARN THE GP
% Here we build the dataset to learn the gp
y = MP(:);
x = repmat(ch2xy,10,1);
% We then build the gp structures
infm = @infGaussLik;
meanf = @meanConst;
covf = {@covMaternard, 5};
likf = @likGauss;
% Note that we get a different minimum if we start ls with 1,1 or 3,3
hyp = struct('mean', [0], 'cov', log([1 1 1]), 'lik', log(0.005));
hypoMaternard = minimize(hyp, @gp, -100, infm, meanf, covf, likf, x, y)
%show likelihood and kern params
nlmlMaternard = gp(hypoMaternard,infm,meanf,covf,likf,x,y)
fprintf('ls1: %f, ls2: %f, sf: %f\n', exp(hypoMaternard.cov));


%% PREDICT
% We also plot the predictions of the learned gp (to compare with data)
[ymu ys2 fmu fs2] = gp(hypoMaternard, infm, meanf, covf, likf, x, y, ch2xy);
figure()
subplot(1,2,1);
imagesc(fmu(xy2ch))
clims = caxis; %for use when we plot gp stds to get same color limits
colorbar
title('GP Means')
subplot(1,2,2);
imagesc(sqrt(fs2(xy2ch)))
caxis(clims)
colorbar
title('GP Standard Deviations')

% And with marco's fixed hyperparameters
% Check for likelihood var (NoiseEstim in marco's code)
% and signal std dev (believe its just not in marco's code, so set to 1
% here)
hypmarco = struct('mean', [0], 'cov', log([3 3 1]), 'lik', log(0.005));
[ymu ys2 fmu fs2] = gp(hypmarco, infm, meanf, covf, likf, x, y, ch2xy);
figure()
subplot(1,2,1);
imagesc(fmu(xy2ch))
clims = caxis; %for use when we plot gp stds to get same color limits
colorbar
title('GP Means - fixed params')
subplot(1,2,2);
imagesc(sqrt(fs2(xy2ch)))
caxis(clims)
colorbar
title('GP Standard Deviations - fixed params')