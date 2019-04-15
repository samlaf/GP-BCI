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
% We normalize MP (can't normalize if we are going to do
% sequential-optimization since we haven't seen all the data)
%MP = (MP - mean(MP(:))) / std(MP(:));


% We plot mean responses
means = mean(MP,2);
figure(1)
imagesc(means(xy2ch))
colorbar
title('Data Means')
% and standard deviations
stds = std(MP,0,2);
figure(2)
imagesc(stds(xy2ch))
colorbar
title('Data Standard Deviations')
%plot(mean(MP,2))
%hold on;
%plot(MP,'.')

%%
% Here we build the dataset to learn the gp
y = MP(:);
x = repmat(ch2xy,10,1);
% We then build the gp structures
infm = @infGaussLik;
meanf = [];
covf = @covSEard;
likf = @likGauss;
hyp = struct('mean', [], 'cov', log([1 1 1]), 'lik', log(0.1));
hypoSEard = minimize(hyp, @gp, -100, infm, meanf, covf, likf, x, y)
nlmlSEard = gp(hypoSEard,infm,meanf,covf,likf,x,y)

%%
%We try different kernels
%covf = {@covMaterniso, 3};
%hypoM = minimize(hyp, @gp, -100, infm, meanf, covf, likf, x, y)
%nlmlM = gp(hypoM,infm,meanf,covf,likf,x,y)
%%
%covf = @covSEiso;
%hyp.cov = log([1 1]);
%hypoSEiso = minimize(hyp, @gp, -100, infm, meanf, covf, likf, x, y)
%nlmlSEiso = gp(hypoSEiso,infm,meanf,covf,likf,x,y)

%%
% We also use the learned gp to predict
[ymu ys2 fmu fs2] = gp(hypoSEard, infm, meanf, covf, likf, x, y, ch2xy);
figure()
imagesc(fmu(xy2ch))
colorbar
figure()
imagesc(sqrt(fs2(xy2ch)))
colorbar

%figure()
%x_ = 1:10;
%y_ = 1:10;
%[X,Y] = meshgrid(x_,y_);
%surf(X,Y,ymu(xy2ch))