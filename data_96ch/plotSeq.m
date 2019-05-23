function plotSeq(queried, datameans , hypos, kappas)
%PLOTSEQ 此处显示有关此函数的摘要
%   此处显示详细说明

xy2ch = [[1 96:-1:89 1]' (88:-1:79)' (78:-1:69)' (68:-1:59)' (58:-1:49)'...
    (48:-1:39)' (38:-1:29)' (28:-1:19)' (18:-1:9)' [1 8:-1:1 1]' ];
xy2ch2 = [[100 96:-1:89 100]' (88:-1:79)' (78:-1:69)' (68:-1:59)' (58:-1:49)'...
    (48:-1:39)' (38:-1:29)' (28:-1:19)' (18:-1:9)' [100 8:-1:1 100]' ];
ch2xy = zeros(96,2);
for i = 1:96
    [y,x] = find(xy2ch2==i);
    ch2xy(i,:) = [x,y];
end
queries = queried(:,1);

%f = figure;
%ax = axes('Parent',f,'position',[0.13 0.39  0.77 0.54]);
ax1 = subplot(2,2,1);
h = imagesc(ax1,datameans(xy2ch));
colorbar
title('real mean')

c = uicontrol('Style','slider','Position',[81,10,419,10],...
              'value',1, 'min',1, 'max',96, 'SliderStep',[1/95 1/10]);
c.Callback = @selection;

    function selection(src,event)
        % We first plot the point queried
        idx = c.Value;
        [y_,x_] = find(xy2ch == queries(idx));
        [y_next, x_next] = find(xy2ch == queries(idx+1));
        % Then plot the gp predictions
        x = ch2xy(queried(1:idx,1),:);
        y = queried(1:idx,2);
        [ymu ys2 fmu fs2] = gp(hypos{idx}, @infGaussLik, [], {@covMaternard,5}, @likGauss, x, y, ch2xy);
        disp(hypos{idx})
        disp(exp(hypos{idx}.cov))
        kappa = kappas(idx);
        acqmap = ymu + kappa* sqrt(ys2);
        [acqmax, acqargmax] = max(acqmap);
        
        subplot(2,2,1);
        h = imagesc(ax1,datameans(xy2ch));
        colorbar
        title('real mean')
        hold on;
        plot(x_, y_, 'r+');
        plot(x_next, y_next, 'kx');
        
        subplot(2,2,2);
        imagesc(acqmap(xy2ch));
        colorbar
        title('acquisition map');
        hold on;
        plot(x_, y_, 'r+');
        plot(x_next, y_next, 'kx');
        
        subplot(2,2,3);
        imagesc(ymu(xy2ch));
        colorbar
        title('gp mean');
        hold on;
        plot(x_, y_, 'r+');
        plot(x_next, y_next, 'kx');
        
        subplot(2,2,4);
        imagesc(sqrt(ys2(xy2ch)));
        colorbar
        title('gp std');
        hold on;
        plot(x_, y_, 'r+');
        plot(x_next, y_next, 'kx');
    end

end
