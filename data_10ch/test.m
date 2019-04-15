exps = [3 4 6 7 8 9 11 12 13 14];
for exp = 3
    x2 = PARA{exp}(2,:);
    x3 = PARA{exp}(3,:);
    x = [x2 x3];
    %x = PARA{exp}(1,:);
    
    uv = unique(x);
    n = histc(x,uv)
end

