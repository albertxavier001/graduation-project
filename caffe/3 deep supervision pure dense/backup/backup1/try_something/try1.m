im = imread('../gt.png');
gd = im(2:end,1:end-1,:) - im(1:end-1,1:end-1,:);
max(gd(:))
min(gd(:))
imshow(mat2gray(gd));
