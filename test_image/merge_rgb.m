r = double(imread('c_1.png'));
g = double(imread('c_2.png'));
b = double(imread('c_3.png'));

[w,h,c] = size(r);
rgb = zeros(w,h,c*3);
rgb(:,:,1) = r;
rgb(:,:,2) = g;
rgb(:,:,3) = b;
imwrite(rgb/255., 'final.png');