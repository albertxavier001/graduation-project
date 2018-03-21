channel = 3
scale = 5.;


addpath('/home/albertxavier/workspace/JCNF/Inference/graphAnalysisToolbox-1.0');
shift = 3
alpha = 1;
albedo = double(imread('out_merge.png')) / 255.;
albedo = albedo(:,:,channel);



im_gx = double(imread('out_merge_dx.png')) / 255.;
im_gy = double(imread('out_merge_dy.png')) / 255.;
im_gx = im_gx(:,:,channel) - 0.5;
im_gy = im_gy(:,:,channel) - 0.5;
pairwise = cat(3, im_gy, im_gx);


% crop
size_ = 416;
hb = 1;
wb = min(1+floor(size_/2)*shift, 1024-size_+1);
wb-1+size_


rows = (1:length(albedo(:)))';
cols = (1:length(albedo(:)))';
valsA = ones([length(albedo(:)),1]);
A0 = sparse(rows, cols, valsA);
b0 = sparse(albedo(:));

rows = [];
cols = [];
values = [];


[~, edges] = lattice(size(albedo,1),size(albedo,2),0);
n = length(incidence(edges));
A1 = [incidence(edges)];
gy = pairwise(1:end-1,:,1);
gx = pairwise(:,1:end-1,2);
b1 = -[gy(:);gx(:)];


%%
A = [scale*A1;A0*alpha];
b = [scale*b1;b0*alpha];

upper_bound = albedo(:) + 100;
lower_bound = albedo(:) - 50;

disp('begin to solve')
options = optimoptions('lsqlin','Display','iter'); %'Algorithm','interior-point',
U = lsqlin(A,b,[],[],[],[],lower_bound,upper_bound,[], options); 
u = reshape(U,size(albedo,1),size(albedo,2),size(albedo,3));

uu = u;
uu(uu<0)= 0.;
uu(uu>255)= 255.;
uu = uu / 255.;
imwrite(mat2gray(u), sprintf('./c_%d.png', channel));
