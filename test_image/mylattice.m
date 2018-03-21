%% mylattice: function description
function [A, b] = mylattice(im, gx, gy, scale, alpha, beta_)
	[h,w,~] = size(gx)
	num0 = h * w;
	ori = reshape(1:num0, [h,w]);
	
	left = ori - h;
    right =  ori + h;
	down = ori + 1;
	up = ori - 1;
    
	ori = ori(2:end-1, 2:end-1);
	left = left(2:end-1, 2:end-1);
	right = right(2:end-1, 2:end-1);
	down = down(2:end-1, 2:end-1);
	up = up(2:end-1, 2:end-1);

	ori = ori(:);
	right = right(:);
	down = down(:);
    left = left(:);
    up = up(:);
	num = numel(ori);
	
	rows = (1:length(im(:)))';
	cols = (1:length(im(:)))';
	valsA = ones([length(im(:)),1]);
	A0 = sparse(rows, cols, valsA);
	alpha1 = repmat(alpha, 1, 3, 1);
	xxx = im .* double(alpha1);
    b0 = sparse(xxx(:));

	row = [ori; ori];
	col = [ori; right];
	val = [ones([num,1]); -ones([num, 1])];
	Ax = sparse(row, col, val, num0, num0);

	row = [ori; ori];
	col = [ori; down];
	val = [ones([num,1]); -ones([num, 1])];
	Ay = sparse(row, col, val, num0, num0);

	row = [ori; ori];
	col = [ori; left];
	val = [ones([num,1]); -ones([num, 1])];
	Axn = sparse(row, col, val, num0, num0);

	row = [ori; ori];
	col = [ori; up];
	val = [ones([num,1]); -ones([num, 1])];
	Ayn = sparse(row, col, val, num0, num0);


	A = [A0; Ax*scale; Ay*scale; Axn*scale; Ayn*scale];


	gx0 = gx;
	gy0 = gy;

	gx(1,:) = 0;
	gx(end,:) = 0;
	gx(:,1) = 0;
	gx(:,end) = 0;
	gy(1,:) = 0;
	gy(end,:) = 0;
	gy(:,1) = 0;
	gy(:,end) = 0;
	
	gx2 = zeros(size(gx0));
	gx2(2:end,:) = gx0(1:end-1,:); 

	gy2 = zeros(size(gy0));
	gy2(:,2:end) = gx0(:,1:end-1); 

	bx =gx(:);
	by =gy(:);
	
	bxn =gx2(:);
	byn =gy2(:);

	b = [b0;-bx*scale;-by*scale; -bxn*scale;-byn*scale;];
end