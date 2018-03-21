%% fine_result: function description
function fine_result(im_path, gx_path, gy_path, scale)

% begin time
t = cputime;
	%% genComponents: function description
	% function [A0, b0, A1, b1] = genComponents(im, gx, gy)
	% 	pairwise = cat(3, gy, gx);

	% 	rows = (1:length(im(:)))';
	% 	cols = (1:length(im(:)))';
	% 	valsA = ones([length(im(:)),1]);
	% 	A0 = sparse(rows, cols, valsA);
	% 	b0 = sparse(im(:));

	% 	rows = [];
	% 	cols = [];
	% 	values = [];

	% 	[~, edges] = lattice(size(im,1),size(im,2),0);
	% 	n = length(incidence(edges));
	% 	A1 = [incidence(edges)];
	% 	gy = pairwise(1:end-1,:,1);
	% 	gx = pairwise(:,1:end-1,2);
	% 	b1 = -[gy(:);gx(:)];
	% end

	%% solve: function description
	function [u] = solve_intrinsic(A, b, ub, lb)
		options = optimoptions('lsqlin','Display','iter');
		U = lsqlin(A,b,[],[],[],[],lb,ub,[], options);
		[w,h,c] = size(im);
		u = reshape(U,w,h,c);
	end

	%% genOnePlane: function description
	function [im, gx, gy] = genOnePlane(im, gx, gy)
		[w,h,c] = size(im);
		im = reshape(im, [w,h*c,1]);
		gx = reshape(gx, [w,h*c,1]);
		gy = reshape(gy, [w,h*c,1]);
	end

	% begin
	im = double(imread(im_path)) / 255.;
	gx = double(imread(gx_path)) / 255.;
	gy = double(imread(gy_path)) / 255.;
	gx = gx - 0.5;
	gy = gy - 0.5;

	[w0,h0,c0] = size(im);

	[im, gx, gy] = genOnePlane(im, gx, gy);
	% A = [];
	% b = [];
	% [A0, b0, A1, b1] = genComponents(im, gx, gy);
	% A = [A0; A1];
	% b = [b0; b1];

	[A,b] = mylattice(im,gx,gy, scale);
	ub = im(:) + 1.;
	lb = im(:) - 1.;
	res = solve_intrinsic(A, b, ub, lb);
	res = reshape(res,[w0,h0,c0]);
	imwrite(res, 'res.png');
    e = cputime-t
    
end
