%% fine_result: function description
%% fine_result('albedo_0001.png', 'albedo_dx_0001.png', 'albedo_dy_0001.png', 2, '/media/albertxavier/data/eccv/graduation-project/pytorch/results/images/image_split/alley_1/alpha_0001.mat', '/media/albertxavier/data/eccv/graduation-project/pytorch/results/images/image_split/alley_1/beta_0001.mat');
function fine_result(im_path, gx_path, gy_path, scale, alpha_path, beta_path)

% begin time
t = cputime;
	%% genComponents: function description
	% function [A0, b0, A1, b1] = genComponents(im, gx, gy)


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
	alpha = load(alpha_path);
	beta_ = load(beta_path);
    alpha = alpha.alpha;
    beta_ = beta_.beta;

	im = double(imread(im_path)) / 255.;
	gx = double(imread(gx_path)) / 255.;
	gy = double(imread(gy_path)) / 255.;
	gx = gx - 0.5;
	gy = gy - 0.5;

	[w0,h0,c0] = size(im);

	[im, gx, gy] = genOnePlane(im, gx, gy);

	[A,b] = mylattice(im,gx,gy, scale, alpha, beta_);
	ub = im(:) + 1.;
	lb = im(:) - 1.;
	res = solve_intrinsic(A, b, ub, lb);
	res = reshape(res,[w0,h0,c0]);
	imwrite(res, 'res.png');
    e = cputime-t
    
end
