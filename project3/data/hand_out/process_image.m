function features = process_image(base_path, img_id )

% Read the raw image and mask of interesting area.
pure_png = imread(strcat(base_path, sprintf('%04d', img_id), '_raw.tif'));
mask_png = imread(strcat(base_path, sprintf('%04d', img_id), '_msk.png'));

% Compute histogram features.
sig_1D = sig_1D_signature(mask_png);
phog_feat = (anna_phog(rgb2gray(pure_png), 8, 360, 3, [1;78;1;78]))';

% Normalize the histograms.
phog_feat = phog_feat / sum(phog_feat);
sig_1D = sig_1D / sum(sig_1D);

% add curvature measure
% area of the image over the area of the convex hull
% should be larger for symmetrical images, i.e. malignant cells

compl = imcomplement(mask_png);
% erode the image
se = strel('disk',2);
erodedI = imerode(compl,se);
area_rel = bwarea(erodedI)/ bwarea(bwconvhull(erodedI, 'object', 4));

% measure the circularity by relation Perimeter/Area
 CC=bwconncomp(compl);
 stats = regionprops(CC, 'all');
 perimeter = [stats.Perimeter];
 area = [stats.Area];
 circularities = (perimeter .^2 ) ./ (4*pi*area);
 circularity = sum(circularities)/length(circularities);
 features = [circularity, area_rel, phog_feat, sig_1D];
