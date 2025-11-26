function run_faz()
% RUN_FAZ
% Semi-automated FAZ segmentation and feature extraction.
%
% In the original App (GUI version), the technician/doctor could adjust
% parameters live while previewing the segmentation. This script keeps the
% same semi-automated spirit by allowing the user to re-run segmentation
% with an updated threshold if the first output is not satisfactory.

clc; close all;

fprintf('\n=== FAZ Segmentation & Feature Extraction (Semi-Automated) ===\n');
fprintf('In the original GUI version, the operator could adjust parameters live.\n');
fprintf('This script will allow you to modify the threshold after previewing the FAZ.\n\n');

%% ------------------------------------------------------------
% Ask user to select image
[filename, pathname] = uigetfile({'*.jpg;*.png;*.jpeg;*.tif','Image Files'});
if isequal(filename,0)
    disp('User cancelled.');
    return;
end

imgPath = fullfile(pathname, filename);
Iorig = imread(imgPath);

% Ensure RGB
if size(Iorig,3) == 1
    Iorig = repmat(Iorig,1,1,3);
end

%% ------------------------------------------------------------
% Default parameters
resizeTo      = 512;
sigma         = 5;
medianFilt    = 4;
gaussSigma    = 2;
sharpenRad    = 0.6;
sharpenAmt    = 20;
scale         = 3;     % mm scaling (as in original App)
threshold     = 6;     % initial default

rerun = true;

while rerun

    fprintf('\nRunning FAZ segmentation with threshold = %d …\n', threshold);

    %% ------------------------------------------------------------
    % Preprocess
    Ires = imresize(Iorig, [resizeTo resizeTo]);
    Igray = rgb2gray(Ires);

    Ienh = adapthisteq(Igray);
    Ienh = medfilt2(Ienh, [medianFilt medianFilt]);
    Ienh = imgaussfilt(Ienh, gaussSigma);
    Ienh = imsharpen(Ienh,'Radius',sharpenRad,'Amount',sharpenAmt);

    %% ------------------------------------------------------------
    % Region Growing (seed at center)
    seed = [round(size(Ienh,1)/2), round(size(Ienh,2)/2)];
    regionMask = regionGrowing(Ienh, seed, threshold);

    %% ------------------------------------------------------------
    % Generate FAZ boundary
    boundary = generateContour(regionMask, sigma);

    if isempty(boundary)
        fprintf('❌ No FAZ detected at threshold = %d.\n', threshold);
    else
        fprintf('✔ FAZ boundary detected.\n');
    end

    selected_boundary = boundary;   
    x = selected_boundary(:,2);
    y = selected_boundary(:,1);

    %% ------------------------------------------------------------
    % Filled FAZ overlay
    filledOverlay = fillContour(selected_boundary, Ires);

    %% ------------------------------------------------------------
    % Max diameter
    [pt1, pt2, maxDistPx, maxDistMm] = maxDiameter(selected_boundary, scale, resizeTo);

    % Bounding rectangle
    rect_xmin = min(x); rect_xmax = max(x);
    rect_ymin = min(y); rect_ymax = max(y);
    rectPos = [rect_xmin, rect_ymin, rect_xmax-rect_xmin, rect_ymax-rect_ymin];

    %% ------------------------------------------------------------
    % Display 2×2 figure
    figure('Name','FAZ Segmentation Preview','Color','white');

    subplot(2,2,1); imshow(Ires); title('Input Image');
    subplot(2,2,2); imshow(Ienh); hold on; plot(x,y,'r','LineWidth',1.5);
        title(sprintf('Enhanced Image (threshold=%d)', threshold));
    subplot(2,2,3); imshow(filledOverlay); title('Filled FAZ');
    subplot(2,2,4);
        Iline = insertShape(Ires,'Line',[pt1(1),pt1(2),pt2(1),pt2(2)],'Color','cyan','LineWidth',2);
        Iline = insertShape(Iline,'Rectangle',rectPos,'Color','green','LineWidth',2);
        imshow(Iline); title('Max Diameter + Bounding Box');

    %% ------------------------------------------------------------
    % Ask user if they want to re-run with new threshold
    answer = input('\nDo you want to adjust the threshold and re-run? (y/n): ','s');

    if strcmpi(answer,'y')
        new_th = input('Enter new threshold value (recommended 3–20): ');
        if isnumeric(new_th) && new_th > 0
            threshold = new_th;
            rerun = true;
        else
            fprintf('Invalid input. Ending segmentation.\n');
            rerun = false;
        end
    else
        rerun = false;
    end

end

%% ------------------------------------------------------------
% Final FAZ feature extraction (after user is satisfied)

height = resizeTo;

area_mm2 = (scale^2) * polyarea(y, x) / (height^2);
perimeter_px = size(boundary,1);
perimeter_mm = (scale/height) * perimeter_px;

% Convex hull
k = convhull(x,y);
hx = x(k); hy = y(k);
hull_area = polyarea(hy, hx);
hull_area_mm2 = (scale^2) * hull_area / (height^2);

hull_per_px = 0;
for i = 1:length(k)-1
    hull_per_px = hull_per_px + pdist2([hx(i),hy(i)], [hx(i+1),hy(i+1)]);
end
hull_per_mm = (scale/height) * hull_per_px;

% Other metrics
form_factor = (4*pi*area_mm2) / (perimeter_mm^2);
roundness = (4*area_mm2) / (pi * maxDistMm^2);
rect_area_mm2 = (scale^2) * ( (rect_xmax-rect_xmin)*(rect_ymax-rect_ymin) ) / (height^2);
extent = area_mm2 / rect_area_mm2;
convexity = hull_per_mm / perimeter_mm;
solidity = area_mm2 / hull_area_mm2;
axial_ratio = (rect_xmax - rect_xmin) / (rect_ymax - rect_ymin);
irregularity = borderIrregularity(selected_boundary);

%% ------------------------------------------------------------
% Print features
fprintf('\n=== FINAL FAZ FEATURES ===\n');
fprintf('FAZ Area (mm^2):           %.3f\n', area_mm2);
fprintf('FAZ Perimeter (mm):        %.3f\n', perimeter_mm);
fprintf('Max Diameter (mm):         %.3f\n', maxDistMm);
fprintf('Form Factor:               %.3f\n', form_factor);
fprintf('Roundness:                 %.3f\n', roundness);
fprintf('Extent:                    %.3f\n', extent);
fprintf('Convexity:                 %.3f\n', convexity);
fprintf('Solidity:                  %.3f\n', solidity);
fprintf('Axial Ratio:               %.3f\n', axial_ratio);
fprintf('Border Irregularity:       %.3f\n\n', irregularity);

end

%% =====================================================================
% --------------------------- HELPER FUNCTIONS --------------------------
%% =====================================================================

function mask = regionGrowing(I, seed, thresh)
I = double(I);
[rows, cols] = size(I);
mask = false(rows, cols);

stack = seed;
seedVal = I(seed(1), seed(2));

while ~isempty(stack)
    p = stack(end,:); stack(end,:) = [];
    r = p(1); c = p(2);

    if r<1||r>rows||c<1||c>cols||mask(r,c), continue; end

    if abs(I(r,c) - seedVal) <= thresh
        mask(r,c) = true;
        for dr = -1:1
            for dc = -1:1
                stack = [stack; r+dr, c+dc];
            end
        end
    end
end
end

function boundary = generateContour(mask, sigma)
blurred = imgaussfilt(double(mask), sigma);
bw = blurred > 0.5;
B = bwboundaries(bw);
if isempty(B)
    boundary = [];
else
    [~, idx] = max(cellfun(@(x) size(x,1), B));
    boundary = B{idx};
end
end

function overlay = fillContour(boundary, image)
mask = poly2mask(boundary(:,2), boundary(:,1), size(image,1), size(image,2));
green = cat(3, zeros(size(mask)), uint8(mask)*255, zeros(size(mask)));
overlay = uint8(double(image)*0.4 + double(green)*0.6);
end

function [p1,p2,maxPx,maxMm] = maxDiameter(boundary, scale, height)
X = boundary(:,2); Y = boundary(:,1);
D = pdist2([X,Y],[X,Y]);
[maxPx, idx] = max(D(:));
[r,c] = ind2sub(size(D), idx);
p1 = [X(r), Y(r)];
p2 = [X(c), Y(c)];
maxMm = (scale/height)*maxPx;
end

function irr = borderIrregularity(boundary)
cx = mean(boundary(:,2));
cy = mean(boundary(:,1));
dist = sqrt((boundary(:,2)-cx).^2 + (boundary(:,1)-cy).^2);
irr = std(dist)/mean(dist);
end
