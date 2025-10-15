clc; clear; close all;
dataDir = '/Users/urja/Downloads/Leaves';
imds = imageDatastore(dataDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
numImages = min(1000, numel(imds.Files));
fprintf("Processing %d images from %s\n", numImages, dataDir);
outputDir = fullfile(pwd, 'LeafEdgeEnhanced');
if ~exist(outputDir, 'dir'); mkdir(outputDir); end
randomIdx = randperm(numImages, min(9,numImages));

%Display original images
figure('Name','Step 0: Original Images');
for k = 1:min(9,numImages)
    subplot(3,3,k); imshow(readimage(imds,randomIdx(k)));
    title(['Original ' num2str(randomIdx(k))]);
end

%Step 1: Gaussian Blur
blurredImages = cell(numImages,1);
for i = 1:numImages
    img = readimage(imds,i);
    blurredImages{i} = imgaussfilt(img, 1); % sigma=1
end

%Display blurred images
figure('Name','Step 1: Gaussian Blurred Images');
for k = 1:min(9,numImages)
    subplot(3,3,k); imshow(blurredImages{randomIdx(k)});
    title(['Blurred ' num2str(randomIdx(k))]);
end

% Step 2: Edge Enhancement (Dilation)
edgeEnhanced = cell(numImages,1);
se = strel('disk', 4);
for i = 1:numImages
    Igray = rgb2gray(blurredImages{i});
    edgeImg = edge(Igray,'sobel');
    enhancedEdge = imdilate(edgeImg, se);
    edgeEnhanced{i} = enhancedEdge;
end

%Display edge-enhanced images
figure('Name','Step 2: Enhanced Edge (Dilated) Images');
for k = 1:min(9,numImages)
    subplot(3,3,k); imshow(edgeEnhanced{randomIdx(k)});
    title(['EdgeEnhanced ' num2str(randomIdx(k))]);
end

%Step 3: Feature Extraction from edge image and display corresponding mask
features = zeros(numImages,7); % [area perimeter aspect_ratio circularity compactness cx cy]
segMasks = cell(numImages,1);
for i = 1:numImages
    BW = edgeEnhanced{i};
    BW = imfill(BW, 'holes');
    BW = bwareaopen(BW, 30); % Remove small blobs
    segMasks{i} = BW;
    cc = bwconncomp(BW);
    stats = regionprops(cc, 'Area','Perimeter','BoundingBox','Centroid');
    if isempty(stats)
        features(i,:) = 0;
        continue;
    end
    [~,maxidx] = max([stats.Area]);
    A = stats(maxidx).Area;
    P = stats(maxidx).Perimeter;
    bbox = stats(maxidx).BoundingBox;
    centroid = stats(maxidx).Centroid;
    asp = bbox(3)/bbox(4);
    circ = 4*pi*A/(P^2);
    comp = A/(bbox(3)*bbox(4));
    features(i,:) = [A, P, asp, circ, comp, centroid];
end

% Display masks used for measurement
figure('Name','Step 3: Segmentation/Masks Used');
for k = 1:min(9,numImages)
    subplot(3,3,k); imshow(segMasks{randomIdx(k)});
    title(['Mask ' num2str(randomIdx(k))]);
end

%Step 4: Save to CSV
featureTable = array2table(features, 'VariableNames', ...
    {'Area','Perimeter','AspectRatio','Circularity','Compactness','CentroidX','CentroidY'});
writetable(featureTable, fullfile(outputDir, 'LeafEdgeEnhanced_Features.csv'));
disp('All leaf features saved to CSV');
