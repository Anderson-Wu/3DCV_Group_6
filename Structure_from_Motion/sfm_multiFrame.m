clear all; close all; clc;
%% Read images and intrinsics
imagePath = "D:\ChengEn\NTU_Research\7_ComputerVision\1_ExpData\SCENE_MOVE\2022_1214\frames";
load("D:\ChengEn\NTU_Research\7_ComputerVision\1_ExpData\SCENE_MOVE\2022_1214\calib\cameraParams.mat")
intrinsics = cameraParams.Intrinsics;
imds = imageDatastore(imagePath);

% Display the images
% figure;
% montage(imds.Files, 'Size',[3,2]);

I = imread(imds.Files{1});
I = undistortImage(I, intrinsics);

% Detect feature points
prevPoints   = detectSURFFeatures(rgb2gray(I), NumOctaves=8);
prevFeatures = extractFeatures(rgb2gray(I), prevPoints, Upright=true);
vSet = imageviewset;
viewId = 1;
vSet = addView(vSet, viewId, rigidtform3d, Points=prevPoints);

for i = 2:length(imds.Files)
    str = extractAfter(imds.Files{i},imagePath+'\');
    fprintf("processing " + str + '\n');
    % Undistort the current image.
    J = imread(imds.Files{i}); 
    J = undistortImage(J, intrinsics);
    
    % Detect, extract and match features.
    currPoints   = detectSURFFeatures(rgb2gray(J), NumOctaves=8);
    currFeatures = extractFeatures(rgb2gray(J), currPoints, Upright=true);    
    indexPairs   = matchFeatures(prevFeatures, currFeatures, ...
        MaxRatio=0.8, Unique=true);
    
    % Select matched points.
    matchedPoints1 = prevPoints(indexPairs(:, 1));
    matchedPoints2 = currPoints(indexPairs(:, 2));
    
    %% Estimate the Essential Matrix
    [relPose, inlierIdx] = helperRelativePose(I, J, ...
        matchedPoints1, matchedPoints2, intrinsics);
    
    % Get the table containing the previous camera pose.
    prevPose = poses(vSet, i-1).AbsolutePose;
        
    % Compute the current camera pose in the global coordinate system 
    % relative to the first view.
    currPose = rigidtform3d(prevPose.A*relPose.A);
    
    % Add the current view to the view set.
    vSet = addView(vSet, i, currPose, Points=currPoints);

    % Store the point matches between the previous and the current views.
    vSet = addConnection(vSet, i-1, i, relPose, Matches=indexPairs(inlierIdx,:));
    
    % Find point tracks across all views.
    tracks = findTracks(vSet);

    % Get the table containing camera poses for all views.
    camPoses = poses(vSet);

    % Triangulate initial locations for the 3-D world points.
    xyzPoints = triangulateMultiview(tracks, camPoses, intrinsics);
    
    % Refine the 3-D world points and camera poses.
    [xyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(xyzPoints, ...
        tracks, camPoses, intrinsics, FixedViewId=1, ...
        PointsUndistorted=true);

    % Store the refined camera poses.
    vSet = updateView(vSet, camPoses);

    prevFeatures = currFeatures;
    prevPoints   = currPoints;  
    I = J;
end
disp("Finished camera pose estimation")

%% Display the camera
% Display camera poses.
camPoses = poses(vSet);
figure;
plotCamera(camPoses, Size=0.2);
hold on

% Exclude noisy 3-D points.
goodIdx = (reprojectionErrors < 5);
xyzPoints = xyzPoints(goodIdx, :);

% Display the 3-D points.
pcshow(xyzPoints, VerticalAxis='y', VerticalAxisDir='down', MarkerSize= 45);
grid on
hold off

% Specify the viewing volume.
loc1 = camPoses.AbsolutePose(1).Translation;
xlim([loc1(1)-5, loc1(1)+4]);
ylim([loc1(2)-5, loc1(2)+4]);
zlim([loc1(3)-1, loc1(3)+20]);
camorbit(0, -30);

title('Refined Camera Poses');

%% Compute Dense Reconstruction (Only use two frames)
% Read and undistort the first and secound image
J = imread(imds.Files{1});
J = undistortImage(J, intrinsics); 
frame2 = imread(imds.Files{2});
frame2 = undistortImage(frame2, intrinsics); 
imagePoints1 = detectMinEigenFeatures(im2gray(J), MinQuality = 0.001);

% Create the point tracker
tracker = vision.PointTracker(MaxBidirectionalError=1, NumPyramidLevels=6);

% Initialize the point tracker
imagePoints1 = imagePoints1.Location;
initialize(tracker, imagePoints1, J);

% Track the points
[imagePoints2, validIdx] = step(tracker, frame2);
matchedPoints1 = imagePoints1(validIdx, :);
matchedPoints2 = imagePoints2(validIdx, :);

% Compute the camera matrices for each position of the camera
% The first camera is at the origin looking along the Z-axis. Thus, its
% transformation is identity.
camMatrix1 = cameraProjection(intrinsics, camPoses.AbsolutePose(1,1));
camMatrix2 = cameraProjection(intrinsics, camPoses.AbsolutePose(2,1));

% Compute the 3-D points
points3D = triangulate(matchedPoints1, matchedPoints2, camMatrix1, camMatrix2);

% Get the color of each reconstructed point
numPixels = size(J, 1) * size(J, 2);
allColors = reshape(J, [numPixels, 3]);
colorIdx = sub2ind([size(J, 1), size(J, 2)], round(matchedPoints1(:,2)), ...
    round(matchedPoints1(:, 1)));
color = allColors(colorIdx, :);

% Create the point cloud
ptCloud = pointCloud(points3D, 'Color', color);

% Visualize the point cloud
pcshow(ptCloud, VerticalAxis='y', VerticalAxisDir='down', MarkerSize=60);

% Label the axes
xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis')

title('Up to Scale Reconstruction of the Scene');

%% Scale Recovery
% i need the 3D points relationship
% i should fit my 3d point cloud
% use pcfitcylinder pcfitsphere function
figure; imshow(frame2);
r = drawrectangle;
inroi = inROI(r,double(matchedPoints2(:,1))',double(matchedPoints2(:,2))');
inroi_idx = find(inroi==1);
tempPtCloud = select(ptCloud, inroi_idx);
figure;  pcshow(tempPtCloud,VerticalAxis='y', VerticalAxisDir='down', MarkerSize=60);
% denoisetempPtCloud= pcdenoise(tempPtCloud);
% figure;  pcshow(denoisetempPtCloud);
% cylinder = pcfitcylinder(tempPtCloud,0.008,[0,1,0]);
cylinder = pcfitcylinder(tempPtCloud,0.008);
hold on;  plot(cylinder);

% Determine the scale factor
%   鐵樂士高度為20.1cm
%   A4高度21.0cm
scaleFactor = 21.0/cylinder.Height;

% Scale the point clound
ptCloud = pointCloud(points3D*scaleFactor, Color = color);
for i=1:length(imds.Files)
    camPoses.AbsolutePose(i,1).Translation = camPoses.AbsolutePose(i,1).Translation*scaleFactor;
end

%% Convert rotation matrix to euler and save csv
savePoses = zeros(length(imds.Files),6);

% "ZYX" (default) – The order of rotation angles is z-axis, y-axis, x-axis.
for i = 2:length(imds.Files)
    savePoses(i,1:3) = camPoses.AbsolutePose(i,1).Translation;
    savePoses(i,4:6) = rotm2eul(camPoses.AbsolutePose(i,1).R);
end

savePosesTable = array2table(savePoses,'VariableNames',{'x(cm)','y(cm)','z(cm)' ...
    ,'yaw(deg)','pitch(deg)','roll(deg)'});

writetable(savePosesTable,'expx_camera_pose.csv')
