function [relPose, inlierIdx] = ...
    helperRelativePose(I, J, matchedPoints1, matchedPoints2, intrinsics)

if ~isnumeric(matchedPoints1)
    matchedPoints1 = matchedPoints1.Location;
end

if ~isnumeric(matchedPoints2)
    matchedPoints2 = matchedPoints2.Location;
end

for i = 1:300
    % Estimate the essential matrix.
    [E, inlierIdx] = estimateEssentialMatrix(matchedPoints1, matchedPoints2,...
        intrinsics, 'Confidence', 99, 'MaxDistance', 0.5);


    % Make sure we get enough inliers
    if sum(inlierIdx) / numel(inlierIdx) < .3
        continue;
    end

    % Get the epipolar inliers.
    inlierPoints1 = matchedPoints1(inlierIdx, :);
    inlierPoints2 = matchedPoints2(inlierIdx, :);

    % Compute the camera pose from the fundamental matrix.
    [relPose, validPointFraction] = ...
        estrelpose(E, intrinsics, inlierPoints1, inlierPoints2);

    if validPointFraction > .8
        return;
    end

end
error('Unable to compute the Essential matrix');
% showMatchedFeatures(I,J, inlierPoints1, inlierPoints2);




