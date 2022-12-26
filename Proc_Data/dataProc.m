clear all; close all; clc;

raft_result = readmatrix("absolute.csv");
camera_pose = readmatrix("camera_pose.csv");

%  X direction
% 因為raft的單位是mm 所以會乘以0.1
fig1 = figure; subplot(2,3,1);
plot(0:1:5, camera_pose(:,1), 'LineWidth',2); hold on;
plot(0:1:5, -0.1*raft_result(:,1), 'LineWidth',2);
xlim([0 5])
ylim([-3 1])
ylabel('Translation (cm)');
title('t_x')
legend('SfM', 'RAFT')

%  Y direction
subplot(2,3,2);
plot(0:1:5, camera_pose(:,2), 'LineWidth',2); hold on;
plot(0:1:5, -0.1*raft_result(:,2), 'LineWidth',2); 
xlim([0 5])
ylim([-3 1])
title('t_y')

%  Z direction
subplot(2,3,3);
plot(0:1:5, camera_pose(:,3), 'LineWidth',2);
xlim([0 5])
ylim([-3 1])
title('t_z')

%  roll_deg
subplot(2,3,4);
plot(0:1:5, camera_pose(:,6), 'LineWidth',2);
xlim([0 5])
ylim([-0.5 0.5])
title('\theta_x')
ylabel('Rotation (degree)');

%  pitch_deg
subplot(2,3,5);
plot(0:1:5, camera_pose(:,5), 'LineWidth',2);
xlim([0 5])
ylim([-0.5 0.5])
title('\theta_y')

%  yaw_deg
subplot(2,3,6);
plot(0:1:5, camera_pose(:,4), 'LineWidth',2);
xlim([0 5])
ylim([-0.5 0.5])
title('\theta_z')

han=axes(fig1,'visible','off'); 
han.XLabel.Visible='on'; han.YLabel.Visible='on';
xlabel('frames');