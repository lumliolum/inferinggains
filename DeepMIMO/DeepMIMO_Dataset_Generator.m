% --------- DeepMIMO: A Generic Dataset for mmWave and massive MIMO ------%
% Author: Ahmed Alkhateeb
% Date: Sept. 5, 2018 
% Goal: Encouraging research on ML/DL for mmWave/massive MIMO applications and
% providing a benchmarking tool for the developed algorithms
% ---------------------------------------------------------------------- %

function [DeepMIMO_dataset,params]=DeepMIMO_Dataset_Generator()
% ------  Inputs to the DeepMIMO dataset generation code ------------ % 

%------Ray-tracing scenario
params.scenario='I1_2p5';                % The adopted ray tracing scenarios [check the available scenarios at www.aalkhateeb.net/DeepMIMO.html]

%------DeepMIMO parameters set
%Active base stations 
params.active_BS=[1];          % Includes the numbers of the active BSs (values from 1-18 for 'O1')

% Active users
params.active_user_first=1;       % The first row of the considered receivers section (check the scenario description for the receiver row map)
params.active_user_last=502;        % The last row of the considered receivers section (check the scenario description for the receiver row map)

% Number of BS Antenna 
params.num_ant_x=1;                  % Number of the UPA antenna array on the x-axis 
params.num_ant_y=1;                 % Number of the UPA antenna array on the y-axis 
params.num_ant_z=1;                  % Number of the UPA antenna array on the z-axis
                                     % Note: The axes of the antennas match the axes of the ray-tracing scenario
                              
% Antenna spacing
params.ant_spacing=.5;               % ratio of the wavelength; for half wavelength enter .5        

% System bandwidth
params.bandwidth=0.02;                % The bandiwdth in GHz 

% OFDM parameters
params.num_OFDM=64;                % Number of OFDM subcarriers
params.OFDM_sampling_factor=1;   % The constructed channels will be calculated only at the sampled subcarriers (to reduce the size of the dataset)
params.OFDM_limit=64;                % Only the first params.OFDM_limit subcarriers will be considered when constructing the channels

% Number of paths
params.num_paths=5;                  % Maximum number of paths to be considered (a value between 1 and 25), e.g., choose 1 if you are only interested in the strongest path

params.saveDataset=0;               % 0 means don't save and 1 means save

% -------------------------- DeepMIMO Dataset Generation -----------------%
[DeepMIMO_dataset,params]=DeepMIMO_generator(params);

% ------------------ extra code added ----------------------------------- %
fprintf(' Converting in suitable format for python \n')
channelgains = zeros(params.num_user,length(params.active_BS),params.num_ant_x*params.num_ant_y*params.num_ant_z,floor(params.OFDM_limit/params.OFDM_sampling_factor));
count = 0;
percent_done = 100*count/params.num_user;
reverseStr = 0;
msg = sprintf('- Percent done: %3.1f', percent_done); %Don't forget this semicolon
fprintf([reverseStr, msg]);
reverseStr = repmat(sprintf('\b'), 1, length(msg));
for i=1:params.num_user
    for j=1:length(params.active_BS)
        channelgains(i,j,:,:) = DeepMIMO_dataset{1,j}.user{1,i}.channel;
    end
    count = count + 1;
    percent_done = 100*count/params.num_user;
    msg = sprintf('- Percent done: %3.1f', percent_done); %Don't forget this semicolon
    fprintf([reverseStr, msg]);
    reverseStr = repmat(sprintf('\b'), 1, length(msg));
end
save('DeepMIMODataset/deepmimo_dataset_I1_2p5_64_ofdm_5_paths.mat','channelgains');
