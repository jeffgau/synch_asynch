clear
close all
tic % Typically ~0.5-1s per moth simulation
addpath('../data/20201111_flapper') % Point to data
addpath('../') % Point to helper functions
load 'roboflapperParamSweep_20x10_02to1.mat'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Have to hard code extraction of r3_range, synch_gain_range, and f_n
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% System Parameters
K = 0.525;          % [Nm rad^-1]
Inertia = 0.00232649;     % [kg m^2]
Gamma = 0.00107;        % [Nm s^2]
w_n = sqrt(K/Inertia);
f_n = w_n/(2*pi);

% Synchronous (sine) forcing
synch_freq = f_n * 1; % [Hz]
synch_w = synch_freq * 2 * pi; % [rad/s] - Simulink block takes rad/s input.

% Sweep Parameters
nSynch  = 10;
nR3s    = 20;
synch_gain_range = linspace(0, 1, nSynch);
r4_rat  = 0.1;
% get r3 for a range of t0 = 0.02 - 1
r3_range = calculate_r3_range_flapper(nR3s,[-1.7 0],f_n,r4_rat);
spectrogram_r3 = r3_range(11);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


plot_power_freq_amp(conv_array, freq_array, est_amp_array,synch_gain_range, r3_range, f_n)
%plot_spectrogram(spect_data);
%plot_force_mag(force_data);
%plot_kernels(sim_param);

sampling_f = 1000;
t = linspace(0, length(limit_pos)/sampling_f, length(limit_pos));
lc_keys = {'limit_pos', 'limit_vel', 'limit_t', 'spectrogram_r3', 'synch_gain_range', 'freq_array'};
lc_vals = {limit_pos, limit_vel, t, spectrogram_r3, synch_gain_range, freq_array};
lc_data = containers.Map(lc_keys, lc_vals);


plot_limit_cycles(lc_data);
save_limit_cycle_data(lc_data);

save_ofp_data(ofp_data);
save_limit_cycle_data(lc_data)
toc