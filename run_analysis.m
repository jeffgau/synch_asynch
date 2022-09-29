clear
system = 'moth'; % moth, flapper, or robobee
tic % Typically ~0.5-1s per moth simulation

%% Load System and Simulation parameters as Map containers
sys_param = load_system_param(system);

sim_param = load_simulation_param(system, sys_param('f_n'));

%% Calibrate force gains so synch and (peak) asynch match
sys_param('asynch_gain')
[sys_param,sim_param] = tune_force_gains(sys_param,sim_param);
sys_param('asynch_gain')


%% Run the full sweep
% Outputs a struct with all the relevant data
datastruct = run_simulation(sys_param, sim_param);


%% Run sims at one value of r3
% r3 = 46,  t0/Tn = 0.5 
% r3 = 120, t0/Tn = 0.19
% r3 = 230, t0/Tn = 0.1
% r3 = 460, t0/Tn = 0.05
sim_param('r3_range') = 46;
datastruct = single_r3_sweep(sys_param,sim_param)

% TO DO
% - make plot_spectrogram work if I want to have spectrograms for several
% r3 values
% - make plot limit cycles do the same, but also plot all on the same
% figure somehow


%% 
close all
plot_power_freq_amp(datastruct, sim_param, sys_param)
plot_spectrogram(datastruct.spect_data);
plot_force_mag(datastruct.force_data);
plot_kernels(sim_param);
for i = 1:10
    plot_limit_cycles(datastruct.lc_data, i);
end
%plot_r3_freq(freq_array, sim_param('r3_range'), sim_param('r4_ratio'))

save_ofp_data(datastruct.ofp_data, sim_param('r3_range'));
save_limit_cycle_data(datastruct.lc_data)
toc



