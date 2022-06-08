%% exp_asynchronous_script.m
close all;
clear all;
clc

encCPR = 4096;
sampTime =0.001;
sampling_f = 1/sampTime;

% Vel Filter Cut-off frequency in rad/s
cutoffRad = 100*2*pi;

% Impulse "kick" parameters
imp_mag     = 1;     % kick torque [Nm]
imp_start   = 0.5;      % start time [s]
imp_dur     = 0.02;     % kick duration [s]

min_torque  = 4;
max_torque  = -4;

% Model name
model = 'exp_asynchronous_model';

% 
K = 0.525;          % [Nm rad^-1]
Inertia = 0.00232649;     % [kg m^2]
Gamma = 0.00107;        % [Nm s^2]

w_n = sqrt(K/Inertia);
f_n = w_n/(2*pi);


% Synchronous (sine) forcing
synch_freq = f_n * 1; % [Hz]
synch_w = synch_freq * 2 * pi; % [rad/s] - Simulink block takes rad/s input.


%% Open Model
open_system(model)

%% Update
set_param(model,'SimulationCommand','update')

%% Sweep across one value of r3, r4
ntests  = 10;
synch_gain_range = linspace(0, 1, ntests);
r3s = logspace(-.2, 1.8, ntests);
% r3s  = 0.63;
% 
t_end = 60;
t = linspace(0, t_end, t_end *1/sampTime); % Simulation frequency: 1 kHz
% temp_kernel = create_kernel_norm(t,r3,r4,k3,k4);
% temp_kernel = temp_kernel;

paramRange = 1:-0.1:0;

gain_constant     = 0.4; % Resulting torque Oscillation amplitude

runtime = 60;
set_param(model,'StopTime',int2str(runtime))
set_param(model,'SimulationMode','external')
set_param(model,'ReturnWorkspaceOutputs','off')

% Set variables for plotting spectrogram and limit cycles
% spectrogram_r3 = r3s(6);
spect_peaks = [];
n_freqs = 200; % Number of fourier modes to plot
limit_pos = [];
limit_vel = [];

spectrogram_r3 = r3s(8);

% Keep track of peak force
force_array = zeros(length(r3s),ntests);
rawData = cell(length(r3s),length(synch_gain_range));
for k = 2:length(synch_gain_range)
    for p = 1:length(r3s)
        p, k
        
        fprintf("r3 = %3.3f, Kr = %1.2f\n",r3s(p),synch_gain_range(k))
        
        gain_scaling = 5*10^-4;
        
        synch_gain = synch_gain_range(k) * gain_constant;
        
        temp_kernel = create_kernel_norm(t,r3s(p),0.9*r3s(p),1,1);
        SA_kernel = gain_scaling * gain_constant*(1-synch_gain_range(k)) * temp_kernel;
        
        t_kern      = (1:length(temp_kernel))*0.001 - 0.001;
        plot(t_kern,SA_kernel)
        
        set_param(model,'SimulationCommand','update')
        set_param(model,'SimulationCommand','start')
        
        while (get_param(model,"SimulationStatus") ~= "stopped")
            fprintf("running\n")
            pause(0.5)
            fprintf(repmat('\b', 1, 8));
        end
        
        time        = ScopeData.time;
        position    = ScopeData.signals(1).values;
        velocity    = ScopeData.signals(2).values;
        inTorque    = ScopeData.signals(3).values;
        asynchTorque= ScopeData.signals(4).values;
        synchTorque = ScopeData.signals(5).values;
        
        
        % Extract properties of dominant oscillation frequency
        pos_trim        = position(floor(end/4):end);
        vel_trim        = velocity(floor(end/4):end);
        tor_trim        = inTorque(floor(end/4):end);
        [freq, peaks]   = fourier_analysis(pos_trim-mean(pos_trim),sampling_f);
        peak_idx = find(peaks == max(peaks));
        
        % Frequency normalized to
        osc_freq = freq(peak_idx)/synch_freq;
        osc_amp = peaks(peak_idx);
        
        
        % Store values for spectrogram
        if r3s(p) == spectrogram_r3
                spect_peaks = [spect_peaks, peaks(1:n_freqs)];
                limit_pos = [limit_pos, position];
                limit_vel = [limit_vel, velocity];
        end
        
        
        % Calculate empirical power requirements and handle cases where
        % there's no oscillation
        
        if (max(pos_trim - mean(pos_trim)) < 0.01)
            conv_array(p,k) = 0;
            est_amp_array(p,k)  = 0;
            freq_array(p,k)     = 0;
        else
            conv_array(p,k) = calculate_power(vel_trim, tor_trim, pos_trim-mean(pos_trim), t);
            est_amp_array(p,k)  = osc_amp;
            freq_array(p,k)     = osc_freq;
        end
        
        % Keep track of peak force
        force_array(p,k) = max(tor_trim);
        
        raw_data{p,k}= ScopeData;

%         clf
%         plot(ScopeData.time, ScopeData.signals(1).values)
%         pause()
        
    end
end

save("roboflapperParamSweep.mat","conv_array","est_amp_array","freq_array","raw_data",...
    "force_array","spect_peaks","limit_pos","limit_vel")







