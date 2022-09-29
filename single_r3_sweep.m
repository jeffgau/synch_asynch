function datastruct = single_r3_sweep(sys_param, sim_param)
% Note: expects that sim_param('r3_range') is a scalar value


% Unpack system parameters
Inertia = sys_param('I');
K = sys_param('K');
Gamma = sys_param('Gamma');
C_viscous  = sys_param('C_viscous');
T = sys_param('T');
%K =K/T^2;
f_n = sys_param('f_n');
synch_freq = sys_param('synch_freq');
v_initial = sys_param('v_initial');
x_initial = sys_param('x_initial');
K_muscle = sys_param('K_muscle');
asynch_gain = sys_param('asynch_gain');
gain_constant = sys_param('gain_constant');

% Unpack simulation parameters
ntests = sim_param('ntests');
r3_range = sim_param('r3_range');
sampling_f = sim_param('sampling_f');
synch_gain_range = sim_param('synch_gain_range');
t = sim_param('t');
t_end = sim_param('t_end');
r4_ratio = sim_param('r4_ratio');

datastruct = struct;

datastruct.freq_array      = zeros(length(r3_range),ntests);
datastruct.conv_array      = zeros(length(r3_range),ntests);
datastruct.est_amp_array   = zeros(length(r3_range),ntests);
datastruct.force_array     = zeros(length(r3_range),ntests);
datastruct.psd_array       = zeros(length(r3_range),ntests);
datastruct.lc_array        = struct("position",cell(length(r3_range),ntests),"velocity",cell(length(r3_range),ntests));


sim_count = 1;


% Set variables for plotting spectrogram and limit cycles
% spectrogram_r3 = sim_param('r3_spect');
spect_peaks = cell(length(r3_range),1);
n_freqs = 2000; % Number of fourier modes to plot
limit_pos = cell(length(r3_range),1);
limit_vel = cell(length(r3_range),1);
limit_t = cell(length(r3_range),1);

fft_freqrange = [];


for p = 1:length(r3_range)
    % p sweeps over r3
    for k = 1:ntests
        % k sweeps over K_r
        
%         p, k
        clc
        fprintf("p: %3d, k: %3d\n", p,k )

        
        % Sweep through parameters to generate forcing functions
        synch_gain = synch_gain_range(k) * gain_constant;
        temp_kernel = create_kernel(t, r3_range(p), r3_range(p)* r4_ratio, 1, 1);
        SA_kernel = asynch_gain * gain_constant * (1 - synch_gain_range(k)) * temp_kernel;
        
        % Run simulation and extract variables
        simIn = Simulink.SimulationInput('asynchronous_model');
        simIn = simIn.setVariable('Gamma', Gamma);
        simIn = simIn.setVariable('Inertia', Inertia);
        simIn = simIn.setVariable('C_viscous', C_viscous);
        simIn = simIn.setVariable('K', K);
        simIn = simIn.setVariable('K_muscle', K_muscle);
        simIn = simIn.setVariable('SA_kernel', SA_kernel);
        simIn = simIn.setVariable('synch_gain', synch_gain);
        simIn = simIn.setVariable('synch_w', synch_freq*2*pi);
        simIn = simIn.setVariable('x_initial', x_initial);
        simIn = simIn.setVariable('v_initial', v_initial);
        simIn = simIn.setModelParameter('StartTime', '0', 'StopTime', num2str(t_end));
        simIn = simIn.setModelParameter('FixedStep', num2str(1/sampling_f));

        %simout = sim('asynchronous_model', 'StartTime', '0', 'StopTime', num2str(t_end));
        simout = sim(simIn);
        dat = simout.simout.Data;
        
        position        = dat(:,4);
        velocity        = dat(:,3);
        upstroke        = dat(:,2);
        downstroke      = dat(:,1);
        net_force       = upstroke + downstroke;
        
%         plot(t, position(1:end-1))
%         hold on
        upstroke_passive    = dat(:,5);
        upstroke_asynch     = dat(:,6);
        downstroke_passive 	= dat(:,7);
        downstroke_asynch   = dat(:,8);
        
        upstroke_synch      = dat(:,9);
        downstroke_synch    = dat(:,10);
        
        %spring_force        = dat(:,11);
        %aero_force          = dat(:,12);
        %visc_force          = dat(:,13);
        
        % Extract properties of dominant oscillation frequency
        [freq, peaks] = fourier_analysis(position(length(t)/2:end) - mean(position(length(t)/2:end)), sampling_f);
        peak_idx = find(peaks == max(peaks));
        osc_freq = freq(peak_idx)/synch_freq;
        osc_amp = peaks(peak_idx);
        datastruct.est_amp_array(p,k)  = osc_amp;
        datastruct.freq_array(p,k)     = osc_freq;
        fft_freqrange = freq;
        
        % Store values for spectrogram
%         if r3_range(p) == spectrogram_r3
        spect_peaks{p} = [spect_peaks{p}, peaks(1:n_freqs)];
        limit_pos{p} = [limit_pos{p}, position];
        limit_vel{p} = [limit_vel{p}, velocity];
%         figure(1)
            %plot(position, velocity)
%         end
        datastruct.lc_array(p,k).position = position;
        datastruct.lc_array(p,k).velocity = velocity;
        
        % Calculate empirical power requirements
        datastruct.conv_array(p,k) = calculate_power(velocity, net_force, position, t);
        
        % Calculate power ratio in dominant frequency
        
        [pxx, f] = periodogram(position, [], sampling_f);
        
        spectral_power = freq.^3.*peaks.^2;

        psd_ratio = max(spectral_power)/sum(spectral_power);
        
        datastruct.psd_array(p,k) = psd_ratio; % p=5, k=3 leads to some doubling
        
        % Keep track of peak force
        datastruct.force_array(p,k) = max(net_force);
        
        sim_count = sim_count + 1;
        
    end
    
end



k3 = 1;  
k4 = 1;
t_o = log( (k3*r3_range)/(k4*r3_range*r4_ratio) )./(r3_range-r3_range*r4_ratio); 
% disp('Double check t_o calc')
yax = f_n.*t_o;

try
    spect_keys = {'synch_gain_range', 'freq', 'spect_peaks'};
    spect_vals = {synch_gain_range, fft_freqrange, spect_peaks};
    datastruct.spect_data = containers.Map(spect_keys, spect_vals);
catch
    disp("couldn't save spectrum data")
end


try
    force_keys = {'synch_gain_range', 'yax', 'force_array'};
    force_vals = {synch_gain_range, yax, datastruct.force_array};
    datastruct.force_data = containers.Map(force_keys, force_vals);
catch
    disp("couldn't save force data")
end

try
    ofp_keys = {'synch_gain_range', 'yax', 'conv_array', 'freq_array', 'est_amp_array', 'psd_array'};
    ofp_vals = {synch_gain_range, yax, datastruct.conv_array, datastruct.freq_array, datastruct.est_amp_array, datastruct.psd_array};
    datastruct.ofp_data = containers.Map(ofp_keys, ofp_vals);
catch
    disp("couldn't save ofp_data")
end

try
    lc_keys = {'limit_pos', 'limit_vel', 'limit_t', 'synch_gain_range', 'freq_array', 'lc_array'};
    lc_vals = {limit_pos, limit_vel, t, synch_gain_range, datastruct.freq_array, datastruct.lc_array};
    datastruct.lc_data = containers.Map(lc_keys, lc_vals);
catch ME
    disp("couldn't save lc_data")
end

% function nUpdateWaitbar(~)
%     waitbar(sim_count/ntests, h);
%     sim_count = sim_count + 1;
% end

end
