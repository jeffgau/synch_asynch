function [force_array, conv_array, freq_array, est_amp_array, spect_data, force_data, ofp_data, lc_data] = run_simulation(sys_param, sim_param)

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


freq_array      = zeros(ntests,ntests);
conv_array      = zeros(ntests,ntests);
est_amp_array   = zeros(ntests,ntests);
force_array = zeros(ntests,ntests);



sim_count = 1;

% Set variables for plotting spectrogram and limit cycles
spectrogram_r3 = sim_param('r3_spect');
spect_peaks = [];
n_freqs = 2000; % Number of fourier modes to plot
limit_pos = [];
limit_vel = [];
limit_t = [];

r4_ratio = 0.1;


for p = 1:ntests
    
    for k = 1:ntests % Load variables for parallel sim
        p, k
        
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
        [freq, peaks] = fourier_analysis(position - mean(position), sampling_f);
        peak_idx = find(peaks == max(peaks));
        osc_freq = freq(peak_idx)/synch_freq;
        osc_amp = peaks(peak_idx);
        est_amp_array(p,k)  = osc_amp;
        freq_array(p,k)     = osc_freq;
        
        % Store values for spectrogram
        if r3_range(p) == spectrogram_r3
            spect_peaks = [spect_peaks, peaks(1:n_freqs)];
            limit_pos = [limit_pos, position];
            limit_vel = [limit_vel, velocity];
            figure(1)
            plot(position, velocity)
        end
        
        % Calculate empirical power requirements
        conv_array(p,k) = calculate_power(velocity, net_force, position, t);
        
        % Keep track of peak force
        force_array(p,k) = max(net_force);
        
        sim_count = sim_count + 1;
        
    end
end


spect_keys = {'synch_gain_range', 'freq', 'spect_peaks'};
spect_vals = {synch_gain_range, freq, spect_peaks};
spect_data = containers.Map(spect_keys, spect_vals);

k3 = 1;  
k4 = 1;
t_o = log( (k3*r3_range)/(k4*r3_range*r4_ratio) )./(r3_range-r3_range*r4_ratio); 
disp('Double check t_o calc')
yax = f_n.*t_o;

force_keys = {'synch_gain_range', 'yax', 'force_array'};
force_vals = {synch_gain_range, yax, force_array};
force_data = containers.Map(force_keys, force_vals);


ofp_keys = {'synch_gain_range', 'yax', 'conv_array', 'freq_array', 'est_amp_array'};
ofp_vals = {synch_gain_range, yax, conv_array, freq_array, est_amp_array};
ofp_data = containers.Map(ofp_keys, ofp_vals);

lc_keys = {'limit_pos', 'limit_vel', 'limit_t', 'spectrogram_r3', 'synch_gain_range', 'freq_array'};
lc_vals = {limit_pos, limit_vel, t, spectrogram_r3, synch_gain_range, freq_array};
lc_data = containers.Map(lc_keys, lc_vals);
    

end
