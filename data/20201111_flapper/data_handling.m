load roboflapperParamSweep_20x10_02to1.mat

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


%% how to pull out raw data

dat = raw_data{5,5};

time        = dat.time;
position    = dat.signals(1).values;
velocity    = dat.signals(2).values;
inTorque    = dat.signals(3).values;
asynchTorque= dat.signals(4).values;
synchTorque = dat.signals(5).values;


