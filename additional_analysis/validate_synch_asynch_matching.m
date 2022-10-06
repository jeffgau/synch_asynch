clear
system = 'moth'; % moth, flapper, or robobee
tic % Typically ~0.5-1s per moth simulation

%% Load System and Simulation parameters as Map containers
sys_param = load_system_param(system);

sim_param = load_simulation_param(system, sys_param('f_n'));

%% Look at Synchronous forcing
synch_param = sim_param;

% assign arbitrary r3 (won't come into play)
synch_param('r3_range') = 1;

% run a synchronous simulation

opt_param = run_one_simulation(sys_param,synch_param,1)