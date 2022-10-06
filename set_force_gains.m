function [sys_param,sim_param, asynch_peak, synch_peak] = set_force_gains(sys_param,sim_param)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% Since synch doesn't have dSA, we only need to run it once
synch_param = containers.Map(sim_param.keys, sim_param.values);
temp = sim_param('r3_range');
synch_param('r3_range') = temp(1);

% Run asynch and synch sims and get the optimization parameter (set in
% run_one_simulation.m) as an output
asynch_mag = run_one_simulation(sys_param, sim_param, 0);
synch_mag = run_one_simulation(sys_param, synch_param, 1);

asynch_peak = max(asynch_mag);
synch_peak = max(synch_mag);

% Compute the error between synch and asynch magnitudes. Update the asynch
% parameter to reduce err
%
% IMPORTANT: The convergence rate needs to be different depending on which
% optimization parameter you choose! Otherwise the tuning may not converge
% or the system could go unstable. 
%
% You might also need to change the inital value of asynch_gain in
% load_system_parameters.m

conv_rate = 10;
err = (synch_peak - asynch_peak)/synch_peak;
sys_param('asynch_gain') = sys_param('asynch_gain') * (1+err/conv_rate);

fprintf("err = %.4f, new asynch gain = %g\n",err,sys_param('asynch_gain'))
end

