% robobee_simulink_script.m
% Run the robobee simulink model

%% open model

model = "robobee_smoothSynchAsynch";
open_system(model)
%%
daq_to_piezo_volt = 25;     % 25 * daq_voltage

Frequency = 1;
Amplitude = 25/daq_to_piezo_volt;
VBias = 200/daq_to_piezo_volt;

samptime = 1/10000; % 10 kHz (hopefully)

t_end = 10;

set_param(model,"SimulationCommand","update")

set_param(model,"StopTime", num2str(t_end));

dataCell = cell(7,4);

%% Specify Single Test 

% freq_range = 20;

i = 1;
n = 1;

% Frequency (Hz)
% Frequency = freq_range(i);
Frequency = 66;

% Amplitude (V). VBias is set at 4 Volts, so
% max(Amp) = 4V. Don't exceed this!
Amplitude = 4;

% Runtime (s). Add 2 seconds on each end for ramp
% up to steady state
t_end = 6+4;

t_switch = t_end/2;
% t_switch = 50;

r3  = 1000;
kap = 0.62;
r4  = r3*kap;

a1 = r3-r4;
a2 = r3+r4;
a3 = r3*r4;

num = [a1 0];
den = [1 a2 a3];

% polynomial to convert from volts to displacement
% This one converts volts to inches
% Be sure to reset this with each new calibration
pf = [0.0008   -0.0042    0.0029];



% Run Single Test

% Update variables
set_param(model,"SimulationCommand","update");
% Set to run in real-time
set_param(model,"SimulationMode","external");
% Set runtime
set_param(model,"StopTime",num2str(t_end)); 
% Run model!
set_param(model,"SimulationCommand","start");

fprintf("Running: Amp = %1.1fV, Freq = %2.1fHz\n",Amplitude,Frequency);

%%

filename = sprintf("ScopeData_%dHz_t%d.mat",Frequency,n);

save(filename,"ScopeData");
clf
plot(ScopeData.time,ScopeData.signals(4).values)

% dataCell{i,n} = ScopeData
fprintf("Saved Test <%d,%d>\n",i,n)
%%
clf
for i = 1:11
    clf
    sDat = dataCell{i}
    
    t = sDat.time;
    pos = sDat.signals(4).values;
    
    trimPos = detrend(pos(20000:120000),1);
    trimT   = t(20000:120000);
    
    [yup,ylow] = envelope(trimPos,500,'peaks');
    
    hold on
    plot(trimT,trimPos)
    plot(trimT,yup)
    plot(trimT,ylow)
    yline(mean(yup))
    yline(mean(ylow))
    hold off
    drawnow()
%     pause()
    amp(i) = (mean(yup)-mean(ylow))/2;
    
end


