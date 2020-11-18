load negVisc_mu035_cofreq1000.mat

trimInd = 100;
trimEnd = 79590;

time    = sdat.time(trimInd:trimEnd);
posV    = sdat.signals(1).values(trimInd:trimEnd); % position volts
velV    = sdat.signals(2).values(trimInd:trimEnd); % velocity Volts/s
torV    = sdat.signals(3).values(trimInd:trimEnd); % applied torque volts (saturated)

% clf
% plot(time-time(1),posV)

clf
subplot(1,2,1)
plot(posV-mean(posV),velV/100)
xlabel("Pos (volts)")
ylabel("Vel (normalized volts/s)")
title("Robobee - Negative viscosity oscillation (initial), Pos vs Vel")
subplot(1,2,2)
plot(torV,velV/100)
ylabel("Pos (volts)")
xlabel("Torque (normalized volts/s)")
title("Robobee - Negative viscosity oscillation (initial), Vel vs Torque")
% figure()
% for i = 1:3
%     subplot(3,1,i)
%     plot(time,sdat.signals(i).values)
%     
% end
