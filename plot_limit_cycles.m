function plot_limit_cycles(lc_data)

pos = lc_data('limit_pos');
vel = lc_data('limit_vel');
t = lc_data('limit_t');
r3 = lc_data('spectrogram_r3');
synch_gain_range = lc_data('synch_gain_range');
freqs = lc_data('freq_array');


for i = 1:length(synch_gain_range)
    k = synch_gain_range(i);
    figure
    plot(pos(:,i), vel(:,i))
end
end

