function save_ofp_data(ofp_data)

osc_amp = ofp_data('est_amp_array');
freq = ofp_data('freq_array');
power = ofp_data('conv_array');
synch_gain_range = ofp_data('synch_gain_range');
yax = ofp_data('yax');
save('data/ofp_data', 'ofp_data');

save('data/r3_freq/osc_amp', 'osc_amp')
save('data/r3_freq/freq', 'freq');
save('data/power', 'power');
save('data/synch_gain_range', 'synch_gain_range')
save('data/yax', 'yax')
end

