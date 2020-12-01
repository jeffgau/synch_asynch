function r3_range = calculate_r3_range_flapper(ntests,t0_range,f_n, r4_ratio)


T_n = 1/f_n; % [s] - resonance period
t_o = logspace(t0_range(1), t0_range(2), ntests) * T_n;

r3_range = log(1/r4_ratio)./((1-r4_ratio)*t_o); 
end

