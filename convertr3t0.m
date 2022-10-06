function [t_0, yax] = convertr3t0(r3_range,r4_ratio,f_n)

k3 = 1;  
k4 = 1;
t_0 = log( (k3*r3_range)/(k4*r3_range*r4_ratio) )./(r3_range-r3_range*r4_ratio); 

% make y-axis, t_
yax = f_n.*t_0;

end