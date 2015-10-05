function lambda = gen_firingrate(phase0, nPop, pop, t)

theta_fr = 7;      
rate_infield = 20;     
v_eger = 32.43567842; 
l_pf = 30;   
l_route = 300; 

time_route = l_route / v_eger;  
R = 300 / (2*pi); 

w_eger = 2*pi / time_route;    
x = mod(w_eger*t,2*pi); 

fi_pf_rad = l_pf/R;     
fi_start = (pop-1) * nPop^(-1) * 2*pi;
fi_end = fi_start + fi_pf_rad;

y = phase0 + 2*pi * theta_fr * t;
shift = fi_start + fi_pf_rad / 2;
m =  - (x-fi_start)*2*pi/fi_pf_rad;

sigma = 0.5;
s = 1 / sigma;

if ((x>=fi_start)  &&  (x<fi_end))   
    lambda1 = cos(2*pi/(2*fi_pf_rad) * (x - shift )) * rate_infield; 
    lambda2 = exp(s * cos(y-m)) / exp(s); 
else
    lambda1 = 0;      
    lambda2 = 1;
end
      
lambda = lambda1 * lambda2;