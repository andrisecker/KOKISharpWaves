function I = inhom_poisson(phase0, nPop, pop, seed)

lambda = 20;
Tmax = 500;

T = [];
rand('twister', seed)
T(1) = exprnd(1 / lambda);
i = 1;
while T(i) < Tmax
   rand('twister', seed + i)
   T(i+1) = T(i) + exprnd(1 / lambda);
   i = i + 1;
end

T = T(1:end-1);
 
I=[];
for i = 1:size(T,2)
    rand('twister', seed + i)
    if (gen_firingrate(phase0, nPop, pop, T(i)) / lambda) >= rand(1)
       I = [I T(i)];
    end
end