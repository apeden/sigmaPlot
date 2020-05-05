import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import statistics

ffi_case2_plate  = "CDI 11 018 FFI vs sCJD plate A_copy.xlsx"
ffi_case1_plate  = "CDI 11 019 FFI and LBD A regression analysis 0 baseline.xls"

df = pd.read_excel(ffi_case1_plate,
                    sheet_name='Plate',
                    usecols='A:G')

print(df)
df = df.iloc[6:9]
print(df)
raw_y_vals = df.to_numpy('float64')
print(raw_y_vals.dtype)
mean_y_vals = np.mean(raw_y_vals, axis = 0)
print(mean_y_vals)
stdev_y_vals = np.std(raw_y_vals, axis = 0)
print(stdev_y_vals)
print(mean_y_vals.dtype)
x_vals = np.array([0,0.25,0.5,0.75,1.0,1.5,2.0])
x_vals_for_fit = np.array([i/16 for i in range(32)])
print(x_vals)
print(x_vals.dtype)

## a = max, b = min, c= exponent, d = [GdnHCl]half
def f(x, a, b, c, d):
    return a / (1. + np.exp(-c * (x - d))) + b

def get_curve_feats(x_values, y_values): 
    starter_estimates = 15000, 2000, 4, 1
    (max_, min_, exponent, GdnHCl_half), covariance_of_optimised_param \
           = opt.curve_fit(f,
                           x_vals,
                           y_values,
                           p0 = starter_estimates,
                           maxfev = 10000)
    
    return {"GdnHCl half": GdnHCl_half,
            "max":max_,
            "min":min_,
            "exponent":exponent,
            "covariance of optimised parameters":covariance_of_optimised_param}

def get_mean_gdnHCl(x_values, y_values_array_2D):
    gdnHCl_half_vals = []
    for i in range(np.shape(y_values_array_2D)[0]):
        curve_feats  = get_curve_feats(x_values, y_values_array_2D[i])
        gdnHCl_half_vals.append(curve_feats["GdnHCl half"])
    return statistics.mean(gdnHCl_half_vals), statistics.stdev(gdnHCl_half_vals)

mean, std = get_mean_gdnHCl(x_vals, raw_y_vals)
print("Mean GdnHCl = "+str(mean) + "\nStd GdnHCl = "+str(std))

av_curve_feats = get_curve_feats(x_vals, mean_y_vals)
y_fit = f(x_vals_for_fit,
          av_curve_feats["max"],
          av_curve_feats["min"],
          av_curve_feats["exponent"],
          av_curve_feats["GdnHCl half"])
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(x_vals, mean_y_vals, 'o')
ax.plot(x_vals_for_fit, y_fit, '-')
ax.errorbar(x_vals, mean_y_vals, yerr = stdev_y_vals, fmt = 'o', capsize = 10)  
plt.title("Equilibrium unfolding of PrPSc in FFI FC\n[GdnHCl]1/2 = "+str(mean.round(3))+"M")
plt.ylabel("CDI value")
plt.xlabel("[GdnHCl] (M)")
plt.show()
