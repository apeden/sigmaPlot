import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

##array of x values
##array of y values
##function that returns a,b,c and d values given x and y values
##function to plot results (as individual points)....
##.....overlayed with sigma model of data

df = pd.read_excel("CDI#13.028A Analysis.xlsx",
                    sheet_name='Sheet1',
                    usecols='C:K')

print(df)
df = df.iloc[4:7]
print(df)
raw_y_vals = df.to_numpy('float64')
print(raw_y_vals.dtype)
mean_y_vals = np.mean(raw_y_vals, axis = 0)
print(mean_y_vals)
print(mean_y_vals.dtype)
x_vals = np.array([i/2 for i in range(9)])
x_vals_for_fit = np.array([i/8 for i in range(36)])
print(x_vals)
print(x_vals.dtype)

## a = max, b = min, c= exponent, d = [GdnHCl]half
def f(x, a, b, c, d):
    return a / (1. + np.exp(-c * (x - d))) + b

starter_estimates = 10000, 1000, 5, 2
(max_, min_, exponent, GdnHCl_half), covariance_of_optimised_param \
       = opt.curve_fit(f,
                       x_vals,
                       mean_y_vals,
                       p0 = starter_estimates)

print("MaxVal = ", max_, "\nMin Val = ", min_,
      "\nExponent = ", exponent, "\nGdnHCl_half = ", GdnHCl_half)
print(covariance_of_optimised_param)

y_fit = f(x_vals_for_fit, max_, min_, exponent, GdnHCl_half)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(x_vals, mean_y_vals, 'o')
ax.plot(x_vals_for_fit, y_fit, '-')
plt.title("Equilibrium unfolding of PrPSc in VPSPr FC")
plt.ylabel("CDI value")
plt.xlabel("GdnHCl (M)")
plt.show()
