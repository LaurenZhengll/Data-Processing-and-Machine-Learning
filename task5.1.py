# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 22:28:11 2020

@author: laure
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({'a': np.random.randint(0, 50, size=100)})
df['b'] = df['a'] + np.random.normal(0, 10, size=100)
df['c'] = 100 - 2*df['a'] + np.random.normal(0, 10, size=100)
df['d'] = np.random.randint(0, 50, 100) 

# calculate Pearson's-r coefficient and corrcoef
pearson_r1 = np.cov(df['a'], df['b'])[0, 1] / (df['a'].std() * df['b'].std())
print("a and b pearson_r:", pearson_r1)
corrcoef1=np.corrcoef(df['a'],df['b'])
print(corrcoef1)

pearson_r2 = np.cov(df['a'], df['c'])[0, 1] / (df['a'].std() * df['c'].std())
print("a and c pearson_r:", pearson_r2)
corrcoef2=np.corrcoef(df['a'],df['c'])
print(corrcoef2)

pearson_r3 = np.cov(df['a'], df['d'])[0, 1] / (df['a'].std() * df['d'].std())
print("a and d pearson_r:", pearson_r3)
corrcoef3=np.corrcoef(df['a'],df['d'])
print(corrcoef3)

# plotting
fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
# s: marker size; color: marker color; alpha: marker opacity.
ax.scatter(df['a'],df['b'], alpha=0.6, edgecolor='none', s=100)
ax.set_xlabel('a')
ax.set_ylabel('b')
# draw line
line_coef = np.polyfit(df['a'], df['b'], 1)
xx = np.arange(0, 50, 0.1)
yy = line_coef[0]*xx + line_coef[1]
ax.plot(xx, yy, 'green', lw=2)

# plotting
fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=100)
#s: marker size; color: marker color; alpha: marker opacity.
ax2.scatter(df['a'],df['c'], alpha=0.6, edgecolor='none', s=100)
ax2.set_xlabel('a')
ax2.set_ylabel('c')
# draw line
line_coef2 = np.polyfit(df['a'], df['c'], 1)
yy2 = line_coef2[0]*xx + line_coef2[1]
ax2.plot(xx, yy2, 'r', lw=2)

# plotting
fig3, ax3 = plt.subplots(figsize=(7, 5), dpi=100)
#s: marker size; color: marker color; alpha: marker opacity.
ax3.scatter(df['a'],df['d'], alpha=0.6, edgecolor='none', s=100)
ax3.set_xlabel('a')
ax3.set_ylabel('d')
# draw line
line_coef3 = np.polyfit(df['a'], df['d'], 1)
yy3 = line_coef3[0]*xx + line_coef3[1]
ax3.plot(xx, yy3, 'yellow', lw=2)

