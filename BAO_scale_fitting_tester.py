'''
Coding up our solution to the BAO Fast Scaling problem. 
Created October 10, 2021
Author(s): Matt Hansen, Alex Krolewski
'''
import BAO_scale_fitting
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter


alphas = np.linspace(0.9, 1.1, 100)

recovered = np.zeros(len(alphas))

running = False

for i in range(len(alphas)):
    if running:
        recovered[i] = BAO_scale_fitting.marginal_linear_bias_second_order(alphas[i]) + 1
    else:
        recovered[i] = BAO_scale_fitting.fixed_linear_bias(alphas[i]) + 1
        
    print('Recovered alpha: ', recovered[i])
    print()


difference = recovered - alphas

difference_error = difference / alphas

#np.savetxt('recovered_alpha_marg_12-11-21.txt', recovered)
#np.savetxt('Marginalized_Predicted_100_iterations.txt', recovered)

fig, ax = plt.subplots()
plt.plot(alphas, recovered, 'r', label=r'Recovered', linewidth=3.0)
plt.plot(alphas, alphas, color='black', linestyle='dotted', label=r'True $\alpha$', linewidth=2.0)
plt.xlabel(r'True $\alpha$ ($\alpha_T$)', fontsize=20)
plt.ylabel(r'Recovered $\alpha$ $(\alpha_R)$', fontsize=20)
if running:
    plt.title(r'Marginalized $\mathcal{B}$', fontsize=20)
else:
    plt.title(r'Fixed $\mathcal{B}$', fontsize=20)
plt.legend(fontsize=15)
plt.xlim(0.9, 1.1)
plt.ylim(0.85, 1.2)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
for tick in plt.gca().xaxis.get_major_ticks():
    tick.label.set_fontsize(15)
for tick in plt.gca().yaxis.get_major_ticks():
    tick.label.set_fontsize(15)
for label in plt.gca().xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
for label in plt.gca().yaxis.get_ticklabels()[::2]:
    label.set_visible(False)
plt.grid()
plt.tight_layout()
#plt.savefig('Marginal_B_With_Second_Plot_12-10-21.pdf')
plt.show()