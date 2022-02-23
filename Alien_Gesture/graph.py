import matplotlib.pyplot as plt
import numpy as np

N = 3
ours = (263.0, 263.85, 269.97)
our_err = (0.014 * 300, 0.014 * 300, 0.013 * 300)
contact = (134.46, 193.19, 262.25)
cotact_err = (0.015 * 300, 0.017 * 300, 0.014 * 300)
temporal = (102.5, 153.49, 31.33)
temporal_err = (0.012 * 300, 0.016 * 300, 0.002 * 300)
dr = (188.7, 250.975, 119.01)
dr_err = (0.018 * 300, 0.015 * 300, 0.014 * 300)

ind = np.arange(N) 
width = 0.2   
plt.bar(ind, ours, width, yerr = our_err, color = '#38404E',  label='Ours')
plt.bar(ind + width, contact, width, yerr = cotact_err,color = '#D4C5B8', label='Without contact consistency')
plt.bar(ind + 2 * width, temporal, width, yerr = temporal_err, color = '#D37879', label='Without temporal consistency')
plt.bar(ind + 3 * width, dr, width,  yerr = dr_err, color = '#F1BABB', label='Without domain randomization')

plt.ylim(0, 350)

plt.ylabel('Success time ratio', fontsize=30, fontfamily = 'serif')
# plt.ylim(300)
plt.xticks(ind + width * 3 /2, ('Stand', 'Sit', 'Walk'), fontsize=24, fontfamily = 'serif')
plt.yticks((0, 300), ('0', '1'), fontsize=30, fontfamily = 'serif')
plt.legend(fontsize=20, loc='best')
plt.show()