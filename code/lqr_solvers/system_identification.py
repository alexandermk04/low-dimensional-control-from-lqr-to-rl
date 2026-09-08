# For now adapted example from https://github.com/CPCLAB-UNIPI/SIPPY/tree/master

import sippy_unipi
print("sippy loaded from:", sippy_unipi.__file__)

from sippy_unipi import *

import matplotlib.pyplot as plt
import numpy as np

from sippy_unipi import functionset as fset
from sippy_unipi import functionsetSIM as fsetSIM

from config import DEFAULT_DT, DEFAULT_A, DEFAULT_B

dt = DEFAULT_DT

# Double Integrator system
A = DEFAULT_A
B = DEFAULT_B
C = np.eye(2)
D = np.array([[0.0]])

tfin = 20
npts = int(tfin / dt) + 1
Time = np.linspace(0, tfin, npts)

# Input sequence
U = np.zeros((1, npts))
U[0], _, _ = fset.GBN_seq(npts, 0.05)

## Output
x, yout = fsetSIM.SS_lsim_process_form(A, B, C, D, U)

#
plt.close("all")
plt.figure(0)
plt.plot(Time, U[0])
plt.ylabel("input")
plt.grid()
plt.xlabel("Time")
#
plt.figure(1)
plt.plot(Time, yout[0])
plt.ylabel("y_out")
plt.grid()
plt.xlabel("Time")
plt.title("Yout")

# System identification
METHOD = ["N4SID", "CVA", "MOESP", "PARSIM-S", "PARSIM-P", "PARSIM-K"]
lege = ["System"]
for i in range(len(METHOD)):
    method = METHOD[i]
    sys_id = system_identification(yout, U, method, SS_fixed_order=2)
    xid, yid = fsetSIM.SS_lsim_process_form(
        sys_id.A, sys_id.B, sys_id.C, sys_id.D, U, sys_id.x0
    )
    #
    plt.plot(Time, yid[0])
    plt.show()
    lege.append(method)
plt.legend(lege)