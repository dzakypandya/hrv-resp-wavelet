import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.title("Plot HRV and Respiratory Signal")

# ===== File path and column names ===== #
file_path = r"samples.txt"
column_names = ['ElapsedTime', 'RESP', 'PLETH', 'V', 'AVR', 'II']

# ===== Read data ===== #
df = pd.read_csv(file_path, sep='\t', skiprows=2, names=column_names)

# ===== Time conversion: mm:ss.sss to seconds ===== #
def time_to_seconds(t):
    try:
        m, s = t.strip().split(':')
        return int(m) * 60 + float(s)
    except:
        return None

df['Time (s)'] = df['ElapsedTime'].apply(time_to_seconds)

# ===== Clean data ===== #
df = df.dropna(subset=['Time (s)', 'RESP', 'II', 'AVR', 'V'])
df[['RESP', 'II', 'AVR', 'V']] = df[['RESP', 'II', 'AVR', 'V']].apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# ===== Plotting ===== # 
st.subheader("Plot Sinyal ECG dan Respiratory Signal Original")
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
axes[0].plot(df['Time (s)'], df['II'], label='Sinyal ECG')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitudo (mV)')
axes[0].legend()

axes[1].plot(df['Time (s)'], df['RESP'], label='Respiratory Signal')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitudo (mV)')
axes[1].legend()

st.pyplot(fig)

# ===== BASELINE ===== #
# Function to shift ECG data to baseline using polynomial regression
def baseline_shift(ecg_data, degree):
    x = np.arange(len(ecg_data))
    # Fit polynomial of given degree to the data
    p = np.polyfit(x, ecg_data, degree)
    # Evaluate the polynomial
    trend = np.polyval(p, x)
    # Subtract the polynomial trend from the original data
    baseline_corrected = ecg_data - trend
    return baseline_corrected

ecg_base = df['II']
resp_base = df['RESP']
t = df['Time (s)']

# Apply baseline correction
ecg = baseline_shift(ecg_base, 2)
resp = baseline_shift(resp_base, 2)

#============ plot nak streamlit ==============#
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
axes[0].plot(t, ecg_base, label='Sinyal ECG')
axes[0].plot(t, ecg, label='Baselined Sinyal ECG')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitudo (mV)')
axes[0].legend()

axes[1].plot(t, resp_base, label='Respiratory Signal')
axes[1].plot(t, resp, label='Baselined Respiratory Signal')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitudo (mV)')
axes[1].legend()

st.pyplot(fig)


# ===== cari filter koef ===== #
def dirac(x):
  if(x==0):
    dirac_delta = 1
  else:
    dirac_delta = 0
  result = dirac_delta
  return result

h = []
g = []
n_list = []
for n in range (-2, 2):
  n_list.append(n)
  temp_h = 1/8 * (dirac(n-1) + 3*dirac(n) + 3*dirac(n+1) + dirac(n+2))
  h.append(temp_h)
  temp_g = -2 * (dirac(n) - dirac(n+1))
  g.append(temp_g)

Hw = np.zeros(20000)
Gw = np.zeros(20000)
fs = 125
i_list = []
for i in range (0, fs + 1):
  i_list.append(i)
  reG = 0
  imG = 0
  reH = 0
  imH = 0
  for k in range (-2, 2):
    reG = reG + g[k + abs(-2)] * np.cos(k * 2 * np.pi * i/fs)
    imG = imG - g[k + abs(-2)] * np.sin(k * 2 * np.pi * i/fs)
    reH = reH + h[k + abs(-2)] * np.cos(k * 2 * np.pi * i/fs)
    imH = imH - h[k + abs(-2)] * np.sin(k * 2 * np.pi * i/fs)
  temp_Hw = np.sqrt( (reH ** 2) + (imH ** 2))
  temp_Gw = np.sqrt( (reG ** 2) + (imG ** 2))
  Hw[i] = temp_Hw
  Gw[i] = temp_Gw
  i_list = i_list[0:round(fs/2)+1]

#algorithm mallat
Q = np.zeros((9, round(fs/2)+1))
i_list = []
for i in range(0, round(fs/2)+1):
  i_list.append(i)
  Q[1][i] = Gw[i]
  Q[2][i] = Gw[2*i]*Hw[i]
  Q[3][i] = Gw[4*i]*Gw[2*i]*Hw[i]
  Q[4][i] = Gw[8*i]*Gw[4*i]*Gw[2*i]*Hw[i]
  Q[5][i] = Gw[16*i]*Gw[8*i]*Gw[4*i]*Gw[2*i]*Hw[i]
  Q[6][i] = Gw[32*i]*Gw[16*i]*Gw[8*i]*Gw[4*i]*Gw[2*i]*Hw[i]
  Q[7][i] = Gw[64*i]*Gw[32*i]*Gw[16*i]*Gw[8*i]*Gw[4*i]*Gw[2*i]*Hw[i]
  Q[8][i] = Gw[128*i]*Gw[64*i]*Gw[32*i]*Gw[16*i]*Gw[8*i]*Gw[4*i]*Gw[2*i]*Hw[i]

st.subheader("Grafik Qj(f)")
fig, ax = plt.subplots()

for i in range(1, 9):
    line_label = "Q{}".format(i)
    ax.plot(i_list, Q[i], label=line_label)

ax.legend()
st.pyplot(fig)

qj = np.zeros((9, 10000))

k_list = []

# ===== skala 1 ===== #
j = 1
a = -(round(2**j) + round(2**(j-1)) - 2)
b = -(1 - round(2**(j-1))) + 1

for k in range (a, b):
  k_list.append(k)
  qj[1][k + abs(a)] = -2 * (dirac(k) - dirac(k+1))

kernel_j1 = qj[1][0:len(k_list)]
ecg_j1 = np.convolve(ecg, kernel_j1, mode='same')
resp_j1 = np.convolve(resp, kernel_j1, mode='same')

# ===== skala 2 ===== #
j = 2
a = -(round(2**j) + round(2**(j-1)) - 2)
b = -(1 - round(2**(j-1))) + 1

for k in range (a, b):
  k_list.append(k)
  qj[2][k + abs(a)] = -1/4 * (dirac(k-1) + 3*dirac(k) + 2*dirac(k+1 - 2*dirac(k+2))
                      - 3*dirac(k+3) - dirac(k+4))

kernel_j2 = qj[2][0:len(k_list)]
ecg_j2 = np.convolve(ecg, kernel_j2, mode='same')
resp_j2 = np.convolve(resp, kernel_j2, mode='same')

# ===== skala 3 ===== #
j = 3
a = -(round(2**j) + round(2**(j-1)) - 2)
b = -(1 - round(2**(j-1))) + 1

for k in range (a, b):
  k_list.append(k)
  qj[3][k + abs(a)] = -1/32 * (dirac(k-3) + 3*dirac(k-2) + 6*dirac(k-1) - 10*dirac(k)
                      + 11*dirac(k+1) + 9*dirac(k+2) + 4*dirac(k+3) - 4*dirac(k+4)
                      - 9*dirac(k+5) - 11*dirac(k+6) - 10*dirac(k+7) - 6*dirac(k+8)
                      - 3*dirac(k+9) - dirac(k+10))

kernel_j3 = qj[3][0:len(k_list)]
ecg_j3 = np.convolve(ecg, kernel_j3, mode='same')
resp_j3 = np.convolve(resp, kernel_j3, mode='same')

# ===== skala 4 ===== #
j = 4
a = -(round(2**j) + round(2**(j-1)) - 2)
b = -(1 - round(2**(j-1))) + 1

for k in range (a, b):
  k_list.append(k)
  qj[4][k + abs(a)] = -1/256 * (dirac(k-7) + 3*dirac(k-6) + 6*dirac(k-5) + 10*dirac(k-4)
                      + 15*dirac(k-3) + 21*dirac(k-2) + 28*dirac(k-1) + 36*dirac(k)
                      + 41*dirac(k+1) + 43*dirac(k+2) + 42*dirac(k+3) + 38*dirac(k+4)
                      + 31*dirac(k+5) + 21*dirac(k+6) + 8*dirac(k+7) - 8*dirac(k+8)
                      - 21*dirac(k+9) - 31*dirac(k+10) - 38*dirac(k+11) - 42*dirac(k+12)
                      - 43*dirac(k+13) - 41*dirac(k+14) - 36*dirac(k+15) - 28*dirac(k+16)
                      - 21*dirac(k+17) - 15*dirac(k+18) - 10*dirac(k+19) - 6*dirac(k+20)
                      - 3*dirac(k+21) - dirac(k+22))

kernel_j4 = qj[4][0:len(k_list)]
ecg_j4 = np.convolve(ecg, kernel_j4, mode='same')
resp_j4 = np.convolve(resp, kernel_j4, mode='same')

# ===== skala 5 ===== #
j = 5
a = -(round(2**j) + round(2**(j-1)) - 2)
b = -(1 - round(2**(j-1))) + 1

for k in range (a, b):
  k_list.append(k)
  qj[5][k + abs(a)] = -1/(512)*(dirac(k-15) + 3*dirac(k-14) + 6*dirac(k-13) + 10*dirac(k-12) + 15*dirac(k-11) + 21*dirac(k-10)
                      + 28*dirac(k-9) + 36*dirac(k-8) + 45*dirac(k-7) + 55*dirac(k-6) + 66*dirac(k-5) + 78*dirac(k-4)
                      + 91*dirac(k-3) + 105*dirac(k-2) + 120*dirac(k-1) + 136*dirac(k) + 149*dirac(k+1) + 159*dirac(k+2)
                      + 166*dirac(k+3) + 170*dirac(k+4) + 171*dirac(k+5) + 169*dirac(k+6) + 164*dirac(k+7) + 156*dirac(k+8)
                      + 145*dirac(k+9) + 131*dirac(k+10) + 114*dirac(k+11) + 94*dirac(k+12) + 71*dirac(k+13) + 45*dirac(k+14)
                      + 16*dirac(k+15) - 16*dirac(k+16) - 45*dirac(k+17) - 71*dirac(k+18) - 94*dirac(k+19) - 114*dirac(k+20)
                      - 131*dirac(k+21) - 145*dirac(k+22) - 156*dirac(k+23) - 164*dirac(k+24) - 169*dirac(k+25)
                      - 171*dirac(k+26) - 170*dirac(k+27) - 166*dirac(k+28) - 159*dirac(k+29) - 149*dirac(k+30)
                      - 136*dirac(k+31) - 120*dirac(k+32) - 105*dirac(k+33) - 91*dirac(k+34) - 78*dirac(k+35)
                      - 66*dirac(k+36) - 55*dirac(k+37) - 45*dirac(k+38) - 36*dirac(k+39) - 28*dirac(k+40)
                      - 21*dirac(k+41) - 15*dirac(k+42) - 10*dirac(k+43) - 6*dirac(k+44) - 3*dirac(k+45)
                      - dirac(k+46))

kernel_j5 = qj[5][0:len(k_list)]
ecg_j5 = np.convolve(ecg, kernel_j5, mode='same')
resp_j5 = np.convolve(resp, kernel_j5, mode='same')

# ===== skala 6 ===== #
j = 6
a = -(round(2**j) + round(2**(j-1)) - 2)
b = -(1 - round(2**(j-1))) + 1

for k in range(a, b):
    k_list.append(k)
    qj[6][k + abs(a)] = -1 / (16384) * (dirac(k - 31) + 3 * dirac(k - 30) + 6 * dirac(k - 29) + 10 * dirac(k - 28) +
                        15 * dirac(k - 27) + 21 * dirac(k - 26) + 28 * dirac(k - 25) + 36 * dirac(k - 24) +
                        45 * dirac(k - 23) + 55 * dirac(k - 22) + 66 * dirac(k - 21) + 78 * dirac(k - 20) +
                        91 * dirac(k - 19) + 105 * dirac(k - 18) + 120 * dirac(k - 17) + 136 * dirac(k - 16) +
                        153 * dirac(k - 15) + 171 * dirac(k - 14) + 190 * dirac(k - 13) + 210 * dirac(k - 12) +
                        231 * dirac(k - 11) + 253 * dirac(k - 10) + 276 * dirac(k - 9) + 300 * dirac(k - 8) +
                        325 * dirac(k - 7) + 351 * dirac(k - 6) + 378 * dirac(k - 5) + 406 * dirac(k - 4) +
                        435 * dirac(k - 3) + 465 * dirac(k - 2) + 496 * dirac(k - 1) + 528 * dirac(k) +
                        557 * dirac(k + 1) + 583 * dirac(k + 2) + 606 * dirac(k + 3) + 626 * dirac(k + 4) +
                        643 * dirac(k + 5) + 657 * dirac(k + 6) + 668 * dirac(k + 7) + 676 * dirac(k + 8) +
                        681 * dirac(k + 9) + 683 * dirac(k + 10) + 682 * dirac(k + 11) + 678 * dirac(k + 12) +
                        671 * dirac(k + 13) + 661 * dirac(k + 14) + 648 * dirac(k + 15) + 632 * dirac(k + 16) +
                        613 * dirac(k + 17) + 591 * dirac(k + 18) + 566 * dirac(k + 19) + 538 * dirac(k + 20) +
                        507 * dirac(k + 21) + 473 * dirac(k + 22) + 436 * dirac(k + 23) + 396 * dirac(k + 24) +
                        353 * dirac(k + 25) + 307 * dirac(k + 26) + 258 * dirac(k + 27) + 206 * dirac(k + 28) +
                        151 * dirac(k + 29) + 93 * dirac(k + 30) + 32 * dirac(k + 31) - 32 * dirac(k + 32) -
                        93 * dirac(k + 33) - 151 * dirac(k + 34) - 206 * dirac(k + 35) - 258 * dirac(k + 36) -
                        307 * dirac(k + 37) - 353 * dirac(k + 38) - 396 * dirac(k + 39) - 436 * dirac(k + 40) -
                        473 * dirac(k + 41) - 507 * dirac(k + 42) - 538 * dirac(k + 43) - 566 * dirac(k + 44) -
                        591 * dirac(k + 45) - 613 * dirac(k + 46) - 632 * dirac(k + 47) - 648 * dirac(k + 48) -
                        661 * dirac(k + 49) - 671 * dirac(k + 50) - 678 * dirac(k + 51) - 682 * dirac(k + 52) -
                        683 * dirac(k + 53) - 681 * dirac(k + 54) - 676 * dirac(k + 55) - 668 * dirac(k + 56) -
                        657 * dirac(k + 57) - 643 * dirac(k + 58) - 626 * dirac(k + 59) - 606 * dirac(k + 60) -
                        583 * dirac(k + 61) - 557 * dirac(k + 62) - 528 * dirac(k + 63) - 496 * dirac(k + 64) -
                        465 * dirac(k + 65) - 435 * dirac(k + 66) - 406 * dirac(k + 67) - 378 * dirac(k + 68) -
                        351 * dirac(k + 69) - 325 * dirac(k + 70) - 300 * dirac(k + 71) - 276 * dirac(k + 72) -
                        253 * dirac(k + 73) - 231 * dirac(k + 74) - 210 * dirac(k + 75) - 190 * dirac(k + 76) -
                        171 * dirac(k + 77) - 153 * dirac(k + 78) - 136 * dirac(k + 79) - 120 * dirac(k + 80) -
                        105 * dirac(k + 81) - 91 * dirac(k + 82) - 78 * dirac(k + 83) - 66 * dirac(k + 84) -
                        55 * dirac(k + 85) - 45 * dirac(k + 86) - 36 * dirac(k + 87) - 28 * dirac(k + 88) -
                        21 * dirac(k + 89) - 15 * dirac(k + 90) - 10 * dirac(k + 91) - 6 * dirac(k + 92) -
                        3 * dirac(k + 93) - dirac(k + 94))

kernel_j6 = qj[6][0:len(k_list)]
ecg_j6 = np.convolve(ecg, kernel_j6, mode='same')
resp_j6 = np.convolve(resp, kernel_j6, mode='same')

# ===== skala 7 ===== #
j = 7
a = -(round(2**j) + round(2**(j-1)) - 2)
b = -(1 - round(2**(j-1))) + 1

for k in range(a, b):
    k_list.append(k)
    qj[7][k + abs(a)] = -1 / 131072 * (
                        dirac(k-63) + 3*dirac(k-62) + 6*dirac(k-61) + 10*dirac(k-60) + 15*dirac(k-59) +
                        21*dirac(k-58) + 28*dirac(k-57) + 36*dirac(k-56) + 45*dirac(k-55) +
                        55*dirac(k-54) + 66*dirac(k-53) + 78*dirac(k-52) + 91*dirac(k-51) +
                        105*dirac(k-50) + 120*dirac(k-49) + 136*dirac(k-48) + 153*dirac(k-47) +
                        171*dirac(k-46) + 190*dirac(k-45) + 210*dirac(k-44) + 231*dirac(k-43) +
                        253*dirac(k-42) + 276*dirac(k-41) + 300*dirac(k-40) + 325*dirac(k-39) +
                        351*dirac(k-38) + 378*dirac(k-37) + 406*dirac(k-36) + 435*dirac(k-35) +
                        465*dirac(k-34) + 496*dirac(k-33) + 528*dirac(k-32) + 561*dirac(k-31) +
                        595*dirac(k-30) + 630*dirac(k-29) + 666*dirac(k-28) + 703*dirac(k-27) +
                        741*dirac(k-26) + 780*dirac(k-25) + 820*dirac(k-24) + 861*dirac(k-23) +
                        903*dirac(k-22) + 946*dirac(k-21) + 990*dirac(k-20) + 1035*dirac(k-19) +
                        1081*dirac(k-18) + 1128*dirac(k-17) + 1176*dirac(k-16) + 1225*dirac(k-15) +
                        1275*dirac(k-14) + 1326*dirac(k-13) + 1378*dirac(k-12) + 1431*dirac(k-11) +
                        1485*dirac(k-10) + 1540*dirac(k-9) + 1596*dirac(k-8) + 1653*dirac(k-7) +
                        1711*dirac(k-6) + 1770*dirac(k-5) + 1830*dirac(k-4) + 1891*dirac(k-3) +
                        1953*dirac(k-2) + 2016*dirac(k-1) + 2080*dirac(k) + 2141*dirac(k+1) +
                        2199*dirac(k+2) + 2254*dirac(k+3) + 2306*dirac(k+4) + 2355*dirac(k+5) +
                        2401*dirac(k+6) + 2444*dirac(k+7) + 2484*dirac(k+8) + 2521*dirac(k+9) +
                        2555*dirac(k+10) + 2586*dirac(k+11) + 2614*dirac(k+12) + 2639*dirac(k+13) +
                        2661*dirac(k+14) + 2680*dirac(k+15) + 2696*dirac(k+16) + 2709*dirac(k+17) +
                        2719*dirac(k+18) + 2726*dirac(k+19) + 2730*dirac(k+20) + 2731*dirac(k+21) +
                        2729*dirac(k+22) + 2724*dirac(k+23) + 2716*dirac(k+24) + 2705*dirac(k+25) +
                        2691*dirac(k+26) + 2674*dirac(k+27) + 2654*dirac(k+28) + 2631*dirac(k+29) +
                        2605*dirac(k+30) + 2576*dirac(k+31) + 2544*dirac(k+32) + 2509*dirac(k+33) +
                        2471*dirac(k+34) + 2430*dirac(k+35) + 2386*dirac(k+36) + 2339*dirac(k+37) +
                        2289*dirac(k+38) + 2236*dirac(k+39) + 2180*dirac(k+40) + 2121*dirac(k+41) +
                        2059*dirac(k+42) + 1994*dirac(k+43) + 1926*dirac(k+44) + 1855*dirac(k+45) +
                        1781*dirac(k+46) + 1704*dirac(k+47) + 1624*dirac(k+48) + 1541*dirac(k+49) +
                        1455*dirac(k+50) + 1366*dirac(k+51) + 1274*dirac(k+52) + 1179*dirac(k+53) +
                        1081*dirac(k+54) + 980*dirac(k+55) + 876*dirac(k+56) + 769*dirac(k+57) +
                        659*dirac(k+58) + 546*dirac(k+59) + 430*dirac(k+60) + 311*dirac(k+61) +
                        189*dirac(k+62) + 64*dirac(k+63) - 64*dirac(k+64) - 189*dirac(k+65) -
                        311*dirac(k+66) - 430*dirac(k+67) - 546*dirac(k+68) - 659*dirac(k+69) -
                        769*dirac(k+70) - 876*dirac(k+71) - 980*dirac(k+72) - 1081*dirac(k+73) -
                        1179*dirac(k+74) - 1274*dirac(k+75) - 1366*dirac(k+76) - 1455*dirac(k+77) -
                        1541*dirac(k+78) - 1624*dirac(k+79) - 1704*dirac(k+80) - 1781*dirac(k+81) -
                        1855*dirac(k+82) - 1926*dirac(k+83) - 1994*dirac(k+84) - 2059*dirac(k+85) -
                        2121*dirac(k+86) - 2180*dirac(k+87) - 2236*dirac(k+88) - 2289*dirac(k+89) -
                        2339*dirac(k+90) - 2386*dirac(k+91) - 2430*dirac(k+92) - 2471*dirac(k+93) -
                        2509*dirac(k+94) - 2544*dirac(k+95) - 2576*dirac(k+96) - 2605*dirac(k+97) -
                        2631*dirac(k+98) - 2654*dirac(k+99) - 2674*dirac(k+100) - 2691*dirac(k+101) -
                        2705*dirac(k+102) - 2716*dirac(k+103) - 2724*dirac(k+104) - 2729*dirac(k+105) -
                        2731*dirac(k+106) - 2730*dirac(k+107) - 2726*dirac(k+108) - 2719*dirac(k+109) -
                        2709*dirac(k+110) - 2696*dirac(k+111) - 2680*dirac(k+112) - 2661*dirac(k+113) -
                        2639*dirac(k+114) - 2614*dirac(k+115) - 2586*dirac(k+116) - 2555*dirac(k+117) -
                        2521*dirac(k+118) - 2484*dirac(k+119) - 2444*dirac(k+120) - 2401*dirac(k+121) -
                        2355*dirac(k+122) - 2306*dirac(k+123) - 2254*dirac(k+124) - 2199*dirac(k+125) -
                        2141*dirac(k+126) - 2080*dirac(k+127) - 2016*dirac(k+128) - 1953*dirac(k+129) -
                        1891*dirac(k+130) - 1830*dirac(k+131) - 1770*dirac(k+132) - 1711*dirac(k+133) -
                        1653*dirac(k+134) - 1596*dirac(k+135) - 1540*dirac(k+136) - 1485*dirac(k+137) -
                        1431*dirac(k+138) - 1378*dirac(k+139) - 1326*dirac(k+140) - 1275*dirac(k+141) -
                        1225*dirac(k+142) - 1176*dirac(k+143) - 1128*dirac(k+144) - 1081*dirac(k+145) -
                        1035*dirac(k+146) - 990*dirac(k+147) - 946*dirac(k+148) - 903*dirac(k+149) -
                        861*dirac(k+150) - 820*dirac(k+151) - 780*dirac(k+152) - 741*dirac(k+153) -
                        703*dirac(k+154) - 666*dirac(k+155) - 630*dirac(k+156) - 595*dirac(k+157) -
                        561*dirac(k+158) - 528*dirac(k+159) - 496*dirac(k+160) - 465*dirac(k+161) -
                        435*dirac(k+162) - 406*dirac(k+163) - 378*dirac(k+164) - 351*dirac(k+165) -
                        325*dirac(k+166) - 300*dirac(k+167) - 276*dirac(k+168) - 253*dirac(k+169) -
                        231*dirac(k+170) - 210*dirac(k+171) - 190*dirac(k+172) - 171*dirac(k+173) -
                        153*dirac(k+174) - 136*dirac(k+175) - 120*dirac(k+176) - 105*dirac(k+177) -
                        91*dirac(k+178) - 78*dirac(k+179) - 66*dirac(k+180) - 55*dirac(k+181) -
                        45*dirac(k+182) - 36*dirac(k+183) - 28*dirac(k+184) - 21*dirac(k+185) -
                        15*dirac(k+186) - 10*dirac(k+187) - 6*dirac(k+188) - 3*dirac(k+189) -
                        dirac(k+190))

kernel_j7 = qj[7][0:len(k_list)]
ecg_j7 = np.convolve(ecg, kernel_j7, mode='same')
resp_j7 = np.convolve(resp, kernel_j7, mode='same')

# ===== skala 8 ===== #
j = 8
a = -(round(2**j) + round(2**(j-1)) - 2)
b = -(1 - round(2**(j-1))) + 1

q_j8 = [
    -1,     -3,     -6,    -10,    -15,    -21,    -28,    -36,
    -45,    -55,    -66,    -78,    -91,   -105,   -120,   -136,
   -153,   -171,   -190,   -210,   -231,   -253,   -276,   -300,
   -325,   -351,   -378,   -406,   -435,   -465,   -496,   -528,
   -561,   -595,   -630,   -666,   -703,   -741,   -780,   -820,
   -861,   -903,   -946,   -990,  -1035,  -1081,  -1128,  -1176,
  -1225,  -1275,  -1326,  -1378,  -1431,  -1485,  -1540,  -1596,
  -1653,  -1711,  -1770,  -1830,  -1891,  -1953,  -2016,  -2080,
  -2145,  -2211,  -2278,  -2346,  -2415,  -2485,  -2556,  -2628,
  -2701,  -2775,  -2850,  -2926,  -3003,  -3081,  -3160,  -3240,
  -3321,  -3403,  -3486,  -3570,  -3655,  -3741,  -3828,  -3916,
  -4005,  -4095,  -4186,  -4278,  -4371,  -4465,  -4560,  -4656,
  -4753,  -4851,  -4950,  -5050,  -5151,  -5253,  -5356,  -5460,
  -5565,  -5671,  -5778,  -5886,  -5995,  -6105,  -6216,  -6328,
  -6441,  -6555,  -6670,  -6786,  -6903,  -7021,  -7140,  -7260,
  -7381,  -7503,  -7626,  -7750,  -7875,  -8001,  -8128,  -8256,
  -8381,  -8503,  -8622,  -8738,  -8851,  -8961,  -9068,  -9172,
  -9273,  -9371,  -9466,  -9558,  -9647,  -9733,  -9816,  -9896,
  -9973, -10047, -10118, -10186, -10251, -10313, -10372, -10428,
 -10481, -10531, -10578, -10622, -10663, -10701, -10736, -10768,
 -10797, -10823, -10846, -10866, -10883, -10897, -10908, -10916,
 -10921, -10923, -10922, -10918, -10911, -10901, -10888, -10872,
 -10853, -10831, -10806, -10778, -10747, -10713, -10676, -10636,
 -10593, -10547, -10498, -10446, -10391, -10333, -10272, -10208,
 -10141, -10071,  -9998,  -9922,  -9843,  -9761,  -9676,  -9588,
  -9497,  -9403,  -9306,  -9206,  -9103,  -8997,  -8888,  -8776,
  -8661,  -8543,  -8422,  -8298,  -8171,  -8041,  -7908,  -7772,
  -7633,  -7491,  -7346,  -7198,  -7047,  -6893,  -6736,  -6576,
  -6413,  -6247,  -6078,  -5906,  -5731,  -5553,  -5372,  -5188,
  -5001,  -4811,  -4618,  -4422,  -4223,  -4021,  -3816,  -3608,
  -3397,  -3183,  -2966,  -2746,  -2523,  -2297,  -2068,  -1836,
  -1601,  -1363,  -1122,   -878,   -631,   -381,   -128,    128,
    381,    631,    878,   1122,   1363,   1601,   1836,   2068,
   2297,   2523,   2746,   2966,   3183,   3397,   3608,   3816,
   4021,   4223,   4422,   4618,   4811,   5001,   5188,   5372,
   5553,   5731,   5906,   6078,   6247,   6413,   6576,   6736,
   6893,   7047,   7198,   7346,   7491,   7633,   7772,   7908,
   8041,   8171,   8298,   8422,   8543,   8661,   8776,   8888,
   8997,   9103,   9206,   9306,   9403,   9497,   9588,   9676,
   9761,   9843,   9922,   9998,  10071,  10141,  10208,  10272,
  10333,  10391,  10446,  10498,  10547,  10593,  10636,  10676,
  10713,  10747,  10778,  10806,  10831,  10853,  10872,  10888,
  10901,  10911,  10918,  10922,  10923,  10921,  10916,  10908,
  10897,  10883,  10866,  10846,  10823,  10797,  10768,  10736,
  10701,  10663,  10622,  10578,  10531,  10481,  10428,  10372,
  10313,  10251,  10186,  10118,  10047,   9973,   9896,   9816,
   9733,   9647,   9558,   9466,   9371,   9273,   9172,   9068,
   8961,   8851,   8738,   8622,   8503,   8381,   8256,   8128,
   8001,   7875,   7750,   7626,   7503,   7381,   7260,   7140,
   7021,   6903,   6786,   6670,   6555,   6441,   6328,   6216,
   6105,   5995,   5886,   5778,   5671,   5565,   5460,   5356,
   5253,   5151,   5050,   4950,   4851,   4753,   4656,   4560,
   4465,   4371,   4278,   4186,   4095,   4005,   3916,   3828,
   3741,   3655,   3570,   3486,   3403,   3321,   3240,   3160,
   3081,   3003,   2926,   2850,   2775,   2701,   2628,   2556,
   2485,   2415,   2346,   2278,   2211,   2145,   2080,   2016,
   1953,   1891,   1830,   1770,   1711,   1653,   1596,   1540,
   1485,   1431,   1378,   1326,   1275,   1225,   1176,   1128,
   1081,   1035,    990,    946,    903,    861,    820,    780,
    741,    703,    666,    630,    595,    561,    528,    496,
    465,    435,    406,    378,    351,    325,    300,    276,
    253,    231,    210,    190,    171,    153,    136,    120,
    105,     91,     78,     66,     55,     45,     36,     28,
     21,     15,     10,      6,      3,      1
]

for k in range(a, b):
   k_list.append(k)
   qj[8][k + abs(a)] = q_j8[k + abs(a)] / 1048576

kernel_j8 = qj[8][0:len(k_list)]
ecg_j8 = np.convolve(ecg, kernel_j8, mode='same')
resp_j8 = np.convolve(resp, kernel_j8, mode='same')

st.subheader("Plot Hasil DWT Sesuai Skala")
skala = st.slider("Masukkan Nilai Skala DWT : ", min_value=1, max_value=8, step=1)

#============ plot nak streamlit ==============#
if skala == 1:
    dwt_ecgSignal = ecg_j1
    dwt_respSignal = resp_j1
elif skala == 2:
    dwt_ecgSignal = ecg_j2
    dwt_respSignal = resp_j2
elif skala == 3:
    dwt_ecgSignal = ecg_j3
    dwt_respSignal = resp_j3
elif skala == 4:
    dwt_ecgSignal = ecg_j4
    dwt_respSignal = resp_j4
elif skala == 5:
    dwt_ecgSignal = ecg_j5
    dwt_respSignal = resp_j5
elif skala == 6:
    dwt_ecgSignal = ecg_j6
    dwt_respSignal = resp_j6
elif skala == 7:
    dwt_ecgSignal = ecg_j7
    dwt_respSignal = resp_j7
elif skala == 8:
    dwt_ecgSignal = ecg_j8
    dwt_respSignal = resp_j8

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
axes[0].plot(t, dwt_ecgSignal, label = f'ECG qj[{skala}] (Wavelet j={skala})')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitudo (mV)')
axes[0].legend()

axes[1].plot(t, dwt_respSignal, label = f'RESP qj[{skala}] (Wavelet j={skala})')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Amplitudo (mV)')
axes[1].legend()

st.pyplot(fig)

# ===== Absolute ===== #
def absolute_signal(signal):
    return np.abs(signal)

ecg_abs = absolute_signal(ecg_j3)

st.subheader("Plot Hasil Absolute ECG menggunakana DWT Skala (j) = 3")
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(ecg_abs, color='darkgreen', label='Absoluted ECG Signal DWT j = 3')
ax.set_title('Absoluted ECG Signal DWT j = 3')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (mV)')
ax.legend()

st.pyplot(fig)

# ===== MAV ===== #
def zero_lag_moving_average(signal, window_size):
    assert window_size % 2 == 1, "Use an odd window size for zero-lag"
    half_window = window_size // 2
    # Use reflect padding to reduce edge bias
    padded_signal = np.pad(signal, (half_window, half_window), mode='reflect')
    
    filtered_signal = np.zeros_like(signal)
    for i in range(len(signal)):
        start = i
        end = i + window_size
        filtered_signal[i] = np.mean(padded_signal[start:end])
    return filtered_signal

st.subheader("Plot Hasil MAV")
wz = st.slider("Masukkan Nilai Window Size : ", min_value=1, max_value=30, step=1)
window_size = wz
mav_ecg = zero_lag_moving_average(ecg_abs, window_size)

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(mav_ecg, color='darkgreen', label='MAV ECG')
ax.set_title('ECG Signal After MAV')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (mV)')
ax.legend()

st.pyplot(fig)

# ===== Thresholding ===== #
# Function for thresholding
def threshold_signal(signal, threshold_value):
    thresholded_signal = np.zeros_like(signal)
    thresholded_signal[signal > threshold_value] = 1.1
    return thresholded_signal

# Apply thresholding
threshold_value = 0.31 # Adjust threshold value as needed
thresholded_ecg = threshold_signal(mav_ecg, threshold_value)

st.subheader("Plot Hasil Thresholding")
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(thresholded_ecg, label='Threshold')
ax.set_title('Thresholded ECG')

ax.plot(mav_ecg, label='ABS + MAV ECG j3')
ax.plot(ecg, label='Original Basaelined ECG')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (mV)')
ax.legend()

st.pyplot(fig)

# ===== RR Interval ===== #
def detect_rising_edges(signal):
    rising_edges = np.where(np.diff(signal) > 0)[0]
    return rising_edges

def detect_falling_edges(signal):
    falling_edges = np.where(np.diff(signal) < 0)[0]
    return falling_edges

# Detect edges
rising_edges = detect_rising_edges(thresholded_ecg)
falling_edges = detect_falling_edges(thresholded_ecg)

# Calculate heart rate (BPM)
fs = 125
time_intervals = np.diff(rising_edges) / fs
#time_intervals = np.diff(falling_edges) / fs
heart_rate = 60 / np.mean(time_intervals)

st.subheader("RR Interval - BPM")
st.markdown(f"**RR Interval**: {time_intervals:}")
st.markdown(f"**Mean RR Interval**: {np.mean(time_intervals):.4f} ")
st.markdown(f"**Heart Rate**: {heart_rate:.2f} BPM")

# ===== Plot HRV & semua ===== #
fs_HRV = 1/np.mean(time_intervals)
HR = 60/time_intervals
sequence_time = np.arange(len(HR)) / fs_HRV
time_hrv = np.cumsum(time_intervals)

st.subheader("Plot HRV - Resp. Signal - ECG DWT Skala (j) = 8")
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 10))
#axes[0].plot(time_hrv, HR, label='HRV', marker='o')
axes[0].plot(sequence_time, fs_HRV, label='HRV', color='red', marker='o')
axes[0].set_xlabel('Time (s)')
axes[0].set_xlim(0, 10)
axes[0].set_ylabel('HR (BPM)')
axes[0].legend()

axes[1].plot(t,resp, label='RESP', color='blue')
axes[1].plot(t,resp_j8, label='RESP j8', color='orange')
axes[1].set_xlabel('Time (s)')
axes[1].set_xlim(0, 10)
axes[1].set_ylabel('Amolitude (mV)')
axes[1].legend()

#axes[2].plot(t, ecg_j8, label='ECG j8', color='darkgreen')
axes[2].plot(t, (ecg_j8*20), label='ECG j8 (gain 20)', color='red')
axes[2].plot(t, resp, label='RESP')
#axes[2].plot(t, resp_j8, label='RESP j8')
axes[2].set_xlabel('Time (s)')
axes[2].set_xlim(0, 10)
axes[2].set_ylabel('Amolitude (mV)')
axes[2].legend()

st.pyplot(fig)