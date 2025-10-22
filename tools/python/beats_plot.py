import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma = 1.5
T_eff = 3 * sigma # Rough effective window width
t_max = 7
t = np.linspace(0, t_max, 800) # Increased points for smoothness

# Fast oscillation frequency
omega_bar = 10 * np.pi

# Decaying function
f_t = np.exp(-t**2 / (2 * sigma**2))

# --- Case 1: T_beat << T_eff ---
T_beat1 = 2.0
delta_omega1 = 4 * np.pi / T_beat1
omega1_1 = omega_bar - delta_omega1 / 2
omega2_1 = omega_bar + delta_omega1 / 2
g_osc1 = 0.5 * (np.cos(omega1_1 * t) + np.cos(omega2_1 * t))
g1 = f_t * g_osc1
beat_env1 = f_t * np.cos(delta_omega1 / 2 * t) # Beat envelope itself

# --- Case 2: T_beat >> T_eff ---
T_beat2 = 40.0
delta_omega2 = 4 * np.pi / T_beat2
omega1_2 = omega_bar - delta_omega2 / 2
omega2_2 = omega_bar + delta_omega2 / 2
g_osc2 = 0.5 * (np.cos(omega1_2 * t) + np.cos(omega2_2 * t))
g2 = f_t * g_osc2
beat_env2 = f_t * np.cos(delta_omega2 / 2 * t)

# --- Case 3: T_beat approx T_eff ---
T_beat3 = 5.0
delta_omega3 = 4 * np.pi / T_beat3
omega1_3 = omega_bar - delta_omega3 / 2
omega2_3 = omega_bar + delta_omega3 / 2
g_osc3 = 0.5 * (np.cos(omega1_3 * t) + np.cos(omega2_3 * t))
g3 = f_t * g_osc3
beat_env3 = f_t * np.cos(delta_omega3 / 2 * t)

# --- Plotting ---
plt.figure(figsize=(12, 9))

# Case 1 Plot
plt.subplot(3, 1, 1)
plt.plot(t, g1, label=f'$g(t)$, $T_{{beat}}={T_beat1:.1f} \ll T_{{eff}}$')
plt.plot(t, f_t, 'k--', label='$f(t)$ (envelope)', alpha=0.7)
plt.plot(t, -f_t, 'k--', alpha=0.7)
# Optional: Plot the beat envelope to make it clearer
plt.plot(t, beat_env1, 'r:', label=r'$f(t) \cos(\frac{\Delta\omega}{2}t)$', alpha=0.6)
plt.plot(t, -beat_env1, 'r:', alpha=0.6)
plt.title(f'Case 1: Fast Beat ($T_{{beat}} = {T_beat1:.1f} \ll T_{{eff}} \\approx {T_eff:.1f}$)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.5)
plt.legend(fontsize='small')
plt.ylim(-1.1, 1.1)

# Case 2 Plot
plt.subplot(3, 1, 2)
plt.plot(t, g2, label=f'$g(t)$, $T_{{beat}}={T_beat2:.1f} \gg T_{{eff}}$')
plt.plot(t, f_t, 'k--', label='$f(t)$ (envelope)', alpha=0.7)
plt.plot(t, -f_t, 'k--', alpha=0.7)
plt.plot(t, beat_env2, 'r:', label=r'$f(t) \cos(\frac{\Delta\omega}{2}t)$', alpha=0.6)
plt.plot(t, -beat_env2, 'r:', alpha=0.6)
plt.title(f'Case 2: Slow Beat ($T_{{beat}} = {T_beat2:.1f} \gg T_{{eff}} \\approx {T_eff:.1f}$)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.5)
plt.legend(fontsize='small')
plt.ylim(-1.1, 1.1)

# Case 3 Plot
plt.subplot(3, 1, 3)
plt.plot(t, g3, label=f'$g(t)$, $T_{{beat}}={T_beat3:.1f} \\approx T_{{eff}}$')
plt.plot(t, f_t, 'k--', label='$f(t)$ (envelope)', alpha=0.7)
plt.plot(t, -f_t, 'k--', alpha=0.7)
plt.plot(t, beat_env3, 'r:', label=r'$f(t) \cos(\frac{\Delta\omega}{2}t)$', alpha=0.6)
plt.plot(t, -beat_env3, 'r:', alpha=0.6)
plt.title(f'Case 3: Comparable Beat ($T_{{beat}} = {T_beat3:.1f} \\approx T_{{eff}} \\approx {T_eff:.1f}$)')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.5)
plt.legend(fontsize='small')
plt.ylim(-1.1, 1.1)


plt.tight_layout()
plt.show()