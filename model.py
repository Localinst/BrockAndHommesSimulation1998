import numpy as np
import matplotlib.pyplot as plt
import os

class BrockHommesAsset1998:
    def __init__(self, R=1.1, D=1.0, C=1.0, g=1.2, eta=0.0):
        self.R = R
        self.D = D
        self.C = C
        self.g = g
        self.eta = eta
        self.x_lag = None
        self.m_lag = None
        self.U_f_lag = 0.0
        self.U_t_lag = 0.0

    def next_state(self, x_current, m_current, beta, noise_range=0.05):
        if self.x_lag is None:
            self.x_lag = x_current
            self.m_lag = m_current
            self.U_f_lag = 0.0
            self.U_t_lag = 0.0
            return x_current, m_current
        
        n1 = (1 + self.m_lag) / 2
        n2 = (1 - self.m_lag) / 2
        
        x_t = (n2 * self.g * self.x_lag) / self.R
        noise = np.random.uniform(-noise_range, noise_range)
        excess = x_t - self.R * self.x_lag + noise

        pi_fund  = (excess * (-self.R * self.x_lag)) / self.D
        pi_trend = (excess * (self.g * self.x_lag - self.R * self.x_lag)) / self.D
        
        U_f_new = pi_fund - self.C + self.eta * self.U_f_lag
        U_t_new = pi_trend + self.eta * self.U_t_lag

        diff = U_f_new - U_t_new
        m_new = np.tanh((beta / 2.0) * diff)
        
        self.x_lag = x_t
        self.U_f_lag = U_f_new
        self.U_t_lag = U_t_new
        self.m_lag = m_new

        return x_t, m_new

    def simulate(self, beta, n=40000, burn=10000, x0=0.01, m0=0.0, noise_range=0.0):
        self.x_lag = None
        self.m_lag = None
        self.U_f_lag = 0.0
        self.U_t_lag = 0.0
        traj = np.zeros((int(n), 2))
        x, m = x0, m0

        for t in range(int(n)):
            traj[t] = [x, m]
            x, m = self.next_state(x, m, beta, noise_range)
        
        return traj[burn:]

    def plot_attractor(self, beta, output_dir='output_fixed', noise_range=0.0):
        os.makedirs(output_dir, exist_ok=True)
        traj = self.simulate(beta, noise_range=noise_range)
        
        plt.figure(figsize=(11, 10))
        plt.scatter(traj[:,0], traj[:,1], s=2, alpha=0.6, color='darkblue', rasterized=True)
        plt.title(f'Attractor β={beta} | g={self.g} | C={self.C}', fontsize=14, fontweight='bold')
        plt.xlabel(r'$x_t$ (deviazione dal fondamentale)')
        plt.ylabel(r'$m_t$ (n_fund - n_trend)')
        plt.xlim(-3.5, 3.5)
        plt.ylim(-1.1, 1.1)
        plt.grid(True, alpha=0.3)
        plt.axhline(0, color='k', lw=0.5, alpha=0.4)
        plt.axvline(0, color='k', lw=0.5, alpha=0.4)
        
        plt.savefig(f"{output_dir}/attractor_beta_{beta:.1f}_noise{noise_range:.2f}.png", dpi=200, bbox_inches='tight')
        plt.close()

    def plot_timeseries(self, beta, output_dir='output_fixed', n=500, noise_range=0.0):
        os.makedirs(output_dir, exist_ok=True)
        traj = self.simulate(beta,  noise_range=noise_range)
        
        fig, axs = plt.subplots(2, 1, figsize=(13, 8))
        axs[0].plot(traj[:n, 0], linewidth=1.1)
        axs[0].set_title(f'Prezzi (x_t) - β = {beta}')
        axs[0].grid(True, alpha=0.3)
        axs[0].set_xlim(100, 600)
        axs[0].set_ylim(-3.5, 3.5)
        axs[1].plot(traj[:n, 1], linewidth=1.1, color='red')
        axs[1].set_ylim(-1.1, 1.1)
        axs[1].set_xlim(100, 600)
        axs[1].set_title(f'm_t (differenza frazioni) - β = {beta}')
        axs[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/timeseries_beta_{beta:.1f}_noise{noise_range:.2f}.png", dpi=180, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    output_dir = 'artifacts'
    os.makedirs(output_dir, exist_ok=True)

    sim = BrockHommesAsset1998(R=1.1, D=1.0, C=1.0, g=1.2, eta=0.0)

    betas = [3.5, 3.6, 4.0]

    for beta in betas:
        print(f"\n→ Beta = {beta}")
        sim.plot_attractor(beta, output_dir=output_dir, noise_range=0.0)
        sim.plot_attractor(beta, output_dir=output_dir, noise_range=0.05)
        
        sim.plot_timeseries(beta, output_dir=output_dir, n=600, noise_range=0.0)
        sim.plot_timeseries(beta, output_dir=output_dir, n=600, noise_range=0.05)