import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Fdtd2d:

    def __init__(self, x_in, y_in, lambda_min_in, lambda_max_in, n_lambd, src_ini_in, epsrc_in, musrc_in, NFREQ_in):

        self.c0 = 299792458 * 100  # speed of light cm/s
        # ======Calculation of steps in the spatial yee grid========
        self.x = x_in * (10 ** (-7))
        self.y = y_in * (10 ** (-7))

        fmax = self.c0 / (lambda_min_in * (10 ** (-7)))  # Max wave frequency in Hz
        fmin = self.c0 / (lambda_max_in * (10 ** (-7)))  # Max wave frequency in Hz

        lambda_min = self.c0 / (fmax * float(np.amax(1)))  # minimum wavelength
        self.delta = (lambda_min / n_lambd)  # Distance of the cell taking into account the wavelength in cm
        self.Nx = int(np.ceil(self.x / self.delta))  # Total steps in grid x
        self.Ny = int(np.ceil(self.y / self.delta))  # Total steps in grid y

        self.src_ini = [int(src_ini_in[0] * self.Nx), int(src_ini_in[1]*self.Ny)]

        # ======Calculation of steps in the temporal yee grid ========
        self.dt = 1 / (self.c0 * np.sqrt((1/np.square(self.delta))+(1/np.square(self.delta))))  # time of each step
        tprop = self.Nx * self.delta / self.c0  # total time it takes for a wave to propagate across the grid one time
        tao = 1 / (2 * fmax)
        self.T = 12 * tao + 3 * tprop  # Total Time of simulation
        self.S = int(np.ceil(self.T / self.dt))  # Total Number of Iterations
        self.t = np.arange(0, self.S - 1, 1) * self.dt  # time axis

        # ======Compute the Source Functions for Ey/Hx Mode ========

        delt = self.delta / (2 * self.c0) + self.dt / 2  # total delay between E and H because Yee Grid
        a = - np.sqrt(epsrc_in / musrc_in)  # amplitude of H field
        self.Esrc_x = np.exp(-((self.t - 6 * tao) / tao) ** 2)  # E field source
        self.Esrc_y = np.exp(-((self.t - 6 * tao) / tao) ** 2)  # E field source
        self.Esrc_z = np.exp(-((self.t - 6 * tao) / tao) ** 2)  # E field source
        self.Hsrc_x = a * np.exp(-((self.t - 6 * tao + delt) / tao) ** 2)  # H field source
        self.Hsrc_y = a * np.exp(-((self.t - 6 * tao + delt) / tao) ** 2)  # H field source
        self.Hsrc_z = a * np.exp(-((self.t - 6 * tao + delt) / tao) ** 2)  # H field source

        # ====== Initialize the Fourier Transforms ========
        self.FREQ = np.linspace(fmin, fmax, NFREQ_in)
        self.K = np.exp(-1j * 2 * np.pi * self.dt * self.FREQ)
        self.REF = np.zeros(NFREQ_in) + 0j
        self.TRN = np.zeros(NFREQ_in) + 0j
        self.SRC = np.zeros(NFREQ_in) + 0j
        self.NREF = np.zeros(NFREQ_in)
        self.NTRN = np.zeros(NFREQ_in)
        self.CON = np.zeros(NFREQ_in)

        # ====== Initialize update coefficients and compensate for numerical dispersion ====
        self.mExx = 1/(np.ones((self.Nx, self.Ny))*self.delta)
        self.mEyy = 1/(np.ones((self.Nx, self.Ny))*self.delta)
        self.mEzz = 1/(np.ones((self.Nx, self.Ny))*self.delta)
        self.mHxx = (self.c0*self.dt)/(np.ones((self.Nx, self.Ny))*self.delta)
        self.mHyy = (self.c0*self.dt)/(np.ones((self.Nx, self.Ny))*self.delta)
        self.mHzz = (self.c0*self.dt)/(np.ones((self.Nx, self.Ny))*self.delta)

        # ====== Declare Variables for Animation ====
        self.Etot = []
        self.Htot = []
        self.frames = 0

    def run_sim(self, fr=None):

        # ======E and H ========
        Hx = np.zeros((self.Nx, self.Ny))
        Hy = np.zeros((self.Nx, self.Ny))
        Hz = np.zeros((self.Nx, self.Ny))
        Dx = np.zeros((self.Nx, self.Ny))
        Dy = np.zeros((self.Nx, self.Ny))
        Dz = np.zeros((self.Nx, self.Ny))
        Ex = np.zeros((self.Nx, self.Ny))
        Ey = np.zeros((self.Nx, self.Ny))
        Ez = np.zeros((self.Nx, self.Ny))

        if fr is not None:
            self.frames = int(np.ceil(self.S / fr))
            self.Etot = np.zeros((int(np.ceil((self.S - 2) / self.frames)), self.Nx, self.Ny))
            self.Htot = np.zeros((int(np.ceil((self.S - 2) / self.frames)), self.Nx, self.Ny))
        else:
            self.frames = self.S

        for i in range(self.S - 2):

            Hx[:, 0:-1] = Hx[:, 0:-1] - self.mHxx[:, 0:-1] * (Ez[:, 1:] - Ez[:, 0:-1])
            Hy[0:-1, :] = Hy[0:-1, :] + self.mHyy[0:-1, :] * (Ez[1:, :] - Ez[0:-1, :])
            Dz[1:, 1:] = Dz[1:, 1:] + (self.c0*self.dt) * (Hy[1:, 1:] - Hy[0:-1, 1:] - Hx[1:, 1:] + Hx[1:, 0:-1])
            Ez = self.mEzz * Dz

            Hz[0:-1, 0:-1] = Hz[0:-1, 0:-1] - self.mHzz[0:-1, 0:-1] * (Ey[1:, 0:-1] - Ey[0:-1, 0:-1] - Ex[0:-1, 1:] + Ex[0:-1, 0:-1])
            Dx[:, 1:] = Dx[:, 1:] + (self.c0 * self.dt) * (Hz[:, 1:] - Hz[:, 0:-1])
            Dy[1:, :] = Dy[1:, :] - (self.c0 * self.dt) * (Hz[1:, :] - Hz[0:-1, :])
            Ey = self.mEyy * Dy
            Ex = self.mExx * Dz

            Ez[self.src_ini[0], self.src_ini[1]] += self.Esrc_z[i]
            Ey[self.src_ini[0], self.src_ini[1]] += self.Esrc_y[i]
            Ex[self.src_ini[0], self.src_ini[1]] += self.Esrc_x[i]
            Hz[self.src_ini[0], self.src_ini[1]] += self.Hsrc_z[i]
            Hy[self.src_ini[0], self.src_ini[1]] += self.Hsrc_y[i]
            Hx[self.src_ini[0], self.src_ini[1]] += self.Hsrc_x[i]

            if i % self.frames == 0:
                if fr is not None:
                    self.Htot[int(np.   ceil(i / self.frames)), :, :] = Hx
                    self.Etot[int(np.ceil(i / self.frames)), :, :] = Hz

    def create_ani(self, filename=None):

        fig = plt.figure()
        plts = []

        for i in range(int(np.floor((self.S - 2) / self.frames))):
            print(i)
            im = plt.imshow(self.Etot[i, :, :], vmin=-0.05, vmax=0.05, animated=True)
            plts.append([im])

        ani = animation.ArtistAnimation(fig, plts, interval=50, repeat_delay=3000)
        plt.show()
