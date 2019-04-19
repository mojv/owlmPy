import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Fdtd1d:

    def __init__(self, ds_in, eps_in, mus_in, lambda_min_in, lambda_max_in, Ny_in=None, Nd_in=None, offset_in=None, nbc_in=None, epsrc_in=None, musrc_in=None, src_ini_in=None, NFREQ_in=None):

        if Ny_in is None:
            Ny_in=40

        if Nd_in is None:
            Nd = 4

        self.offset = offset_in
        if offset_in is None:
            self.offset = 103

        if nbc_in is None:
            nbc_in = 1

        if epsrc_in is None:
            epsrc_in = 1

        if musrc_in is None:
            musrc_in = 1

        if src_ini_in is None:
            src_ini_in = 2

        if NFREQ_in is None:
            NFREQ_in = 1000

        # ======Calculation of steps in the spatial yee grid========
        self.ds = ds_in * (10 ** (-7))
        self.c0 = 299792458 * 100  # speed of light cm/s
        fmax = self.c0 / (lambda_min_in * (10 ** (-7)))  # Max wave frequency in Hz
        fmin = self.c0 / (lambda_max_in * (10 ** (-7)))  # Max wave frequency in Hz
        self.index = np.sqrt(eps_in * mus_in)  # Refraction index vector
        lambda_min = self.c0 / (fmax * float(np.amax(self.index)))  # minimum wavelength
        delta_lambda = (lambda_min / Ny_in)  # Distance of the cell taking into account the wavelength in cm
        delta_ds = np.amin(self.ds) / Nd_in  # Distance of the cell taking into account the layer
        delta_z = np.amin([delta_lambda, delta_ds])  # Distance of the cell
        d = np.sum(self.ds)  # total distance of the space to model
        self.N = int(np.ceil(d/delta_z))  # Total steps in grid
        self.dz = d / self.N
        self.N += self.offset
        self.z = np.arange(0, self.N, 1) * self.dz  # z axis
        self.src_ini = zsrc_ini_in
    
        # ========Compute Position of Materials on Grid ========
        eps_d = np.ones(self.N)
        mus_d = np.ones(self.N)
        init = int(np.ceil(self.offset/2))
        for i in range(np.size(self.index)):
            temp = int(init+np.round(self.ds[i]/self.dz, 0)+1)
            eps_d[init:temp] = eps_in[i]
            mus_d[init:temp] = mus_in[i]
            init = temp

        # ======Calculation of steps in the temporal yee grid ========
        dt = (nbc_in*self.dz)/(2*self.c0)  # time of each step
        tprop = np.amax(self.index)*self.N*self.dz/self.c0 # total time it takes for a wave to propagate across the grid one time
        tao = 1/(2*fmax)
        self.T = 12*tao + 3*tprop  # Total Time of simulation
        self.S = int(np.ceil(self.T/dt))  # Total Number of Iterations
        self.t = np.arange(0, self.S-1, 1)*dt  # time axis

        # ======Compute the Source Functions for Ey/Hx Mode ========

        delt = nbc_in*self.dz/(2*self.c0) + dt/2  # total delay between E and H because Yee Grid
        a = - np.sqrt(epsrc_in/musrc_in)  # amplitude of H field
        self.Esrc = np.exp(-((self.t-6*tao)/tao)**2)  # E field source
        self.Hsrc = a*np.exp(-((self.t-6*tao+delt)/tao)**2)  # H field source

        # ====== Initialize the Fourier Transforms ========
        self.FREQ = np.linspace(fmin, fmax, NFREQ_in)
        self.K = np.exp(-1j*2*np.pi*dt*self.FREQ)
        self.REF = np.zeros(NFREQ_in)+0j
        self.TRN = np.zeros(NFREQ_in)+0j
        self.SRC = np.zeros(NFREQ_in)+0j
        self.NREF = np.zeros(NFREQ_in)
        self.NTRN = np.zeros(NFREQ_in)
        self.CON = np.zeros(NFREQ_in)

        # ====== Initialize update coefficients and compensate for numerical dispersion ====
        self.mE = self.c0*dt/eps_d
        self.mH = self.c0*dt/mus_d

        # ====== Declare Variables for Animation ====
        self.Eys = []
        self.Hxs = []
        self.frames = 0

    def run_sim(self, fr=None):

        # ======E and H ========
        Ey = np.zeros(self.N)
        Hx = np.zeros(self.N)
        if fr is not None:
            self.frames = int(np.ceil(self.S / fr))
            self.Eys = np.zeros((int(np.ceil((self.S - 2) / self.frames)), self.N))
            self.Hxs = np.zeros((int(np.ceil((self.S - 2) / self.frames)), self.N))
        else:
            self.frames = self.S

        # ======Initialize PML values ========
        h1 = 0
        h2 = 0
        e1 = 0
        e2 = 0

        for i in range(self.S - 2):

            # Update PML values for H
            h2 = h1
            h1 = Hx[0]

            # Apply PML values for H
            Hx[-1] = Hx[-1] + self.mH[self.N - 1] * (e2 - Ey[-1]) / self.dz

            # Update H from E (Dirichlet Boundary Conditions)
            Hx[0:self.N - 1] = Hx[0:self.N - 1] + self.mH[0:self.N - 1] * (Ey[1:self.N] - Ey[0:self.N - 1]) / self.dz

            # Correction for Hx adjacent to TFSF boundary
            Hx[self.src_ini - 1] -= self.mH[self.src_ini - 1] * self.Esrc[i] / self.dz

            # Update PML values for E
            e2 = e1
            e1 = Ey[-1]

            # Apply PML values for E
            Ey[0] = Ey[0] + self.mE[0] * (Hx[0] - h2) / self.dz

            # Update E from H
            Ey[1:self.N] = Ey[1:self.N] + (self.mE[1:self.N] * (Hx[1:self.N] - Hx[0:self.N - 1])) / self.dz

            # Correction for Ey adjacent to TFSF boundary
            Ey[self.src_ini] -= self.mE[self.src_ini] * self.Hsrc[i] / self.dz

            # Computing discrete fourier transform
            self.REF += (self.K ** i) * Ey[1]
            self.TRN += (self.K ** i) * Ey[-1]
            self.SRC += (self.K ** i) * self.Esrc[i]

            if i % self.frames == 0:
                if fr is not None:
                    self.Hxs[int(np.ceil(i / self.frames))] = Hx
                    self.Eys[int(np.ceil(i / self.frames))] = Ey

        self.NREF = np.absolute(self.REF / self.SRC) ** 2
        self.NTRN = np.absolute(self.TRN / self.SRC) ** 2
        self.CON = self.NREF + self.NTRN

    def create_ani(self, filename=None):

        if filename is None:
            filename = "video.mp4"

        fig = plt.figure()
        plts = []

        for i in range(int(np.floor((self.S - 2) / self.frames))):
            p, = plt.plot(self.z, self.Hxs[i], color="blue")
            p2, = plt.plot(self.z, self.Eys[i], '-r')
            plts.append([p, p2])

        # Add Layers Background
        plt.axis([0, (self.N - 1) * self.dz, -2, 2])
        init = int(np.ceil(self.offset / 2))
        color = 0.2

        for j in range(np.size(self.index)):
            temp = int(init + np.round(self.ds[j] / self.dz, 0) + 1)
            plt.axvspan(init * self.dz, temp * self.dz, facecolor=str(color), alpha=0.5)
            init = temp
            color += 0.1

        ani = animation.ArtistAnimation(fig, plts, interval=50, repeat_delay=3000)
        plt.show()
