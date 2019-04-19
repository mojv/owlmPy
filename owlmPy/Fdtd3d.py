import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import *
from tqdm import tqdm

class Fdtd3d:

    def __init__(self, x_in, y_in,  z_in, pml_in, lambda_min_in, lambda_max_in, n_lambd, src_ini_in, epsrc_in, musrc_in, NFREQ_in):

        self.c0 = 299792458 * 100  # speed of light cm/s
        self.eps_o = 8.85e-14  # permittivity of vacuum in F/cm

        # ======Calculation of steps in the spatial yee grid========
        self.x = x_in * (10 ** (-7))  # with of the domain in cm
        self.y = y_in * (10 ** (-7))  # large of the domain in cm
        self.z = z_in * (10 ** (-7))  # height of the domain in cm
        self.pml = pml_in * (10 ** (-7))  # size of the Perfectly matched layer (PML) in cm
        fmax = self.c0 / (lambda_min_in * (10 ** (-7)))  # Max wave frequency in Hz
        fmin = self.c0 / (lambda_max_in * (10 ** (-7)))  # Max wave frequency in Hz
        lambda_min = self.c0 / (fmax * float(np.amax(1)))  # minimum wavelength
        self.delta = (lambda_min / n_lambd)  # zise of the Yee cell taking into account the wavelength in cm
        self.Npml = int(np.ceil(self.pml / self.delta))  # Total steps in PML
        self.Nx = int(np.ceil(self.x / self.delta)) + 2 * self.Npml  # Total steps in grid x
        self.Ny = int(np.ceil(self.y / self.delta)) + 2 * self.Npml  # Total steps in grid y
        self.Nz = int(np.ceil(self.z / self.delta)) + 2 * self.Npml  # Total steps in grid z
        self.src_ini = [int(src_ini_in[2]*self.Nz), int(src_ini_in[0] * self.Nx), int(src_ini_in[1]*self.Ny)]  # position of the source

        # ======Calculation of steps in the temporal yee grid ========
        self.dt = 1 / (self.c0 * np.sqrt((1/np.square(self.delta))*3))  # time of each step
        tprop = self.Nx * self.delta / self.c0  # total time it takes for a wave to propagate across the grid one time
        tao = 1 / (2 * fmax)
        self.T = 12 * tao + 3 * tprop  # Total Time of simulation
        self.S = int(np.ceil(self.T / self.dt))  # Total Number of Iterations
        self.t = np.arange(0, self.S - 1, 1) * self.dt  # time axis

        # ====== Initialize Perfectly mached layer conductivity coefficients ====
        self.sx = np.zeros((self.Nz, self.Nx, self.Ny))
        self.sy = np.zeros((self.Nz, self.Nx, self.Ny))
        self.sz = np.zeros((self.Nz, self.Nx, self.Ny))
        # setting the coefficient equal to 0 at the beginning of the PML and 1 at the end
        for i in range(self.Npml):
            self.sx[:, i, :] = (self.Npml - i) / self.Npml
            self.sx[:, -i-1, :] = (self.Npml - i) / self.Npml
            self.sy[:, :, i] = (self.Npml - i) / self.Npml
            self.sy[:, :, -i-1] = (self.Npml - i) / self.Npml
            self.sz[i, :, :] = (self.Npml - i) / self.Npml
            self.sz[-i-1, :, :] = (self.Npml - i) / self.Npml
        # Converting to conductivity
        self.sx = (self.eps_o / (2 * self.dt)) * (self.sx ** 3)
        self.sy = (self.eps_o / (2 * self.dt)) * (self.sx ** 3)
        self.sz = (self.eps_o / (2 * self.dt)) * (self.sx ** 3)
        #self.sx = self.sx * 0
        #self.sy = self.sy * 0
        #self.sz = self.sz * 0

        # ======Compute the Source Functions for Ey/Hx Mode ========
        delt = self.delta / (2 * self.c0) + self.dt / 2  # total delay between E and H because Yee Grid
        a = - np.sqrt(epsrc_in / musrc_in)  # amplitude of H field
        self.Esrc_x = np.exp(-((self.t - 6 * tao) / tao) ** 2) * 10 # E field source
        self.Esrc_y = np.exp(-((self.t - 6 * tao) / tao) ** 2) * 10 # E field source
        self.Esrc_z = np.exp(-((self.t - 6 * tao) / tao) ** 2) * 10 # E field source
        self.Hsrc_x = a * np.exp(-((self.t - 6 * tao + delt) / tao) ** 2) * 10  # H field source
        self.Hsrc_y = a * np.exp(-((self.t - 6 * tao + delt) / tao) ** 2) * 10  # H field source
        self.Hsrc_z = a * np.exp(-((self.t - 6 * tao + delt) / tao) ** 2) * 10  # H field source

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
        self.mExx = 1/(np.ones((self.Nz, self.Nx, self.Ny))*self.delta)
        self.mEyy = 1/(np.ones((self.Nz, self.Nx, self.Ny))*self.delta)
        self.mEzz = 1/(np.ones((self.Nz, self.Nx, self.Ny))*self.delta)
        self.mHxx = -(self.c0*self.dt)/(np.ones((self.Nz, self.Nx, self.Ny))*self.delta)
        self.mHyy = -(self.c0*self.dt)/(np.ones((self.Nz, self.Nx, self.Ny))*self.delta)
        self.mHzz = -(self.c0*self.dt)/(np.ones((self.Nz, self.Nx, self.Ny))*self.delta)
        self.mu_xx = np.ones((self.Nz, self.Nx, self.Ny))
        self.mu_yy = np.ones((self.Nz, self.Nx, self.Ny))
        self.mu_zz = np.ones((self.Nz, self.Nx, self.Ny))
        self.eps_xx = np.ones((self.Nz, self.Nx, self.Ny))
        self.eps_yy = np.ones((self.Nz, self.Nx, self.Ny))
        self.eps_zz = np.ones((self.Nz, self.Nx, self.Ny))


        # ====== Declare Variables for Animation ====
        self.Etot = []
        self.Htot = []
        self.frames = 0

        #====== Set Fields Variables =============
        self.Hx = np.zeros((self.Nz, self.Nx, self.Ny))
        self.Hy = np.zeros((self.Nz, self.Nx, self.Ny))
        self.Hz = np.zeros((self.Nz, self.Nx, self.Ny))
        self.Dx = np.zeros((self.Nz, self.Nx, self.Ny))
        self.Dy = np.zeros((self.Nz, self.Nx, self.Ny))
        self.Dz = np.zeros((self.Nz, self.Nx, self.Ny))
        self.Ex = np.zeros((self.Nz, self.Nx, self.Ny))
        self.Ey = np.zeros((self.Nz, self.Nx, self.Ny))
        self.Ez = np.zeros((self.Nz, self.Nx, self.Ny))

    def run_sim(self, fr=None):

        # ======E and H =======
        if fr is not None:
            self.frames = int(np.ceil(self.S / fr))
            self.Etot = np.zeros((int(np.ceil((self.S - 2) / self.frames)), self.Nz, self.Nx, self.Ny))
            self.Htot = np.zeros((int(np.ceil((self.S - 2) / self.frames)), self.Nz, self.Nx, self.Ny))
        else:
            self.frames = self.S

        for i in tqdm(range(self.S - 2)):

            self.Hx[0:-1, :, 0:-1] = self.Hx[0:-1, :, 0:-1] + self.mHxx[0:-1, :, 0:-1] * (self.Ez[0:-1, :, 1:] - self.Ez[0:-1, :, 0:-1] - self.Ey[1:, :, 0:-1] + self.Ey[0:-1, :, 0:-1])
            self.Hy[0:-1, 0:-1, :] = self.Hy[0:-1, 0:-1, :] + self.mHyy[0:-1, 0:-1, :] * (self.Ex[1:, 0:-1, :] - self.Ex[0:-1, 0:-1, :] - self.Ez[0:-1, 1:, :] + self.Ez[0:-1, 0:-1, :])
            self.Dz[:, 1:, 1:] = self.Dz[:, 1:, 1:] + (self.c0 * self.dt) * (self.Hy[:, 1:, 1:] - self.Hy[:, 0:-1, 1:] - self.Hx[:, 1:, 1:] + self.Hx[:, 1:, 0:-1])

            #self.Ez[:, 1:, 1:] = self.Ez[:, 1:, 1:] + (self.c0 * self.dt) * self.mEzz[:, 1:, 1:] * (self.Hy[:, 1:, 1:] - self.Hy[:, 0:-1, 1:] - self.Hx[:, 1:, 1:] + self.Hx[:, 1:, 0:-1])
            self.Ez = self.mEzz * self.Dz

            self.Hz[:, 0:-1, 0:-1] = self.Hz[:, 0:-1, 0:-1] + self.mHzz[:, 0:-1, 0:-1] * (self.Ey[:, 1:, 0:-1] - self.Ey[:, 0:-1, 0:-1] - self.Ex[:, 0:-1, 1:] + self.Ex[:, 0:-1, 0:-1])
            self.Dx[1:, :, 1:] = self.Dx[1:, :, 1:] + (self.c0 * self.dt) * (self.Hz[1:, :, 1:] - self.Hz[1:, :, 0:-1] - self.Hy[1:, :, 1:] + self.Hy[0:-1, :, 1:])
            self.Dy[1:, 1:, :] = self.Dy[1:, 1:, :] + (self.c0 * self.dt) * (self.Hx[1:, 1:, :] - self.Hx[0:-1, 1:, :] - self.Hz[1:, 1:, :] + self.Hz[1:, 0:-1, :])

            #self.Ex[1:, :, 1:] = self.Ex[1:, :, 1:] + (self.c0 * self.dt) * self.mExx[1:, :, 1:] * (self.Hz[1:, :, 1:] - self.Hz[1:, :, 0:-1] - self.Hy[1:, :, 1:] + self.Hy[0:-1, :, 1:])
            #self.Ey[1:, 1:, :] = self.Ey[1:, 1:, :] + (self.c0 * self.dt) * self.mEyy[1:, 1:, :] * (self.Hx[1:, 1:, :] - self.Hx[0:-1, 1:, :] - self.Hz[1:, 1:, :] + self.Hz[1:, 0:-1, :])
            self.Ex = self.mExx * self.Dx
            self.Ey = self.mEyy * self.Dy

            self.Ez[self.src_ini[0], self.src_ini[1], self.src_ini[2]] += self.Esrc_z[i]
            self.Ey[self.src_ini[0], self.src_ini[1], self.src_ini[2]] += self.Esrc_y[i]
            self.Ex[self.src_ini[0], self.src_ini[1], self.src_ini[2]] += self.Esrc_x[i]
            self.Hz[self.src_ini[0], self.src_ini[1], self.src_ini[2]] += self.Hsrc_z[i]
            self.Hy[self.src_ini[0], self.src_ini[1], self.src_ini[2]] += self.Hsrc_y[i]
            self.Hx[self.src_ini[0], self.src_ini[1], self.src_ini[2]] += self.Hsrc_x[i]


            if i % self.frames == 0:
                if fr is not None:
                    self.Htot[int(np.   ceil(i / self.frames)), :, :, :] = self.Hx
                    self.Etot[int(np.ceil(i / self.frames)), :, :, :] = self.Ez

    def run_sim_pml(self, fr=None):

        # ===============Setting PML parameters==========================================
        mHx0 = (1/self.dt) + ((self.sy+self.sz)/(2*self.eps_o)) + (self.sy*self.sz*self.dt/(4*(self.eps_o**2)))
        mHx1 = (1/mHx0)*((1/self.dt) - ((self.sy+self.sz)/(2*self.eps_o)) - (self.sy*self.sz*self.dt/(4*(self.eps_o**2))))
        mHx2 = -(self.c0/(mHx0*self.mu_xx))
        mHx3 = -(self.c0*self.dt*self.sx/(mHx0*self.eps_o*self.mu_xx))
        mHx4 = -(self.dt*self.sy*self.sz/(mHx0*(self.eps_o**2)))
        ICEx = np.zeros((self.Nz, self.Nx, self.Ny))
        IHx = np.zeros((self.Nz, self.Nx, self.Ny))
        CEx = np.zeros((self.Nz, self.Nx, self.Ny))

        mHy0 = (1/self.dt) + ((self.sx+self.sz)/(2*self.eps_o)) + (self.sx*self.sz*self.dt/(4*(self.eps_o**2)))
        mHy1 = (1/mHy0)*((1/self.dt) - ((self.sx+self.sz)/(2*self.eps_o)) - (self.sx*self.sz*self.dt/(4*(self.eps_o**2))))
        mHy2 = -(self.c0/(mHy0*self.mu_yy))
        mHy3 = -(self.c0*self.dt*self.sy/(mHy0*self.eps_o*self.mu_yy))
        mHy4 = -(self.dt*self.sx*self.sz/(mHy0*(self.eps_o**2)))
        ICEy = np.zeros((self.Nz, self.Nx, self.Ny))
        IHy = np.zeros((self.Nz, self.Nx, self.Ny))
        CEy = np.zeros((self.Nz, self.Nx, self.Ny))

        mHz0 = (1/self.dt) + ((self.sx+self.sy)/(2*self.eps_o)) + (self.sx*self.sy*self.dt/(4*(self.eps_o**2)))
        mHz1 = (1/mHz0)*((1/self.dt) - ((self.sx+self.sy)/(2*self.eps_o)) - (self.sx*self.sy*self.dt/(4*(self.eps_o**2))))
        mHz2 = -(self.c0/(mHz0*self.mu_zz))
        mHz3 = -(self.c0*self.dt*self.sz/(mHz0*self.eps_o*self.mu_zz))
        mHz4 = -(self.dt*self.sx*self.sy/(mHz0*(self.eps_o**2)))
        ICEz = np.zeros((self.Nz, self.Nx, self.Ny))
        IHz = np.zeros((self.Nz, self.Nx, self.Ny))
        CEz = np.zeros((self.Nz, self.Nx, self.Ny))

        mDx0 = (1/self.dt) + ((self.sy+self.sz)/(2*self.eps_o)) + (self.sz*self.sz*self.dt/(4*(self.eps_o**2)))
        mDx1 = (1/mDx0)*((1/self.dt) - ((self.sy+self.sz)/(2*self.eps_o)) - (self.sy*self.sz*self.dt/(4*(self.eps_o**2))))
        mDx2 = (self.c0/mDx0)
        mDx3 = (self.c0*self.dt*self.sx/(mDx0*self.eps_o))
        mDx4 = -(self.dt*self.sy*self.sz/(mDx0*(self.eps_o**2)))
        ICHx = np.zeros((self.Nz, self.Nx, self.Ny))
        IDx = np.zeros((self.Nz, self.Nx, self.Ny))
        CHx = np.zeros((self.Nz, self.Nx, self.Ny))

        mDy0 = (1/self.dt) + ((self.sx+self.sz)/(2*self.eps_o)) + (self.sx*self.sz*self.dt/(4*(self.eps_o**2)))
        mDy1 = (1/mDy0)*((1/self.dt) - ((self.sx+self.sz)/(2*self.eps_o)) - (self.sx*self.sz*self.dt/(4*(self.eps_o**2))))
        mDy2 = (self.c0/mDy0)
        mDy3 = (self.c0*self.dt*self.sy/(mDy0*self.eps_o))
        mDy4 = -(self.dt*self.sx*self.sz/(mDy0*(self.eps_o**2)))
        ICHy = np.zeros((self.Nz, self.Nx, self.Ny))
        IDy = np.zeros((self.Nz, self.Nx, self.Ny))
        CHy = np.zeros((self.Nz, self.Nx, self.Ny))

        mDz0 = (1/self.dt) + ((self.sx+self.sy)/(2*self.eps_o)) + (self.sx*self.sy*self.dt/(4*(self.eps_o**2)))
        mDz1 = (1/mDz0)*((1/self.dt) - ((self.sx+self.sy)/(2*self.eps_o)) - (self.sx*self.sy*self.dt/(4*(self.eps_o**2))))
        mDz2 = (self.c0/mDz0)
        mDz3 = (self.c0*self.dt*self.sz/(mDz0*self.eps_o))
        mDz4 = -(self.dt*self.sx*self.sy/(mDz0*(self.eps_o**2)))
        ICHz = np.zeros((self.Nz, self.Nx, self.Ny))
        IDz = np.zeros((self.Nz, self.Nx, self.Ny))
        CHz = np.zeros((self.Nz, self.Nx, self.Ny))

        mEx1 = 1/self.eps_xx
        mEy1 = 1/self.eps_yy
        mEz1 = 1/self.eps_zz

        # ====== Record Frames For Animation =======
        if fr is not None:
            self.frames = int(np.ceil(self.S / fr))
            self.Etot = np.zeros((int(np.ceil((self.S - 2) / self.frames)), self.Nz, self.Nx, self.Ny))
            self.Htot = np.zeros((int(np.ceil((self.S - 2) / self.frames)), self.Nz, self.Nx, self.Ny))
        else:
            self.frames = self.S

        # ======== Time Evolution ====================
        for i in tqdm(range(self.S - 2)):
            # =============== Update PML parameters=========================
            CEx[0:-1, :, 0:-1] = (self.Ez[0:-1, :, 1:] - self.Ez[0:-1, :, 0:-1] - self.Ey[1:, :, 0:-1] + self.Ey[0:-1, :, 0:-1]) / self.delta
            CEx[0:-1, :, -1] = (0 - self.Ez[0:-1, :, -1] - self.Ey[1:, :, -1] + self.Ey[0:-1, :, -1]) / self.delta
            CEx[-1, :, 0:-1] = (self.Ez[-1, :, 1:] - self.Ez[-1, :, 0:-1] - 0 + self.Ey[-1, :, 0:-1]) / self.delta
            CEx[-1, :, -1] = (0 - self.Ez[-1, :, -1] - 0 + self.Ey[-1, :, -1]) / self.delta
            ICEx += CEx
            IHx += self.Hx
            CEy[0:-1, 0:-1, :] = (self.Ex[1:, 0:-1, :] - self.Ex[0:-1, 0:-1, :] - self.Ez[0:-1, 1:, :] + self.Ez[0:-1, 0:-1, :]) / self.delta
            CEy[-1, 0:-1, :] = (0 - self.Ex[-1, 0:-1, :] - self.Ez[-1, 1:, :] + self.Ez[-1, 0:-1, :]) / self.delta
            CEy[0:-1, -1, :] = (self.Ex[1:, -1, :] - self.Ex[0:-1, -1, :] - 0 + self.Ez[0:-1, -1, :]) / self.delta
            CEy[-1, -1, :] = (0 - self.Ex[-1, -1, :] - 0 + self.Ez[-1, -1, :]) / self.delta
            ICEy += CEy
            IHy += self.Hy
            CEz[:, 0:-1, 0:-1] = (self.Ey[:, 1:, 0:-1] - self.Ey[:, 0:-1, 0:-1] - self.Ex[:, 0:-1, 1:] + self.Ex[:, 0:-1, 0:-1]) / self.delta
            CEz[:, -1, 0:-1] = (0 - self.Ey[:, -1, 0:-1] - self.Ex[:, -1, 1:] + self.Ex[:, -1, 0:-1]) / self.delta
            CEz[:, 0:-1, -1] = (self.Ey[:, 1:, -1] - self.Ey[:, 0:-1, -1] - 0 + self.Ex[:, 0:-1, -1]) / self.delta
            CEz[:, -1, -1] = (0 - self.Ey[:, -1, -1] - 0 + self.Ex[:, -1, -1]) / self.delta
            ICEy += CEz
            IHz += self.Hz

            self.Hx = mHx1*self.Hx + mHx2*CEx + mHx3*ICEx + mHx4*IHx
            self.Hy = mHy1*self.Hy + mHy2*CEy + mHy3*ICEy + mHy4*IHy
            self.Hz = mHz1*self.Hz + mHz2*CEz + mHz3*ICEz + mHz4*IHz

            CHx[1:, :, 1:] = (self.Hz[1:, :, 1:] - self.Hz[1:, :, 0:-1] - self.Hy[1:, :, 1:] + self.Hy[0:-1, :, 1:]) / self.delta
            CHx[1:, :, 0] = (self.Hz[1:, :, 0] - 0 - self.Hy[1:, :, 0] + self.Hy[0:-1, :, 0]) / self.delta
            CHx[0, :, 1:] = (self.Hz[0, :, 1:] - self.Hz[0, :, 0:-1] - self.Hy[0, :, 1:] + 0) / self.delta
            CHx[0, :, 0] = (self.Hz[0, :, 0] - 0 - self.Hy[0, :, 0] + 0) / self.delta
            ICHx += CHx
            IDx += self.Dx
            CHy[1:, 1:, :] = (self.Hx[1:, 1:, :] - self.Hx[0:-1, 1:, :] - self.Hz[1:, 1:, :] + self.Hz[1:, 0:-1, :]) / self.delta
            CHy[0, 1:, :] = (self.Hx[0, 1:, :] - 0 - self.Hz[0, 1:, :] + self.Hz[0, 0:-1, :]) / self.delta
            CHy[1:, 0, :] = (self.Hx[1:, 0, :] - self.Hx[0:-1, 0, :] - self.Hz[1:, 0, :] + 0) / self.delta
            CHy[0, 0, :] = (self.Hx[0, 0, :] - 0 - self.Hz[0, 0, :] + 0) / self.delta
            ICHy += CHy
            IDy += self.Dy
            CHz[:, 1:, 1:] = (self.Hy[:, 1:, 1:] - self.Hy[:, 0:-1, 1:] - self.Hx[:, 1:, 1:] + self.Hx[:, 1:, 0:-1]) / self.delta
            CHz[:, 0, 1:] = (self.Hy[:, 0, 1:] - 0 - self.Hx[:, 0, 1:] + self.Hx[:, 0, 0:-1]) / self.delta
            CHz[:, 1:, 0] = (self.Hy[:, 1:, 0] - self.Hy[:, 0:-1, 0] - self.Hx[:, 1:, 0] + 0) / self.delta
            CHz[:, 0, 0] = (self.Hy[:, 0, 0] - 0 - self.Hx[:, 0, 0] + 0) / self.delta
            ICHy += CHz
            IDz += self.Dz

            self.Dx = mDx1*self.Dx + mDx2*CHx + mDx3*ICHx + mDx4*IDx
            self.Dy = mDy1*self.Dy + mDy2*CHy + mDy3*ICHy + mDy4*IDy
            self.Dz = mDz1*self.Dz + mDz2*CHz + mDz3*ICHz + mDz4*IDz

            self.Ez = mEx1 * self.Dz
            self.Ex = mEy1 * self.Dx
            self.Ey = mEz1 * self.Dy

            self.Ez[self.src_ini[2], self.src_ini[0], self.src_ini[1]] += self.Esrc_z[i]
            self.Ey[self.src_ini[0], self.src_ini[1], self.src_ini[2]] += self.Esrc_y[i]
            self.Ex[self.src_ini[0], self.src_ini[1], self.src_ini[2]] += self.Esrc_x[i]
            self.Hz[self.src_ini[0], self.src_ini[1], self.src_ini[2]] += self.Hsrc_z[i]
            self.Hy[self.src_ini[0], self.src_ini[1], self.src_ini[2]] += self.Hsrc_y[i]
            self.Hx[self.src_ini[0], self.src_ini[1], self.src_ini[2]] += self.Hsrc_x[i]

            if i % self.frames == 0:
                if fr is not None:
                    self.Htot[int(np.   ceil(i / self.frames)), :, :, :] = self.Hx
                    self.Etot[int(np.ceil(i / self.frames)), :, :, :] = self.Ez

    def create_ani(self):

        fig = plt.figure()
        plts = []

        for i in range(int(np.floor((self.S - 2) / self.frames))):
            im = plt.imshow(self.Etot[i, self.src_ini[0], :, :], vmin=-0.05, vmax=0.05, animated=True)
            plts.append([im])

        ani = animation.ArtistAnimation(fig, plts, interval=50, repeat_delay=3000)
        plt.show()