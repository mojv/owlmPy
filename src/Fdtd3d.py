import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm


class Fdtd3d:

    def __init__(self, x_in, y_in, z_in, pml_in, lambda_min_in, lambda_max_in, n_lambd, src_ini_in):

        self.c0 = 299792458 * 100  # speed of light cm/s
        self.eps_o = 1  # permittivity of vacuum

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
        self.src_ini = [int(src_ini_in[2] * self.Nz), int(src_ini_in[0] * self.Nx),
                        int(src_ini_in[1] * self.Ny)]  # position of the source

        # ======Calculation of steps in the temporal yee grid ========
        self.dt = 1 / (self.c0 * np.sqrt((1 / np.square(self.delta)) * 3))  # time of each step
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
            self.sx[:, -i - 1, :] = (self.Npml - i) / self.Npml
            self.sy[:, :, i] = (self.Npml - i) / self.Npml
            self.sy[:, :, -i - 1] = (self.Npml - i) / self.Npml
            self.sz[i, :, :] = (self.Npml - i) / self.Npml
            self.sz[-i - 1, :, :] = (self.Npml - i) / self.Npml
        # Converting to conductivity
        self.sx = (self.eps_o / (2 * self.dt)) * (self.sx ** 3)
        self.sy = (self.eps_o / (2 * self.dt)) * (self.sx ** 3)
        self.sz = (self.eps_o / (2 * self.dt)) * (self.sx ** 3)
        # self.sx = self.sx * 0
        # self.sy = self.sy * 0
        # self.sz = self.sz * 0

        # ====== Compute the Source Functions ========
        delay = self.delta / (2 * self.c0) + self.dt / 2  # total delay because Yee Grid
        self.Esrc_x = np.exp(-((self.t - 6 * tao + delay) / tao) ** 2)  # E field source
        self.Esrc_y = np.exp(-((self.t - 6 * tao + delay) / tao) ** 2)  # E field source
        self.Esrc_z = np.exp(-((self.t - 6 * tao + delay) / tao) ** 2)  # E field source
        self.Hsrc_x = -np.exp(-((self.t - 6 * tao + delay) / tao) ** 2)  # H field source
        self.Hsrc_y = -np.exp(-((self.t - 6 * tao + delay) / tao) ** 2)  # H field source
        self.Hsrc_z = -np.exp(-((self.t - 6 * tao + delay) / tao) ** 2)  # H field source

        # ====== Initialize permittivity and permeability ====
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

        # ====== Initialize Fields Variables =============
        self.Hx = np.zeros((self.Nz, self.Nx, self.Ny))
        self.Hy = np.zeros((self.Nz, self.Nx, self.Ny))
        self.Hz = np.zeros((self.Nz, self.Nx, self.Ny))
        self.Dx = np.zeros((self.Nz, self.Nx, self.Ny))
        self.Dy = np.zeros((self.Nz, self.Nx, self.Ny))
        self.Dz = np.zeros((self.Nz, self.Nx, self.Ny))
        self.Ex = np.zeros((self.Nz, self.Nx, self.Ny))
        self.Ey = np.zeros((self.Nz, self.Nx, self.Ny))
        self.Ez = np.zeros((self.Nz, self.Nx, self.Ny))

        # ===============Setting PML update coefficients ==========================================
        self.mHx0 = (1 / self.dt) + ((self.sy + self.sz) / (2 * self.eps_o)) + (
                    self.sy * self.sz * self.dt / (4 * (self.eps_o ** 2)))
        self.mHx1 = (1 / self.mHx0) * ((1 / self.dt) - ((self.sy + self.sz) / (2 * self.eps_o)) - (
                    self.sy * self.sz * self.dt / (4 * (self.eps_o ** 2))))
        self.mHx2 = -(self.c0 / (self.mHx0 * self.mu_xx))
        self.mHx3 = -(self.c0 * self.dt * self.sx / (self.mHx0 * self.eps_o * self.mu_xx))
        self.mHx4 = -(self.dt * self.sy * self.sz / (self.mHx0 * (self.eps_o ** 2)))
        self.ICEx = np.zeros((self.Nz, self.Nx, self.Ny))
        self.IHx = np.zeros((self.Nz, self.Nx, self.Ny))
        self.CEx = np.zeros((self.Nz, self.Nx, self.Ny))

        self.mHy0 = (1 / self.dt) + ((self.sx + self.sz) / (2 * self.eps_o)) + (
                    self.sx * self.sz * self.dt / (4 * (self.eps_o ** 2)))
        self.mHy1 = (1 / self.mHy0) * ((1 / self.dt) - ((self.sx + self.sz) / (2 * self.eps_o)) - (
                    self.sx * self.sz * self.dt / (4 * (self.eps_o ** 2))))
        self.mHy2 = -(self.c0 / (self.mHy0 * self.mu_yy))
        self.mHy3 = -(self.c0 * self.dt * self.sy / (self.mHy0 * self.eps_o * self.mu_yy))
        self.mHy4 = -(self.dt * self.sx * self.sz / (self.mHy0 * (self.eps_o ** 2)))
        self.ICEy = np.zeros((self.Nz, self.Nx, self.Ny))
        self.IHy = np.zeros((self.Nz, self.Nx, self.Ny))
        self.CEy = np.zeros((self.Nz, self.Nx, self.Ny))

        self.mHz0 = (1 / self.dt) + ((self.sx + self.sy) / (2 * self.eps_o)) + (
                    self.sx * self.sy * self.dt / (4 * (self.eps_o ** 2)))
        self.mHz1 = (1 / self.mHz0) * ((1 / self.dt) - ((self.sx + self.sy) / (2 * self.eps_o)) - (
                    self.sx * self.sy * self.dt / (4 * (self.eps_o ** 2))))
        self.mHz2 = -(self.c0 / (self.mHz0 * self.mu_zz))
        self.mHz3 = -(self.c0 * self.dt * self.sz / (self.mHz0 * self.eps_o * self.mu_zz))
        self.mHz4 = -(self.dt * self.sx * self.sy / (self.mHz0 * (self.eps_o ** 2)))
        self.ICEz = np.zeros((self.Nz, self.Nx, self.Ny))
        self.IHz = np.zeros((self.Nz, self.Nx, self.Ny))
        self.CEz = np.zeros((self.Nz, self.Nx, self.Ny))

        self.mDx0 = (1 / self.dt) + ((self.sy + self.sz) / (2 * self.eps_o)) + (
                    self.sz * self.sz * self.dt / (4 * (self.eps_o ** 2)))
        self.mDx1 = (1 / self.mDx0) * ((1 / self.dt) - ((self.sy + self.sz) / (2 * self.eps_o)) - (
                    self.sy * self.sz * self.dt / (4 * (self.eps_o ** 2))))
        self.mDx2 = (self.c0 / self.mDx0)
        self.mDx3 = (self.c0 * self.dt * self.sx / (self.mDx0 * self.eps_o))
        self.mDx4 = -(self.dt * self.sy * self.sz / (self.mDx0 * (self.eps_o ** 2)))
        self.ICHx = np.zeros((self.Nz, self.Nx, self.Ny))
        self.IDx = np.zeros((self.Nz, self.Nx, self.Ny))
        self.CHx = np.zeros((self.Nz, self.Nx, self.Ny))

        self.mDy0 = (1 / self.dt) + ((self.sx + self.sz) / (2 * self.eps_o)) + (
                    self.sx * self.sz * self.dt / (4 * (self.eps_o ** 2)))
        self.mDy1 = (1 / self.mDy0) * ((1 / self.dt) - ((self.sx + self.sz) / (2 * self.eps_o)) - (
                    self.sx * self.sz * self.dt / (4 * (self.eps_o ** 2))))
        self.mDy2 = (self.c0 / self.mDy0)
        self.mDy3 = (self.c0 * self.dt * self.sy / (self.mDy0 * self.eps_o))
        self.mDy4 = -(self.dt * self.sx * self.sz / (self.mDy0 * (self.eps_o ** 2)))
        self.ICHy = np.zeros((self.Nz, self.Nx, self.Ny))
        self.IDy = np.zeros((self.Nz, self.Nx, self.Ny))
        self.CHy = np.zeros((self.Nz, self.Nx, self.Ny))

        self.mDz0 = (1 / self.dt) + ((self.sx + self.sy) / (2 * self.eps_o)) + (
                    self.sx * self.sy * self.dt / (4 * (self.eps_o ** 2)))
        self.mDz1 = (1 / self.mDz0) * ((1 / self.dt) - ((self.sx + self.sy) / (2 * self.eps_o)) - (
                    self.sx * self.sy * self.dt / (4 * (self.eps_o ** 2))))
        self.mDz2 = (self.c0 / self.mDz0)
        self.mDz3 = (self.c0 * self.dt * self.sz / (self.mDz0 * self.eps_o))
        self.mDz4 = -(self.dt * self.sx * self.sy / (self.mDz0 * (self.eps_o ** 2)))
        self.ICHz = np.zeros((self.Nz, self.Nx, self.Ny))
        self.IDz = np.zeros((self.Nz, self.Nx, self.Ny))
        self.CHz = np.zeros((self.Nz, self.Nx, self.Ny))

        self.mEx1 = 1 / self.eps_xx
        self.mEy1 = 1 / self.eps_yy
        self.mEz1 = 1 / self.eps_zz

    def run_sim(self, fr=None):

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
            self.ICEx += self.CEx
            self.IHx += self.Hx
            self.CEx[0:-1, :, 0:-1] = (self.Ez[0:-1, :, 1:] - self.Ez[0:-1, :, 0:-1] -
                                       self.Ey[1:, :, 0:-1] + self.Ey[0:-1, :,0:-1]) / self.delta
            self.CEx[0:-1, :, -1] = (0 - self.Ez[0:-1, :, -1] - self.Ey[1:, :, -1] + self.Ey[0:-1, :, -1]) / self.delta
            self.CEx[-1, :, 0:-1] = (self.Ez[-1, :, 1:] - self.Ez[-1, :, 0:-1] - 0 + self.Ey[-1, :, 0:-1]) / self.delta
            self.CEx[-1, :, -1] = (0 - self.Ez[-1, :, -1] - 0 + self.Ey[-1, :, -1]) / self.delta

            self.ICEy += self.CEy
            self.IHy += self.Hy
            self.CEy[0:-1, 0:-1, :] = (self.Ex[1:, 0:-1, :] - self.Ex[0:-1, 0:-1, :] -
                                       self.Ez[0:-1, 1:, :] + self.Ez[0:-1, 0:-1, :]) / self.delta
            self.CEy[-1, 0:-1, :] = (0 - self.Ex[-1, 0:-1, :] - self.Ez[-1, 1:, :] + self.Ez[-1, 0:-1, :]) / self.delta
            self.CEy[0:-1, -1, :] = (self.Ex[1:, -1, :] - self.Ex[0:-1, -1, :] - 0 + self.Ez[0:-1, -1, :]) / self.delta
            self.CEy[-1, -1, :] = (0 - self.Ex[-1, -1, :] - 0 + self.Ez[-1, -1, :]) / self.delta

            self.ICEy += self.CEz
            self.IHz += self.Hz
            self.CEz[:, 0:-1, 0:-1] = (self.Ey[:, 1:, 0:-1] - self.Ey[:, 0:-1, 0:-1] -
                                       self.Ex[:, 0:-1, 1:] + self.Ex[:, 0:-1, 0:-1]) / self.delta
            self.CEz[:, -1, 0:-1] = (0 - self.Ey[:, -1, 0:-1] - self.Ex[:, -1, 1:] + self.Ex[:, -1, 0:-1]) / self.delta
            self.CEz[:, 0:-1, -1] = (self.Ey[:, 1:, -1] - self.Ey[:, 0:-1, -1] - 0 + self.Ex[:, 0:-1, -1]) / self.delta
            self.CEz[:, -1, -1] = (0 - self.Ey[:, -1, -1] - 0 + self.Ex[:, -1, -1]) / self.delta

            self.Hx = self.mHx1 * self.Hx + self.mHx2 * self.CEx + self.mHx3 * self.ICEx + self.mHx4 * self.IHx
            self.Hy = self.mHy1 * self.Hy + self.mHy2 * self.CEy + self.mHy3 * self.ICEy + self.mHy4 * self.IHy
            self.Hz = self.mHz1 * self.Hz + self.mHz2 * self.CEz + self.mHz3 * self.ICEz + self.mHz4 * self.IHz

            self.ICHx += self.CHx
            self.IDx += self.Dx
            self.CHx[1:, :, 1:] = (self.Hz[1:, :, 1:] - self.Hz[1:, :, 0:-1] -
                                   self.Hy[1:, :, 1:] + self.Hy[0:-1, :, 1:]) / self.delta
            self.CHx[1:, :, 0] = (self.Hz[1:, :, 0] - 0 - self.Hy[1:, :, 0] + self.Hy[0:-1, :, 0]) / self.delta
            self.CHx[0, :, 1:] = (self.Hz[0, :, 1:] - self.Hz[0, :, 0:-1] - self.Hy[0, :, 1:] + 0) / self.delta
            self.CHx[0, :, 0] = (self.Hz[0, :, 0] - 0 - self.Hy[0, :, 0] + 0) / self.delta

            self.ICHy += self.CHy
            self.IDy += self.Dy
            self.CHy[1:, 1:, :] = (self.Hx[1:, 1:, :] - self.Hx[0:-1, 1:, :] -
                                   self.Hz[1:, 1:, :] + self.Hz[1:, 0:-1, :]) / self.delta
            self.CHy[0, 1:, :] = (self.Hx[0, 1:, :] - 0 - self.Hz[0, 1:, :] + self.Hz[0, 0:-1, :]) / self.delta
            self.CHy[1:, 0, :] = (self.Hx[1:, 0, :] - self.Hx[0:-1, 0, :] - self.Hz[1:, 0, :] + 0) / self.delta
            self.CHy[0, 0, :] = (self.Hx[0, 0, :] - 0 - self.Hz[0, 0, :] + 0) / self.delta

            self.ICHy += self.CHz
            self.IDz += self.Dz
            self.CHz[:, 1:, 1:] = (self.Hy[:, 1:, 1:] - self.Hy[:, 0:-1, 1:] -
                                   self.Hx[:, 1:, 1:] + self.Hx[:, 1:, 0:-1]) / self.delta
            self.CHz[:, 0, 1:] = (self.Hy[:, 0, 1:] - 0 - self.Hx[:, 0, 1:] + self.Hx[:, 0, 0:-1]) / self.delta
            self.CHz[:, 1:, 0] = (self.Hy[:, 1:, 0] - self.Hy[:, 0:-1, 0] - self.Hx[:, 1:, 0] + 0) / self.delta
            self.CHz[:, 0, 0] = (self.Hy[:, 0, 0] - 0 - self.Hx[:, 0, 0] + 0) / self.delta

            self.Dx = self.mDx1 * self.Dx + self.mDx2 * self.CHx + self.mDx3 * self.ICHx + self.mDx4 * self.IDx
            self.Dy = self.mDy1 * self.Dy + self.mDy2 * self.CHy + self.mDy3 * self.ICHy + self.mDy4 * self.IDy
            self.Dz = self.mDz1 * self.Dz + self.mDz2 * self.CHz + self.mDz3 * self.ICHz + self.mDz4 * self.IDz

            self.Ez = self.mEx1 * self.Dz
            self.Ex = self.mEy1 * self.Dx
            self.Ey = self.mEz1 * self.Dy

            self.Ez[self.src_ini[2], self.src_ini[0], self.src_ini[1]] = self.Ez[self.src_ini[2], self.src_ini[0],
                                                                                 self.src_ini[1]] + self.Esrc_z[i]
            self.Ey[self.src_ini[0], self.src_ini[1], self.src_ini[2]] = self.Ey[self.src_ini[0], self.src_ini[1],
                                                                                 self.src_ini[2]] + self.Esrc_y[i]
            self.Ex[self.src_ini[0], self.src_ini[1], self.src_ini[2]] = self.Ex[self.src_ini[0], self.src_ini[1],
                                                                                 self.src_ini[2]] + self.Esrc_x[i]
            self.Hz[self.src_ini[0], self.src_ini[1], self.src_ini[2]] = self.Hz[self.src_ini[0], self.src_ini[1],
                                                                                 self.src_ini[2]] + self.Hsrc_z[i]
            self.Hy[self.src_ini[0], self.src_ini[1], self.src_ini[2]] = self.Hy[self.src_ini[0], self.src_ini[1],
                                                                                 self.src_ini[2]] + self.Hsrc_y[i]
            self.Hx[self.src_ini[0], self.src_ini[1], self.src_ini[2]] = self.Hx[self.src_ini[0], self.src_ini[1],
                                                                                 self.src_ini[2]] + self.Hsrc_x[i]

            if i % self.frames == 0:
                if fr is not None:
                    self.Htot[int(np.ceil(i / self.frames)), :, :, :] = self.Hx
                    self.Etot[int(np.ceil(i / self.frames)), :, :, :] = self.Ez

    def create_ani(self):

        fig = plt.figure()
        plts = []

        for i in range(int(np.floor((self.S - 2) / self.frames))):
            im = plt.imshow(self.Etot[i, self.src_ini[0], :, :], vmin=-0.05, vmax=0.05, animated=True)
            plts.append([im])

        ani = animation.ArtistAnimation(fig, plts, interval=50, repeat_delay=3000)
        plt.show()