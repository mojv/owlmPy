import owlmPy as ow

x = 500  # with of the domain in nm
y = 500  # large of the domain in nm
z = 500  # height of the domain in nm
pml = 100  # size of the Perfectly matched layer (PML) in nm
lambda_min = 100  # minimum wave length in nm
lambda_max = 1000  # Maximum wave length in nmimport numexpr as ne
n_lambd = 5  # Number of cells in a wavelength
src_ini = [0.5, 0.5, 0.5]  # relative position of source (in % of dx, dy and dz)

model = ow.Fdtd3d(x, y, z, pml, lambda_min, lambda_max, n_lambd, src_ini)
model.run_sim(fr=600)
model.create_ani()
