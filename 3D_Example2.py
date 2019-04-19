import owlmPy as ow


dx = 300
dy = 300
dz = 300
pml = 30
lambda_min = 100
lambda_max = 1000
n_lambd = 20
src_ini=[0.5, 0.5, 0.5]
epsrc=1
musrc=1
NFREQ = 1000

model = ow.Fdtd3d(dx, dy, dz, pml, lambda_min, lambda_max, n_lambd, src_ini, epsrc, musrc, NFREQ)
model.run_sim_pml(fr=100)
model.create_ani()
