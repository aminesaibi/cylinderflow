import numpy as np
import os
import dill
import sys
from rl.model import Model
from utils import load_yaml
import torch
import math
from tqdm import tqdm
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import (Constant, Function, FunctionSpace, dirichletbc, form,
                         assemble_scalar, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.io import (gmshio, XDMFFile)
from ufl import (FiniteElement, TestFunction, TrialFunction, VectorElement, TensorElement,
                 div, dot, dx, inner, lhs, grad, nabla_grad, rhs)
from copy import copy
import pickle
import contextlib
import io
torch.set_default_dtype(torch.float64)


class NS():
    def __init__(self,
                 comm=MPI.COMM_WORLD,
                 dt=1.0e-2,
                 mu=0.001,
                 rho=1,
                 xa=[0.3, 0.2],
                 xs=[2.0, 0.2],
                 var=0.4,
                 controller=None
                 ):
        self.comm = comm
        self.xs = np.array(xs)
        self.ns = self.xs.shape[0]
        self.xa = np.array(xa)
        self.na = self.xa.shape[0]
        self.var = var
        self.gdim = 2
        self.mesh, _, self.ft = gmshio.read_from_msh("./environment/mesh.msh",
                                                     self.comm,
                                                     rank=0,
                                                     gdim=self.gdim
                                                     )
        if controller is not None:
            model_params = load_yaml(controller+'model_params')
            self.controller = Model(self.ns, self.na, **model_params)
            self.controller.load_state_dict(torch.load(controller+'state_dict.pt'))

        self.ft.name = "Facet markers"

        # define our problem specific physical and descritization parameters
        self.dt = Constant(self.mesh, PETSc.ScalarType(dt))
        self.mu = Constant(self.mesh, PETSc.ScalarType(mu))  # Dynamic viscosity
        self.rho = Constant(self.mesh, PETSc.ScalarType(rho))     # Density

        v_cg2 = VectorElement("CG", self.mesh.ufl_cell(), 2)
        s_cg1 = FiniteElement("CG", self.mesh.ufl_cell(), 1)
        self.V = FunctionSpace(self.mesh, v_cg2)
        self.Q = FunctionSpace(self.mesh, s_cg1)

        # Boundary conditions
        # As we have created the mesh and relevant mesh tags, we can now specify the function spaces V and Q along with
        # the boundary conditions. As the ft contains markers for facets,
        # we use this class to find the facets for the inlet and walls.
        fdim = self.mesh.topology.dim - 1

        inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5

        # Inlet
        u_inlet = Function(self.V)
        # inlet_velocity = InletVelocity(t)
        u_inlet.interpolate(self.inlet_velocity)
        bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(
            self.V, fdim, self.ft.find(inlet_marker)))
        # Walls
        u_nonslip = np.array((0,) * self.mesh.geometry.dim, dtype=PETSc.ScalarType)
        bcu_walls = dirichletbc(u_nonslip, locate_dofs_topological(
            self.V, fdim, self.ft.find(wall_marker)), self.V)
        # Obstacle
        bcu_obstacle = dirichletbc(u_nonslip, locate_dofs_topological(
            self.V, fdim, self.ft.find(obstacle_marker)), self.V)
        self.bcu = [bcu_inflow, bcu_obstacle, bcu_walls]
        # Outlet
        bcp_outlet = dirichletbc(PETSc.ScalarType(
            0), locate_dofs_topological(self.Q, fdim, self.ft.find(outlet_marker)), self.Q)
        self.bcp = [bcp_outlet]

        ##################################################
        # Variational form
        ############################################
        # defining all the variables used in the variational formulations

        # Trial and test functions
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        p = TrialFunction(self.Q)
        q = TestFunction(self.Q)

        # Velocity solutions
        self.u_ = Function(self.V)
        self.u_.name = "u"
        self.u_s = Function(self.V)
        self.u_n = Function(self.V)
        self.u_n1 = Function(self.V)

        # Pressure solutions
        # self.p_n = Function(self.Q)
        self.p_ = Function(self.Q)
        self.p_.name = "p"
        self.phi = Function(self.Q)

        # forcing term
        # f = Constant(self.mesh, PETSc.ScalarType((0, 0)))
        self.forcing = self.InstForcing()
        self.f = self.forcing.get_function()

        self.sensors = self.InstSensors()

        self.r = form(inner(self.u_, self.u_)*dx)

        # we define the variational formulation for the first step, where we have integrated the diffusion term,
        # as well as the pressure term by parts.
        F1 = self.rho / self.dt * dot(u - self.u_n, v) * dx
        F1 += inner(dot(1.5 * self.u_n - 0.5 * self.u_n1, 0.5 * nabla_grad(u + self.u_n)), v) * dx
        # F1 += inner(dot(self.u_n, 0.5 * nabla_grad(u + self.u_n)), v) * dx
        # F1 += inner(self.rk4(nl, self.u_n, self.dt), v) * dx
        F1 += 0.5 * self.mu * inner(grad(u + self.u_n), grad(v))*dx - dot(self.p_, div(v))*dx
        # F1 += 0.5 * self.mu * inner(grad(u + self.u_n), grad(v))*dx - dot(self.p_n, div(v))*dx
        F1 -= dot(self.f, v) * dx
        self.a1 = form(lhs(F1))
        self.L1 = form(rhs(F1))
        self.A1 = create_matrix(self.a1)
        # self.A1.zeroEntries()
        # assemble_matrix(self.A1, self.a1, bcs=self.bcu)
        # self.A1.assemble()
        # viewer = PETSc.Viewer().createBinary('./ns_cylinder/amat/A.dat', 'w')
        # viewer(self.A1)
        # A = copy(self.A1)
        # A.convert('dense')
        # A= A.getDenseArray()
        # np.savetxt(f"./ns_cylinder/amat/{t}", A)
        self.b1 = create_vector(self.L1)

        # Next we define the second step
        self.a2 = form(dot(grad(p), grad(q))*dx)
        self.L2 = form(-self.rho / self.dt * dot(div(self.u_s), q) * dx)
        A2 = assemble_matrix(self.a2, bcs=self.bcp)
        A2.assemble()
        self.b2 = create_vector(self.L2)

        # finally create the last step
        a3 = form(self.rho * dot(u, v)*dx)
        self.L3 = form(self.rho * dot(self.u_s, v)*dx - self.dt * dot(nabla_grad(self.phi), v)*dx)
        # self.L3 = form(self.rho * dot(self.u_s, v)*dx - self.dt * dot(nabla_grad(self.p_), v)*dx)
        A3 = assemble_matrix(a3)
        A3.assemble()
        self.b3 = create_vector(self.L3)

        # we use PETSc as a linear algebra backend
        # Solver for step 1
        self.solver1 = PETSc.KSP().create(self.comm)
        self.solver1.setOperators(self.A1)
        self.solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = self.solver1.getPC()
        pc1.setType(PETSc.PC.Type.JACOBI)

        # Solver for step 2
        self.solver2 = PETSc.KSP().create(self.comm)
        self.solver2.setOperators(A2)
        self.solver2.setType(PETSc.KSP.Type.MINRES)
        pc2 = self.solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")

        # Solver for step 3
        self.solver3 = PETSc.KSP().create(self.comm)
        self.solver3.setOperators(A3)
        self.solver3.setType(PETSc.KSP.Type.CG)
        pc3 = self.solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)

        # create output files for the velocity and pressure
        # self.file = XDMFFile(self.comm, "./ns_cylinder/u_p.xdmf", "w")
        # self.file.write_mesh(self.mesh)
        # if self.comm.rank == 0:
        #     print('Environment initialized')

    @staticmethod
    def rk4(f, x0, dt=0.1):
        k1 = dt*f(x0)
        k2 = dt*f(x0 + 0.5*k1)
        k3 = dt*f(x0 + 0.5*k2)
        k4 = dt*f(x0 + k3)
        x1 = x0 + (k1+2*(k2+k3) + k4)/6
        return x1

    @staticmethod
    def nl(u):
        return dot(u, nabla_grad(u))

    class Forcing:
        def __init__(self, NS_obj):
            self.xa = NS_obj.xa
            self.na = self.xa.shape[0]
            self.var = NS_obj.var
            self.gdim = NS_obj.gdim
            self.fun = Function(NS_obj.V)
            self.coef = (1/(np.sqrt(2*math.pi)*self.var))
            self.u = np.zeros((1, self.na))

        def get_function(self, u=None):
            if u is not None:
                self.u = u
            self.fun.interpolate(self.__call__)
            return self.fun

        def __call__(self, x):
            values = np.zeros((self.gdim, x.shape[1]), dtype=PETSc.ScalarType)
            values[0] = (self.u*self.coef * np.exp(-(np.tile(x[0].reshape(-1, 1),
                         self.na) - self.xa[:, 0])**2/self.var)).sum(1)
            values[1] = (self.u*self.coef * np.exp(-(np.tile(x[1].reshape(-1, 1),
                         self.na) - self.xa[:, 1])**2/self.var)).sum(1)
            return values

    class Sensors:
        def __init__(self, NS_obj):
            self.xs = NS_obj.xs
            self.ns = self.xs.shape[0]
            self.var = NS_obj.var
            self.gdim = NS_obj.gdim
            self.coef = (1/(np.sqrt(2*math.pi)*self.var))

            v_cg3 = TensorElement("CG", NS_obj.mesh.ufl_cell(), 2, shape=(3, 2))
            self.S = FunctionSpace(NS_obj.mesh, v_cg3)
            self.h = Function(self.S)
            self.h.interpolate(self.__call__)

        def build_forms(self, u):
            forms = [form(inner(self.h[i, :], u)*dx) for i in range(self.ns)]
            return forms

        def __call__(self, x):
            values = np.zeros((self.gdim, self.ns, x.shape[1]), dtype=PETSc.ScalarType)
            values[0] = self.coef * \
                np.exp(-(np.tile(x[0].reshape(-1, 1), self.ns) - self.xs[:, 0])**2/self.var).T
            values[1] = self.coef * \
                np.exp(-(np.tile(x[1].reshape(-1, 1), self.ns) - self.xs[:, 1])**2/self.var).T
            values = values.reshape((self.ns*self.gdim, -1), order='F')
            return values

        def get_values(self, u):
            forms = self.build_forms(u)
            y = np.array([assemble_scalar(form) for form in forms])
            cost = np.dot(y, y)
            return y, cost

    def InstSensors(self):
        return NS.Sensors(self)

    def InstForcing(self):
        return NS.Forcing(self)

    def inlet_velocity(self, x):
        values = np.zeros((self.gdim, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = 4 * 1.5 * x[1] * (0.41 - x[1])/(0.41**2)
        return values

    def init_cond_generator(self):
        return self.u_n, self.u_n1

    def advance(self, dt=None, u_n=None, u_n1=None, a=None):
        if dt is not None:
            self.dt = Constant(self.mesh, PETSc.ScalarType(dt))
        if u_n is not None:
            self.u_n = u_n
        if u_n1 is not None:
            self.u_n1 = u_n1

        self.f = self.forcing.get_function(a)

        # solve  for one-step the time-dependent problem
        # Step 1: Tentative velocity step
        self.A1.zeroEntries()
        assemble_matrix(self.A1, self.a1, bcs=self.bcu)
        self.A1.assemble()
        self.b1 = create_vector(self.L1)
        with self.b1.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [self.bcu])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, self.bcu)
        self.solver1.solve(self.b1, self.u_s.vector)
        self.u_s.x.scatter_forward()

        # Step 2: Pressure corrrection step
        with self.b2.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [self.bcp])
        self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, self.bcp)
        self.solver2.solve(self.b2, self.phi.vector)
        # self.solver2.solve(self.b2, self.p_.vector)
        self.phi.x.scatter_forward()
        # self.p_.x.scatter_forward()

        self.p_.vector.axpy(1, self.phi.vector)
        self.p_.x.scatter_forward()

        # Step 3: Velocity correction step
        with self.b3.localForm() as loc:
            loc.set(0)
        assemble_vector(self.b3, self.L3)
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self.solver3.solve(self.b3, self.u_.vector)
        self.u_.x.scatter_forward()

        energy = self.comm.gather(assemble_scalar(self.r), root=0)
        y, cost = self.sensors.get_values(self.u_)
        reward = -cost

        return (self.u_, self.p_), y, reward, energy

    def run_model(self, args):
        model, n_rollout, rank = args
        progress = tqdm(desc="Running environment", total=n_rollout, leave=False, position=2+rank)
        transitions = [None]*n_rollout
        with torch.no_grad():
            x1, _ = self.sensors.get_values(self.u_n1)
            x1 = torch.tensor(x1).unsqueeze(0)

            for i in range(n_rollout):
                progress.update(1)
                a = model.pi(x1)
                (_, _), x2, r, _ = self.advance(a=a.squeeze(0)
                                                .detach()
                                                .numpy()
                                                )
                x2 = torch.tensor(x2).unsqueeze(0)
                r = torch.tensor(r).view(1, 1)
                transitions[i] = (x1, a, r, x2)
        return transitions

    def run_with_controller(self, n_rollout):
        progress = tqdm(desc="Running environment", total=n_rollout, leave=False)
        transitions = [None]*n_rollout
        with torch.no_grad():
            x1, _ = self.sensors.get_values(self.u_n1)
            x1 = torch.tensor(x1).unsqueeze(0)

            for i in range(n_rollout):
                progress.update(1)
                a = self.controller.pi(x1)
                (_, _), x2, r, _ = self.advance(a=a.squeeze(0)
                                                .detach()
                                                .numpy()
                                                )
                x2 = torch.tensor(x2).unsqueeze(0)
                r = torch.tensor(r).view(1, 1)
                transitions[i] = (x1, a, r, x2)
        return transitions

    def run(self, t=0, T=5.0, controller=None):

        num_steps = int(T/self.dt.value)
        energies = np.zeros(num_steps)
        observations = np.zeros((num_steps, self.sensors.ns))
        actions = np.zeros((num_steps, self.forcing.na))

        progress = tqdm(desc="Solving PDE", total=num_steps)
        for i in range(num_steps):
            progress.update(1)

            # Update current time step
            t += self.dt.value
            # self.forcing.t = t
            if controller is not None:
                a = controller(torch.tensor(observations[i]))
                a = a.squeeze(0).detach().numpy()
            else:
                a = None
            self.f = self.forcing.get_function(a)

            (self.u_, self.p_), y, reward, energy = self.advance(t)

            if self.comm.rank == 0:
                actions[i] = a
                energies[i] = sum(energy)
                observations[i] = y

            # Write solutions to file
            self.file.write_function(self.p_, t)
            self.file.write_function(self.u_, t)

            # # Update variable with solution form this time step
            # with self.u_.vector.localForm() as u_, self.u_n.vector.localForm() as u_n:
            #     u_.copy(u_n)
            # with self.p_.vector.localForm() as p_, self.p_n.vector.localForm() as p_n:
            #     p_.copy(p_n)

            # Update variable with solution form this time step
            with self.u_.vector.localForm() as loc_, self.u_n.vector.localForm() as loc_n, self.u_n1.vector.localForm() as loc_n1:
                loc_n.copy(loc_n1)
                loc_.copy(loc_n)

        np.savetxt('./ns_cylinder/energies.txt', energies)
        np.savetxt('./ns_cylinder/observations.txt', observations)
        np.savetxt('./ns_cylinder/actions.txt', actions)
        self.file.close()


# def main(tmp_file):
#     ns = NS(
#         dt=1.0e-2,
#         mu=0.001,
#         rho=1,
#         xa=[[0.25, 0.15], [0.25, 0.25], [0.9, 0.2]],
#         xs=[[2.0, 0.2], [1.5, 0.2], [0.9, 0.2]],
#         var=0.4,
#         controller=tmp_file
#     )
#     ns.run(t=0, T=5.0)

# # Custom context manager to redirect stdout to a temporary file
# @contextlib.contextmanager
# def nostdout():
#     save_stdout = sys.stdout
#     sys.stdout = io.BytesIO()
#     yield
#     sys.stdout = save_stdout


if __name__ == "__main__":
    tmp_file = sys.argv[1]
    n_rollout = int(sys.argv[2])
    env_params = load_yaml(tmp_file+'/env_params')
    ns = NS(**env_params, controller=tmp_file)
    transitions = ns.run_with_controller(n_rollout)
    if ns.comm.rank == 0:
        transitions_ = pickle.dumps(transitions)
        sys.stdout.buffer.write(transitions_)
        sys.stdout.flush()
