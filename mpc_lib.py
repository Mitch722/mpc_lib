import osqp
import numpy as np
import scipy.sparse as sparse
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MpcSys:

    def __init__(self, A, B, C, N=10, D=0):

        self.Ad = sparse.csc_matrix(A)
        self.Bd = sparse.csc_matrix(B)
        self.Cd = sparse.csc_matrix(C)
        self.Dd = sparse.csc_matrix(D)

        self.N = N

        [self.nx, self.nu] = self.Bd.shape

        self.P = 0
        self.q = 0

        self.Q = 0
        self.QN = 0

        self.A = 0
        self.l = 0
        self.u = 0

    def make_obj_func(self, Q, R, xr):

        QN = sparse.csr_matrix.copy(Q)
        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(self.N), Q), QN,
                               sparse.kron(sparse.eye(self.N), R)]).tocsc()
        # - linear objective
        q = np.hstack([np.kron(np.ones(self.N), -Q.dot(xr)), -QN.dot(xr),
                       np.zeros(self.N * self.nu)])

        self.Q = Q
        self.QN = QN
        self.P = P
        self.q = q
        # return [self.P, self.q]

    def make_costra(self, xmin, xmax, umin, umax, x0):

        Ax = sparse.kron(sparse.eye(self.N + 1), -sparse.eye(self.nx)) + sparse.kron(sparse.eye(self.N + 1, k=-1), self.Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), self.Bd)
        Aeq = sparse.hstack([Ax, Bu])

        leq = np.hstack([-x0, np.zeros(self.N * self.nx)])
        ueq = leq

        Aineq = sparse.eye((self.N + 1) * self.nx + self.N * self.nu)
        lineq = np.hstack([np.kron(np.ones(self.N + 1), xmin), np.kron(np.ones(self.N), umin)])
        uineq = np.hstack([np.kron(np.ones(self.N + 1), xmax), np.kron(np.ones(self.N), umax)])
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq]).tocsc()
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        self.A = A
        self.l = l
        self.u = u
        # return [A, l, u]


class OSQP_MPC:

    def __init__(self, mpc_sys_class, warm_start_bool=True):
        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.prob.setup(mpc_sys_class.P, mpc_sys_class.q, mpc_sys_class.A,
                        mpc_sys_class.l, mpc_sys_class.u, warm_start=warm_start_bool)

        self.mpc_problem = mpc_sys_class

    def solve(self):
        res = self.prob.solve()

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        # Apply first control input to the plant
        ctrl = res.x[-self.mpc_problem.N * self.mpc_problem.nu:-(self.mpc_problem.N - 1) * self.mpc_problem.nu]

        return [ctrl, res]

    def update(self, x0, xr):
        self.mpc_problem.l[:self.mpc_problem.nx] = -x0
        self.mpc_problem.u[:self.mpc_problem.nx] = -x0

        self.mpc_problem.q = np.hstack([np.kron(np.ones(self.mpc_problem.N), -self.mpc_problem.Q.dot(xr)),
                                        -self.mpc_problem.QN.dot(xr), np.zeros(self.mpc_problem.N * self.mpc_problem.nu)
                                        ])

        self.prob.update(q=self.mpc_problem.q, l=self.mpc_problem.l, u=self.mpc_problem.u)


if __name__ == "__main__":
    # x = [, , height, x_pos, y_pos, , , , , , ]
    Ad = np.array([
        [1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0.],
        [0.0488, 0., 0., 1., 0., 0., 0.0016, 0., 0., 0.0992, 0., 0.],
        [0., -0.0488, 0., 0., 1., 0., 0., -0.0016, 0., 0., 0.0992, 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.0992],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0.9734, 0., 0., 0., 0., 0., 0.0488, 0., 0., 0.9846, 0., 0.],
        [0., -0.9734, 0., 0., 0., 0., 0., -0.0488, 0., 0., 0.9846, 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9846]
    ])

    Bd = np.array([
        [0., -0.0726, 0., 0.0726],
        [-0.0726, 0., 0.0726, 0.],
        [-0.0152, 0.0152, -0.0152, 0.0152],
        [-0., -0.0006, -0., 0.0006],
        [0.0006, 0., -0.0006, 0.0000],
        [0.0106, 0.0106, 0.0106, 0.0106],
        [0, -1.4512, 0., 1.4512],
        [-1.4512, 0., 1.4512, 0.],
        [-0.3049, 0.3049, -0.3049, 0.3049],
        [-0., -0.0236, 0., 0.0236],
        [0.0236, 0., -0.0236, 0.],
        [0.2107, 0.2107, 0.2107, 0.2107]
    ])

    # Constraints
    u0 = 10.5916
    umin = np.array([9.6, 9.6, 9.6, 9.6]) - u0
    umax = np.array([13., 13., 13., 13.]) - u0
    xmin = np.array([-np.pi/6,-np.pi/6,-np.inf,-np.inf,-np.inf,-1.,
                 -np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf])
    xmax = np.array([ np.pi/6, np.pi/6, np.inf, np.inf, np.inf, np.inf,
                  np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

    # Objective function
    Q = sparse.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
    R = 0.1*sparse.eye(4)

    # Initial and reference states
    x0 = np.zeros(12)
    xr = np.array([0.,0.,2.0,0.,0.,0.,0.,0.,0.,0.,0.,0.])

    # Prediction horizon
    N = 10

    # Simulate in closed loop
    nsim = 100

    quad_cptr_sys = MpcSys(A=Ad, B=Bd, C=np.eye(Ad.shape[0]), N=N)

    quad_cptr_sys.make_obj_func(Q=Q, R=R, xr=xr)
    quad_cptr_sys.make_costra(xmin=xmin, xmax=xmax, umin=umin, umax=umax, x0=x0)

    ctrl_solver = OSQP_MPC(quad_cptr_sys)

    x_r = np.array([0.0, 0., 1.0, 1.0, 0.5, 0., 0., 0., 0., 0., 0., 0.])
    x_0 = np.zeros(12)

    Ts = nsim / 1.5

    rad = 1

    x_states = np.zeros((quad_cptr_sys.nx, 1))

    for k in range(nsim):

        [ctrl, res] = ctrl_solver.solve()
        x_0 = Ad.dot(x_0) + Bd.dot(ctrl) + 0.01 * np.random.randn(*x_0.shape)
        x_states = np.append(x_states, x_0.reshape(12, 1), axis=1)

        x = rad * np.cos(2 * np.pi * k / Ts)
        y = rad * np.sin(2 * np.pi * k / Ts)

        x_r = np.array([0.0, 0., 1.5, x, y, 0., 0., 0., 0., 0., 0., 0.])

        ctrl_solver.update(x0=x_0, xr=x_r)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x_states[3, :], x_states[4, :], x_states[2, :], label='parametric curve')

    plt.show()
