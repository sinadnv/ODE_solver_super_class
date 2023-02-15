import numpy as np
import matplotlib.pyplot as plt


class ODE_Solver:
    def __init__(self, f, u0=.1, T=3, N=100):
        self.f, self.u0, self.T, self.N = f, u0, T, N
        self.dt = self.T/self.N

    def advance(self):
        raise NotImplementedError

    def solve(self):
        self.t = np.linspace(0, self.T, self.N)
        self.u = np.zeros(self.N)
        self.u[0] = self.u0
        for i in range(self.N-1):
            self.i = i
            self.u[i+1] = self.u[i]+self.advance()
        return self.t, self.u


class Forward_Euler(ODE_Solver):
    def advance(self):
        u_adv = self.dt*self.f(self.u[self.i],self.t[self.i])
        return u_adv


class Explicit_MidPoint(ODE_Solver):
    def advance(self):
        h = self.dt/2
        k1 = self.f(self.u[self.i],self.t[self.i])
        u_adv = self.dt*self.f(self.t[self.i]+h, self.u[self.i]+h*k1)
        return u_adv


class RK4(ODE_Solver):
    def advance(self):
        h, i = self.dt, self.i
        k1 = self.f(self.u[i], self.t[i])
        k2 = self.f(self.u[i]+h*k1/2, self.t[i]+h/2)
        k3 = self.f(self.u[i]+h*k2/2, self.t[i]+h/2)
        k4 = self.f(self.u[i]+h*k3, self.t[i]+h)
        u_adv = h*(k1+2*k2+2*k3+k4)/6
        return u_adv


def f(u,t):
    return u


eq1 = Forward_Euler(f)
t1, u1 = eq1.solve()
plt.plot(t1,u1)

eq2 = Explicit_MidPoint(f)
t2, u2 = eq2.solve()
plt.plot(t2,u2)

eq3 = RK4(f)
t3, u3 = eq3.solve()
plt.plot(t3,u3)

# Exact solution
u0, T, N = .1, 3, 100
t4 = np.linspace(0, T, N)
u4 = u0*np.exp(t4)
plt.plot(t4,u4,'--')

plt.legend(['eq1', 'eq2', 'eq3', 'Exact'])
plt.grid()
plt.show()

