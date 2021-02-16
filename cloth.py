import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


CONST_G = 9.81


class RandomPointPool:
    def __init__(self, low, hi, dim=2):
        assert low < hi, "Low must be less than hi"
        self.low = low
        self.hi = hi
        self.dim = dim
        self.irand = 0
        self.nrand = 1024 * 100
        self.shape = (self.nrand, self.dim)
        self.vals = np.random.randint(self.low, self.hi, size=self.shape)

    def __call__(self):
        if self.irand >= self.nrand:
            self.irand = 0
            self.vals = np.random.randint(self.low, self.hi, size=self.shape)
        pt = self.vals[self.irand]
        self.irand += 1
        return tuple(pt)


class RandomPool:
    def __init__(self):
        self.irand = 0
        self.nrand = 1024 * 10
        self.vals = np.random.rand(self.nrand)

    def __call__(self):
        if self.irand >= self.nrand:
            self.irand = 0
            self.vals = np.random.rand(self.nrand)
        v = self.vals[self.irand]
        self.irand += 1
        return v


class RandomIntPool:
    def __init__(self, low, hi):
        self.low = low
        self.hi = hi
        self.irand = 0
        self.nrand = 1024 * 10
        self.vals = np.random.randint(self.low, self.hi, size=self.nrand)

    def __call__(self):
        if self.irand >= self.nrand:
            self.irand = 0
            self.vals = np.random.randint(self.low, self.hi, size=self.nrand)
        v = self.vals[self.irand]
        self.irand += 1
        return v


class RandomNormalPool:
    def __init__(self):
        self.irand = 0
        self.nrand = 1024 * 10
        self.vals = np.random.randn(self.nrand)

    def __call__(self):
        if self.irand >= self.nrand:
            self.irand = 0
            self.vals = np.random.randn(self.nrand)
        v = self.vals[self.irand]
        self.irand += 1
        return v


class Cloth:
    def __init__(self, L, kT, k, m, l, sig):
        self.L = L
        self.N = L * L
        self.kT = kT
        self.k = k
        self.m = m
        self.l = l
        self.sig = sig

        self.lattice = np.zeros((L, L, 3))
        x, y = np.meshgrid(range(self.L), range(self.L))
        self.lattice[..., 0] = x
        self.lattice[..., 1] = y
        self.lattice
        self.E = 0
        self.evec = []
        self.mcs = 0

        nx = [1, -1, 0, 0]
        ny = [0, 0, 1, -1]
        self.dirs = np.array(list(zip(nx, ny)))
        self.rand_pt = RandomPointPool(0, self.L)
        self.randint = RandomIntPool(0, 4)
        self.rand = RandomPool()
        self.randn = RandomNormalPool()
        self.perturbations = [
            lambda: (self.randn() * self.sig, 0, 0),
            lambda: (0, self.randn() * self.sig, 0),
            lambda: (0, 0, self.randn() * 2 * self.sig),
            lambda: (0, 0, self.randn() * 2 * self.sig),
        ]
        self.corners = frozenset(
            [
                (0, 0),
                (self.L - 1, 0),
                (0, self.L - 1),
                (self.L - 1, self.L - 1),
            ]
        )
        outside = set()
        for i in [-1, self.L]:
            for j in range(0, self.L):
                outside.add((i, j))
        for j in [-1, self.L]:
            for i in range(0, self.L):
                outside.add((i, j))
        self.outside = frozenset(outside)

    def step(self):
        for i in range(self.N):
            pt = self.rand_pt()
            if pt in self.corners:
                continue
            dr, dE = self.perturb(pt)
            if dE < 0 or self.rand() < np.exp(-dE / self.kT):
                self.lattice[pt] += dr
                self.E += dE
        self.evec.append(self.E)
        self.mcs += 1

    def get_delta(self, pt, dr):
        nn = self.dirs + pt
        mask = [tuple(n) not in self.outside for n in nn]
        nn = nn[mask]
        delta = 0
        # Unperturbed r
        r = self.lattice[pt]
        # Perturbed r, r prime
        rp = self.lattice[pt] + dr
        for n in nn:
            rn = self.lattice[n[0], n[1]]
            tmp = rp - rn
            tmp *= tmp
            # Manual sum gives a large (20%) speedup. This is likely because
            # the overhead of np.sum is large compared to summing a vec with
            # size 3.
            dp = np.sqrt(tmp[0] + tmp[1] + tmp[2])
            tmp = r - rn
            tmp *= tmp
            d = np.sqrt(tmp[0] + tmp[1] + tmp[2])
            perturbed = dp - self.l
            unperturbed = d - self.l
            delta += (perturbed * perturbed) - (unperturbed * unperturbed)
        delta *= 0.5 * self.k
        delta += self.m * CONST_G * dr[-1]
        return delta

    def get_perturbation_no_branch(self):
        dr = self.perturbations[self.randint()]()
        return dr

    def get_perturbation(self):
        p = self.rand()
        dx = 0
        dy = 0
        dz = 0
        if p < 0.25:
            dx = self.randn() * self.sig
        elif p < 0.5:
            dy = self.randn() * self.sig
        else:
            dz = self.randn() * 2 * self.sig
        return (dx, dy, dz)

    def perturb(self, pt):
        dr = self.get_perturbation_no_branch()
        dE = self.get_delta(pt, dr)
        return dr, dE


def run_sim(sim, max_iter):
    i = 0
    while i < max_iter:
        sim.step()
        i += 1


def plot_cloth(cloth):
    x = cloth.lattice[..., 0]
    y = cloth.lattice[..., 1]
    z = cloth.lattice[..., 2]
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.viridis)
    plt.figure()
    plt.plot(cloth.evec)
    plt.show()


if __name__ == "__main__":
    L = 16
    k = 50
    l = 1
    m = 0.04
    sigma = 0.25 * l
    kT = 1e-3
    sim = Cloth(L, kT, k, m, l, sigma)
    run_sim(sim, 20_000)
    plot_cloth(sim)
