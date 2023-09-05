import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Coupling:
    type: str
    dofs: np.ndarray
    value: float

class MultiMassOscillator:

    def __init__(self, m, d, k, u, couplings):

        m = m.flatten()
        d = d.flatten()
        k = k.flatten()
        
        self.dof = m.shape[0]

        self.M = self.assemble_mass(m)
        self.D = self.assemble_damping(d)
        self.K = self.assemble_stiffness(k)
        self.f = self.assemble_force(u, k)

        self.omegas = None
        self.x_hat = None

        self.nominal_m = m
        self.nominal_d = d
        self.nominal_k = k
        self.nominal_u = u

        self.nominal_couplings = couplings

        for coupling in couplings:
            self.insert_coupling(coupling)

    def assemble_mass(self, m):
        return np.diag(m)

    def assemble_damping(self, d):
        d_diag = d[1:] + d[:-1]
        d_off_diag = d[1:-1]
        D = np.diag(d_diag) - np.diag(d_off_diag, 1) - np.diag(d_off_diag, -1)
        return D

    def assemble_stiffness(self, k):
        k_diag = k[1:] + k[:-1]
        k_off_diag = k[1:-1]
        K = np.diag(k_diag) - np.diag(k_off_diag, -1) - np.diag(k_off_diag, 1) 
        return K

    def insert_coupling(self, coupling):
        dof_pair = coupling.dofs
        dof_idx = dof_pair - 1
        
        if coupling.type == 'stiff':
            self.K[dof_idx[0], dof_idx[1]] = - coupling.value
            self.K[dof_idx[1], dof_idx[0]] = - coupling.value
            self.K[dof_idx[0], dof_idx[0]] = self.K[dof_idx[0], dof_idx[0]] + coupling.value
            self.K[dof_idx[1], dof_idx[1]] = self.K[dof_idx[1], dof_idx[1]] + coupling.value
        
        if coupling.type == 'damp':
            self.D[dof_idx[0], dof_idx[1]] = - coupling.value
            self.D[dof_idx[1], dof_idx[0]] = - coupling.value
            self.D[dof_idx[0], dof_idx[0]] = self.D[dof_idx[0], dof_idx[0]] + coupling.value
            self.D[dof_idx[1], dof_idx[1]] = self.D[dof_idx[1], dof_idx[1]] + coupling.value
        
    def assemble_force(self, u, k):
        f_vec = np.zeros(self.dof, dtype=np.complex_)
        f_vec[0] = 0* 1j + u * k[0]
        return f_vec

    def assemble_system(self, omega):
        A = - np.square(omega) * self.M + 1j * omega * self.D + self.K
        return A 

    def frequency_response(self, omegas):

        self.omegas = omegas
        x_hat = np.zeros((1,omegas.size,self.dof), dtype= np.complex_)

        for idx, w in enumerate(omegas):
            A = self.assemble_system(w)
            x_hat[0,idx,:] = np.linalg.solve(A,self.f)

        self.x_hat = x_hat

        return x_hat


    def frequency_response_para_variation_matrix(self, omegas, m_sample, d_sample, k_sample):
        
        if not(m_sample.shape[0] == d_sample.shape[0] == m_sample.shape[0]):
            raise ValueError("Number of samples of m,d,k must be the same")

        self.omegas = omegas
        x_hat = np.zeros((m_sample.shape[0], omegas.size, self.dof), dtype= np.complex_)

        # Loop over the samples
        for i, (m, d, k) in enumerate(zip(m_sample, d_sample, k_sample)):   
            self.M = self.assemble_mass(m)
            self.D = self.assemble_damping(d)
            self.K = self.assemble_stiffness(k)

            for coupling in self.nominal_couplings:
                self.insert_coupling(coupling)

            for j, omega_i in enumerate(omegas):
                A = self.assemble_system(omega_i)
                x_hat[i,j,:] = np.linalg.solve(A,self.f)

        self.x_hat = x_hat

        return x_hat


    def frequency_response_para_variation_alpha(self, omegas, alpha):

        # Compute frequency response
        self.omegas = omegas
        x_hat = np.zeros((alpha.size, omegas.size, self.dof), dtype= np.complex_)

        for i, alpha_i in enumerate(alpha):
            
            # Parameter variation in the stiffness matrix
            k = self.nominal_k
            k[4] = 2 + 2 * alpha_i
            self.K = self.assemble_stiffness(k)
            k_1_4 = 1 + 2 * alpha_i
            coupling_1_4 = Coupling(type='stiff', dofs= np.array([1,4]), value=k_1_4)
            self.insert_coupling(coupling_1_4)

            # Parameter variation in the damping matrix
            d = self.nominal_d
            d[4] = alpha_i
            self.D = self.assemble_damping(d)

            for j, w in enumerate(omegas):
                A = self.assemble_system(w)
                x_hat[i,j,:] = np.linalg.solve(A,self.f)

        self.x_hat = x_hat

        return x_hat
    
    def vari_k(self, omegas, k_vec):

        # Compute frequency response
        self.omegas = omegas
        x_hat = np.zeros((k_vec.size, omegas.size, self.dof), dtype= np.complex_)

        for i, k_val in enumerate(k_vec):
            
            # Parameter variation in the stiffness matrix
            k = self.nominal_k
            k[0] = k_val
            k[1] = k_val
            k[2] = k_val
            k[3] = k_val
            k[4] = k_val
            self.K = self.assemble_stiffness(k)

            # Parameter variation in the damping matrix

            for j, w in enumerate(omegas):
                A = self.assemble_system(w)
                x_hat[i,j,:] = np.linalg.solve(A,self.f)

        self.x_hat = x_hat

        return x_hat
    
    def vari_d(self, omegas, d_vec):

        # Compute frequency response
        self.omegas = omegas
        x_hat = np.zeros((d_vec.size, omegas.size, self.dof), dtype= np.complex_)

        for i, d_val in enumerate(d_vec):
            
            # Parameter variation in the stiffness matrix
            d = self.nominal_d
            d[0] = d_val
            d[1] = d_val
            d[2] = d_val
            d[3] = d_val
            d[4] = d_val
            self.D = self.assemble_stiffness(d)

            # Parameter variation in the damping matrix

            for j, w in enumerate(omegas):
                A = self.assemble_system(w)
                x_hat[i,j,:] = np.linalg.solve(A,self.f)

        self.x_hat = x_hat

        return x_hat

    def plot_FRF(self, dof):

        if self.x_hat is not None:
            # plt.semilogx(self.omegas, 20 * np.log10(np.abs(self.x_hat[dof,:])).T) # log10 is usual measure in acoustics; NN could use that directly
            plt.semilogx(self.omegas, 10 * np.log(np.abs(self.x_hat[:,:,dof])).T) # log10 is usual measure in acoustics; NN could use that directly
            plt.grid()
            plt.xlabel('Angular frequency in rad/s')
            plt.ylabel('Amplitude in dB re 1m')
            plt.show()
        else:
            print("Please call compute_frequency_response before calling the plotting routine")


    # def compute_frequency_response(self, freqs, k = None, m = None, xi = None, F = None):
    #     """Computes freuency response function (FRF) for single mass oscillator

    #     Parameters
    #     ----------
    #     freqs : ndarray (m,)
    #         Vector containing the frequency steps
    #     k : ndarray (n,)
    #         Vector of stiffness parameters
    #     m : ndarray (n,)
    #         Vector of mass parameters
    #     xi : ndarray (n,)
    #         Vector of damping parameters
    #     F : ndarray (n,)
    #         Vector of load

    #     Returns
    #     -------
    #     ndarray (m,n)
    #         a ndarray of FRFs, where each column represents one FRF
    #     """