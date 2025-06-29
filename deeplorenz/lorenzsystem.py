import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class LorenzClass:
    def __init__(self, 
                params = (10.0, 8.0 / 3.0, 28.0),
                X0 = (0.0, 1.0, 20.0),
                t_span = (0.0, 50.0),
                n_steps  = 50_000
                ):
        """ Construct parameters and class variables"""
        # Lorenz system parameters
        self.sigma = float(params[0])
        self.beta = float(params[1])
        self.rho = float(params[2])

        # Initial state, (x,y,z) = (X0[0],X0[1],X0[2])
        self.X0 = np.asarray(X0, dtype=float)        
        
        # Time grid 
        self.t = np.linspace(float(t_span[0]), float(t_span[1]), int(n_steps), endpoint=True)
        self.dt = np.insert(np.diff(self.t), 0, self.t[1] - self.t[0])
        

        # Solution invalidate
        self._solution = None  # shape (3,n_steps)  


    ## Update class parameters, initial value and time interval ==============
    def set_parameters(self, params = None):
        """  Update Parameters if only those provided"""
        if params is not None:
            self.sigma = float(params[0])
            self.beta = float(params[1])
            self.rho = float(params[2])
        self._solution = None ## reset the solution

    def set_time_grid(self, t0, tf, n_steps):
        """ Redefine the time array t and dt."""
        self.t = np.linspace(float(t0), float(tf), int(n_steps), endpoint=True)
        self.dt = np.insert(np.diff(self.t), 0, self.t[1] - self.t[0])
        self._solution = None  ## reset the solution

    def set_initial_value(self, X0):
        self.X0 = np.asarray(X0, dtype=float) 
        self._solution = None ## reset the solution

    ## =========================================================

    # Dynamic Equations 
    def lorenz(self, vars, time=0):
        """ Lorenz vector field (dx/dt, dy/dt, dz/dt)."""
        x, y, z = vars
        a, b, c = self.sigma, self.beta, self.rho
        dx = a * (y - x)
        dy = x * (c - z) - y
        dz = x * y - b * z
        return np.array([dx, dy, dz])

    def solve(self, rtol=1e-12, atol_scale=1e-12):
        """ Integrate the ODE system and store the result as shape(n_steps,3)"""
        atol = atol_scale * np.ones_like(self.X0)
        self._solution = odeint(
            self.lorenz, self.X0, self.t, rtol=rtol, atol=atol
        )
        return self._solution
    
    def components(self):
        """ Provide the solution components as dictionary ('x': x_val, 'y': y_val, 'z': z_val))"""
        sol = self._solution
        return dict(x= sol[:,0],y= sol[:,1],z= sol[:,2]) 
    
    ######################### Plotting Part #################

    # ===  3d plot of Lorenz Equations === 
    def plot_3d(self):
        X, Y, Z = self.solve().T     
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(X, Y, Z, linewidth=1, label='Lorenz Attractor')
        ax.set_title("Lorenz Attractor")
        ax.set_xlabel(' X ')
        ax.set_ylabel(' Y ')
        ax.set_zlabel(' Z ')
        plt.savefig("lorenz3dxyz.png")
        ax.legend()
        plt.show() 

    # ===  Plot all component (X,Y,Z) as subplots  in one figure === 
    def plot_2d(self):
        X, Y, Z = self.solve().T

        fig, axs = plt.subplots(3, 1, figsize=(7, 6))  # figsize=(horizontal_length = 7,vertical_length = 5)           
      
        axs[0].plot(self.t,X, label='X data')
        axs[0].set_title("X variable in Lorenz Attractor  ")
        #ax[0].set_xlabel(' Time ')
        axs[0].set_ylabel(' X ',rotation=0)       
        #axs[0].legend()        

        axs[1].plot(self.t, Y, label='Y data')
        axs[1].set_title("Y variable in Lorenz Attractor")
        #ax[1].set_xlabel(' Time ')
        axs[1].set_ylabel(' Y ',rotation=0)       
        #axs[1].legend()

        axs[2].plot(self.t, Z, label='Z data')
        axs[2].set_title("Z variable in Lorenz Attractor")
        axs[2].set_xlabel(' Time ')
        axs[2].set_ylabel(' Z ',rotation=0)       
        #axs[2].legend()

        plt.savefig("lorenz2dxyz.png")
        fig.tight_layout()
        
        plt.show()
            
    # ===  Individual Components X,Y and Z of the Lorenz System plotting over time. === 
    def plotcomp2d(self,var):
        """Plot the 2D component of the Lorenz system for given variable x, y or z."""
        if self._solution is None:
            self.solve()

        # Find index of variable
        x = {'x': 0, 'y': 1, 'z': 2}.get(var)
        
        if x is None:
            raise ValueError("Variable must be one of 'x', 'y', or 'z'.")

        data = self._solution[:, x]

        plt.figure(figsize=(8, 3))
        plt.plot(self.t, data, label=f'{var.upper()} data')
        plt.title(f"{var.upper()} variable in Lorenz Attractor")
        plt.xlabel('Time')
        plt.ylabel(var.upper(),rotation=0)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"lorenz2d{var}.png")
        plt.show()   

    # === 2d PHASE PLOTS  ===
    def plot_phase2d(self,var):
        """Plot the 2D phase space of the Lorenz system for given components x1 and x2."""
        if self._solution is None:
            self.solve()
        
        # Find indexes of variables
        v1, v2 = var[0], var[1]
        x1 = {'x': 0, 'y': 1, 'z': 2}.get(v1)
        x2 = {'x': 0, 'y': 1, 'z': 2}.get(v2)

        if x1 is None or x2 is None:
            raise ValueError("Components must be one of 'x', 'y', or 'z'.")
        
        x1_data = self._solution[:, x1]
        x2_data = self._solution[:, x2]

        plt.figure(figsize=(8, 6))
        plt.plot(x1_data, x2_data, lw=0.8)
        plt.xlabel(v1.upper())
        plt.ylabel(v2.upper(),rotation=0)
        plt.title(f'Phase Plot: {v1.upper()} vs {v2.upper()}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    

# Example usage ---------------------------------------------------------
if __name__ == "__main__":
    
    lc = LorenzClass()
    lc.solve()          # integrates with defaults
    lc.plot_2d()        # 2‑D component plot
    lc.plot_3d()        # 3‑D trajectory  
    
    lc.plot_phase2d('xy')  # 2d phase plot for x and y)    
    lc.plotcomp2d('x')  # plot x component over time
    lc.plotcomp2d('y')  # plot y component over time
    lc.plotcomp2d('z')  # plot z component over time
    

    # Change initial conditions and re‑solve
    lc.set_initial_value(X0=(5.0, 5.0, 5.0))
    lc.set_parameters(params= (5.0, 6.0 / 3.0, 12.0),)
    lc.set_time_grid(0, 30, n_steps=30_000)

    lc.plot_2d()    
    lc.plot_3d()
    lc.plotcomp2d('z')  # plot z component over time
    
