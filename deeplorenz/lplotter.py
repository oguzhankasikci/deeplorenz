
import numpy as np
import matplotlib.pyplot as plt

class Plotter:

    def __init__(self, parent):
        """Initialize the Plotter with a Lorenz system instance."""        
        self._solution = parent.solve()
        self.t = parent.t  # Time grid from the Lorenz system

    # ===  3d plot of Lorenz Equations === 
    def plot_3d(self):
        X, Y, Z = self._solution.T     
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
        X, Y, Z = self._solution.T

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
        v1, v2 = var
        comps = {'x': 0, 'y': 1, 'z': 2}
        x1, x2 = comps.get(v1), comps.get(v2)

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