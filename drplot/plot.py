import numpy as np
import matplotlib.pyplot as plt

def plot_line(w,b=0,**kwargs):
    # plot the set of points v = [x,y] in R^2 such that
    # w@v = b
    # w[0]*x + w[1]*y= b
    # y = x*(-w[0]/w[1])+b/w[1]
    ax = plt.gca()
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    xs = np.linspace(xlims[0],xlims[1])
    ys = -xs*(w[0]/w[1])+b/w[1]
    ax.plot(xs,ys,**kwargs)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    return None


def dec_bdry_lin_3c_2d(W, linestyle = ':', c = 'k',**kwargs):
    # decision boundaries of linear classifier with 3 classes in 2 dimensions
    plot_line(W[:,0] - W[:,1], linestyle = linestyle, c = c,**kwargs)
    plot_line(W[:,0] - W[:,2], linestyle = linestyle, c = c,**kwargs)
    plot_line(W[:,1] - W[:,2], linestyle = linestyle, c = c,**kwargs)
    return None

def get_grid_from_plot(n_grid=50):
    # creates a square grid of size n_grid by n_grid 
    # that covers the current plot area
    # returns the grid points as a (n_grid^2, 2) numpy matrix
    ax = plt.gca()
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    x_grid_points = np.linspace(xlims[0],xlims[1],n_grid)
    y_grid_points = np.linspace(ylims[0],ylims[1],n_grid)
    x1_test, x2_test =  np.meshgrid(x_grid_points,y_grid_points)

    x1 = x1_test.flatten()
    x2 = x2_test.flatten()

    x_grid = np.array([x1,x2]).T
    return x_grid


def plot_dec_regi_from_y_grid(y_grid):
    # input:
    #     y_grid = a numpy vector of length n_grid^2 taking values in {0,1,...,n_classes-1}
    #     lims = [xmin, xmax, ymin, ymax], e.g., the output of 'get_plot_lims'
    
    # plot the decision regions of y_grid, 
    # it is expected that y_grid is a list of length n_grid^2
    ax = plt.gca()
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    lims = xlims+ylims
    
    n_grid = int(np.sqrt(y_grid.shape[0]))
    Z = np.flipud(y_grid.reshape((n_grid,n_grid)))
    plt.imshow(Z,extent=lims, alpha=0.2, aspect='auto')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    return None

def plot_dec_regi(W,n_grid = 50):
    grid = get_grid_from_plot(n_grid = n_grid)
    plot_dec_regi_from_y_grid(np.argmax(grid@W,axis=1))
    return None
