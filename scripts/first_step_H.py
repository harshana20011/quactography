"""
Écrire un code qui génère un graphe avec N nodes et M edges : 
Input: fodf.nii.gz, fodf_threshold, M                       : 
Output: graph (matrice d'adjacence)                         : 
Poids sur les edge. Plus c’est fort, plus c’est connecté?   : 


Écrire un script qui génère l’Hamiltonien correspondent à n’importe quel graph :
Input: graph (matrice d'adjacence)                                             :
Output: Hamiltonien                                                            : 

 Tweaking / tuning de l’Hamiltonien 


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

percole = False
isRepeat = True
maxframes = 1
frame = 0
# setting up the values for the grid
solidity = 0.9
N = 200
Nmax = 1
updateInterval = 10
convolutionFactor = 2


def randomGrid(N):
#returns a grid of NxN random values
    return np.random.choice([0, 100], N * N).reshape(N, N)  # type: ignore


grid = randomGrid(N)
newGrid = grid
medium_done = False


def update(frameNum, img, grid, N, frame, medium_done):
    # copy grid since we require 8 neighbors
    # for calculation and we go line by line

    if percole == True:
        updateInterval = -1
    elif frameNum < maxframes:
        print(frameNum)
        # frameNum = frameNum + 1
        # if frameNum >= maxframes:
        #     medium_done = True
        mean()
    elif frameNum == maxframes:
        round()
    elif percole == False:
        percolation()
    img.set_data(newGrid)
    grid[:] = newGrid[:]

    return img


def main():
    if percole == False:
        fig, ax = plt.subplots()
        img = ax.imshow(grid, interpolation="nearest")
        ani = animation.FuncAnimation(
            fig,
            update,
            fargs=(img, grid, N, frame, medium_done),
            frames=400,
            interval=updateInterval,
            save_count=50,
            repeat=False,
        )
        ani = plt.show()


def mean():
    for i in range(N):
        for j in range(N):

            # compute 8-neighbor sum
            # using toroidal boundary conditions - x and y wrap around
            # so that the simulation takes place on a toroidal surface.
            total = (
                grid[i, (j - 1) % N]
                + grid[i, (j + 1) % N]
                + grid[(i - 1) % N, j]
                + grid[(i + 1) % N, j]
                + grid[(i - 1) % N, (j - 1) % N]
                + grid[(i - 1) % N, (j + 1) % N]
                + grid[(i + 1) % N, (j - 1) % N]
                + grid[(i + 1) % N, (j + 1) % N]
            )

            newGrid[i, j] = grid[i, j] + (total - 400) / convolutionFactor
            if newGrid[i, j] >= 100:
                newGrid[i, j] = 100
            elif newGrid[i, j] <= 0:
                newGrid[i, j] = 0


def round():
    for i in range(0, N):
        for j in range(0, N):
            if grid[i, j] >= 30:
                grid[i, j] = 100
            else:
                grid[i, j] = 0


def percolation():
    for i in range(0, N):
        # premiere ligne 0
        grid[1, i] = 50

        # premiere et derniere colonne zero
        # grid[i][0] == 0
        # grid[i][N-1] == 0
    count = 0
    for a in range(1, N - 1):
        i = N - a - 1
        for b in range(1, N - 1):
            j = N - b - 1
            if grid[i, j] >= 90:
                if grid[i + 1, j] == 50:
                    grid[i, j] = 50
                    count = count + 1
                    Nmax = max(N, i)
                elif grid[i - 1, j] == 50:
                    grid[i, j] = 50
                    count = count + 1
                    Nmax = max(N, i)
                elif grid[i, j + 1] == 50:
                    grid[i, j] = 50
                    count = count + 1
                    Nmax = max(N, i)
                elif grid[i, j - 1] == 50:
                    grid[i, j] = 50
                    count = count + 1
                    Nmax = max(N, i)


main()

if Nmax == 99:
    percole = True
    print("Le systeme percole")
else:
    print("Le systeme ne percole pas")
"""
