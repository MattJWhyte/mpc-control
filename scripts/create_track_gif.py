
import imageio
import os

with imageio.get_writer('track_walk.gif', mode='I') as writer:
    N = 0
    while os.path.exists("../track/{}.png".format(N)):
        N += 1
    N = N-1
    for i in range(N):
        image = imageio.imread("../track/{}.png".format(i))
        writer.append_data(image)
        i += 1