

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate(snapshots,name):
    import matplotlib.pyplot as plt

    import matplotlib.animation as animation

    fps = 60
    nSeconds = 2
    # snapshots = [np.random.rand(5, 5) for _ in range(nSeconds * fps)]

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(8, 8))

    a = snapshots[0]
    im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1,  cmap='gist_gray')

    def animate_func(i):
        if i % fps == 0:
            print('.', end='')

        im.set_array(snapshots[i])
        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=nSeconds * fps,
        interval=1000 / fps,  # in ms
    )

    anim.save(name + '.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

    print('Done!')

    plt.show()  # Not required, it seems!


animate([], "blabla")
