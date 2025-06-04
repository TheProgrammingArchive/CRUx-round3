import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use('fivethirtyeight')

def animate(i):
    f = open('dataw.txt', 'r')
    data = f.readlines()[-1]
    y1 = data.split(',')[1]
    y2 = data.split(',')[2]

    print(data)

    plt.cla()
    plt.plot(data.split(',')[0], y1, label='Acc')
    plt.plot(data.split(',')[0], y2, label='Loss')

    plt.legend(loc='upper left')
    plt.tight_layout()

ani = animation.FuncAnimation(plt.gcf(), animate, interval=10000, save_count=10000)
plt.tight_layout()
plt.show()