import matplotlib.pyplot as plt

fig = plt.figure()
plot = fig.add_subplot(111)

# create some curves
for i in range(4):
    plot.plot(
        [i*1,i*2,i*3,i*4],
        gid=i)

def on_plot_hover(event):
    for curve in plot.get_lines():
        if curve.contains(event)[0]:
            print "over %s" % curve.get_gid()

fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)           
plt.show()