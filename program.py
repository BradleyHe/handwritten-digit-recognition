import numpy

from tkinter import *
from keras.models import load_model

def update():
	global grid_updated
	row, col = 1, 1
	x = main.winfo_pointerx() - main.winfo_rootx()
	y = main.winfo_pointery() - main.winfo_rooty()

	# if mouse is clicked in the grid
	if button_pressed and offset+cell_size < x < offset+(cell_num-1)*cell_size and offset+cell_size < y < offset+(cell_num-1)*cell_size:
		col = (int)((x-offset)/cell_size)
		row = (int)((y-offset)/cell_size)

		# draw a 3x3 square around cursor
		grid[col-1][row-1] += 64
		grid[col-1][row+1] += 64
		grid[col+1][row+1] += 64
		grid[col+1][row-1] += 64
 
		grid[col][row-1] += 128
		grid[col-1][row] += 128
		grid[col][row+1] += 128
		grid[col+1][row] += 128

		grid[col][row] += 255

		grid_updated = True

	# draw grid and update prediction
	if grid_updated:
		for x in range(-1, 2):
			for y in range(-1, 2):
				draw_rec(row+y, col+x)

		# process grid
		# for some reason the mnist training data from keras.datasets is flipped across the x axis and rotated 90 degrees
		# this adjusts the grid back to that orientation 
		adjust_grid = [a for a in grid[::-1]]
		adjust_grid = zip(*adjust_grid[::-1])

		npgrid = numpy.array([[[[min(x, 255)] for x in row] for row in adjust_grid]])
		npgrid = npgrid.reshape(npgrid.shape[0], 28, 28, 1)
		npgrid = npgrid.astype('float32')
		npgrid /= 255

		pred = model.predict(npgrid)[0]
		pred_str = ['{p:.3%}'.format(p=float(i)) for i in pred] 
		pred_num = 0
		curr_pred = 0

		c.delete('text')
		for x in range(0, 10):
			if pred[x] > pred_num: pred_num = pred[x]; curr_pred = x
			c.create_text(offset*2+cell_size*cell_num+175, 100+40*x, font='Verdana 15', text='{}: {} confidence'.format(x, pred_str[x]), tag='text')
		c.create_text(offset*2+cell_size*cell_num+175, 550, font='Verdana 15 bold', text='Current prediction: {}'.format(curr_pred), tag='text')

		grid_updated = False
	main.after(30, update)

def draw_rec(row, col):
	c.create_rectangle(offset+col*cell_size, 
				offset+row*cell_size, 
				offset+(col+1)*cell_size, 
				offset+(row+1)*cell_size, 
				fill='#%02x%02x%02x' % (255-min(grid[col][row], 255), 255-min(grid[col][row], 255), 255-min(grid[col][row], 255)),
				outline='',
				tag='grid_square')
	c.tag_lower('grid_square')

def callback(event):
	global button_pressed
	button_pressed = not button_pressed

def clear():
	global grid_updated

	for x in range(0, cell_num):
		for y in range(0, cell_num):
			grid[x][y] = 0

	grid_updated = True
	c.delete('grid_square')

w, h = 1100, 700
button_pressed = False
grid_updated = True
grid = []
model = load_model('nnmodels2conv/model-17-0.99.hdf5')

main = Tk()
main.title('Handwritten Digit Recognition')
c = Canvas(main, width=w, height=h)
c.config(bg='white')
c.bind('<Button-1>', callback)
c.bind('<ButtonRelease-1>', callback)

#####################################################################################

# create 28x28 grid
cell_num = 28
cell_size = 22
offset = (int)((h-cell_num*cell_size)/2)

# draw a grid
# horizontal lines
# for i in range(offset, offset+cell_size*(cell_num+1), cell_size):
# 	c.create_line([(offset, i), (offset+cell_num*cell_size, i)], tag='grid_line')

# vertical lines
# for i in range(offset, offset+cell_size*(cell_num+1), cell_size):
# 	c.create_line([(i, offset), (i, offset+cell_num*cell_size)], tag='grid_line')

# draw outer boundaries
c.create_line([(offset, offset), (offset, offset+cell_num*cell_size)], tag='grid_line')
c.create_line([(offset, offset), (offset+cell_num*cell_size, offset)], tag='grid_line')
c.create_line([(offset+cell_num*cell_size, offset), (offset+cell_num*cell_size, offset+cell_num*cell_size)], tag='grid_line')
c.create_line([(offset, offset+cell_num*cell_size), (offset+cell_num*cell_size, offset+cell_num*cell_size)], tag='grid_line')

# initialize grid variable
for x in range(0, cell_num):
	grid.append([0] * cell_num)

#####################################################################################

c.pack()

b = Button(main, text='Clear Grid', command=clear)
b.place(x=(w/4)+20, y=h-(int)(offset*3/4))

main.after(10, update)
main.mainloop()