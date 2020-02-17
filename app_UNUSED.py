import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from math import cos, sin

from env import Environment

class App(tk.Frame):
    def __init__(self, env: Environment, name="Game", window_size=900):
        self._master = tk.Tk()
        tk.Frame.__init__(self, self._master, relief=tk.SUNKEN, bd=4)
        self.master.title(name)
        # Create a container
        frame = tk.Frame(self._master)
        frame.pack()
        self.canvas = tk.Canvas(frame, width=window_size, height=window_size)
        self.canvas.pack()
        self.pack()
