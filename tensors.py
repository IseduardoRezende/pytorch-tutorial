import torch
import random
from utils import Utils

def tensor_ones():
    a = torch.ones(3)
    print(a)
    print(f"f value: {float(a[1])}")

    a[2] = 2.0
    print(a)

def tensor_1d_points():
    points = torch.zeros(6)
    print(points)

    for i in range(len(points)):
        points[i] = float(random.randint(1, 5))  
    print(points)

    points = torch.tensor([4.0, 1.0, 5.0, 3.0, 2.0, 1.0])
    print(points)

    print(f"Getting the first point: ({float(points[0])},{float(points[1])})")
    
def tensor_2d_points():
    points = torch.zeros(3, 2)
    print(points)
   
    points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
    print(points)
    Utils.get_shape(points)

    print(points[0, 1])
    print(points[0])

def indexing_notation_tensor():
    tensor = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
    print(tensor) # All items - 3x2 (rows, columns)
    print(tensor[1:]) # From row 1 to end
    print(tensor[1:, :]) # From row 1 to end
    print(tensor[1:, 0]) # From row 1 to end, column 0 (Only the first elements of each line)
    print(tensor[None]) # Add dimension - 1x3x2 (layer, rows columns)

def indexing_notation_list():
    a = list(range(6))    
    print(a[:]) # [0, 1, 2, 3, 4, 5]
    print(a[1:4]) # [1, 2, 3]
    print(a[1:]) # [1, 2, 3, 4, 5]
    print(a[:4]) # [0, 1, 2, 3]
    print(a[:-1]) # [0, 1, 2, 3, 4]
    print(a[1:4:2]) # [1, 3]

indexing_notation_tensor()