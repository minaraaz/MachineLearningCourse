import numpy as np
from keras.datasets import mnist
from copy import copy, deepcopy
import matplotlib.pyplot as plt


def return_next(visited, stack, counter):
    # print "empty stack"
    dim = len(visited)
    for col in range(dim):
        for row in range(dim):
            if not visited[row][col]:
                stack.append((row, col))
                counter += 1
                return stack, counter

def CC_counter_four(matrix):
    # matrix = [[0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0]]
    visited = deepcopy(matrix)
    dim = len(matrix)
    for col in range(dim):
        for row in range(dim):
            if matrix[row][col] == 0:
                visited[row][col] = False
            else:
                visited[row][col] = True
    
    # print visited

    stack = []
    counter = 0

    while not all(v for row in visited for v in row):
        if not stack:
            stack, counter = return_next(visited, stack, counter)
            
        # print stack
        popped = stack.pop()
        visited[popped[0]][popped[1]] = True
        if popped[0] - 1 >= 0 and not visited[popped[0] - 1][popped[1]]:
            stack.append((popped[0] - 1, popped[1]))
        if popped[0] + 1 < dim and not visited[popped[0] + 1][popped[1]]:
            stack.append((popped[0] + 1, popped[1]))
        if popped[1] - 1 >= 0 and not visited[popped[0]][popped[1] - 1]:
            stack.append((popped[0], popped[1] - 1))
        if popped[1] + 1 < dim and not visited[popped[0]][popped[1] + 1]:
            stack.append((popped[0], popped[1] + 1))
        # print "now"
        # print stack
        # print "===================="
                
    return counter

def CC_counter_eight(matrix):
    # matrix = [[0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0]]
    visited = deepcopy(matrix)
    dim = len(matrix)
    for col in range(dim):
        for row in range(dim):
            if matrix[row][col] == 0:
                visited[row][col] = False
            else:
                visited[row][col] = True
    
    # print visited

    stack = []
    counter = 0

    while not all(v for row in visited for v in row):
        if not stack:
            stack, counter = return_next(visited, stack, counter)
            
        # print stack
        popped = stack.pop()
        visited[popped[0]][popped[1]] = True
        if popped[0] - 1 >= 0 and not visited[popped[0] - 1][popped[1]]:
            stack.append((popped[0] - 1, popped[1]))
        if popped[0] + 1 < dim and not visited[popped[0] + 1][popped[1]]:
            stack.append((popped[0] + 1, popped[1]))
        if popped[1] - 1 >= 0 and not visited[popped[0]][popped[1] - 1]:
            stack.append((popped[0], popped[1] - 1))
        if popped[1] + 1 < dim and not visited[popped[0]][popped[1] + 1]:
            stack.append((popped[0], popped[1] + 1))
        if popped[0] + 1 < dim and popped[1] + 1 < dim and not visited[popped[0] + 1][popped[1] + 1]:
            stack.append((popped[0] + 1,popped[1] + 1))
        if popped[0] - 1 >= 0 and popped[1] - 1 >= 0 and not visited[popped[0] - 1][popped[1] - 1]:
            stack.append((popped[0] - 1,popped[1] - 1))
        if popped[0] - 1 >= 0 and popped[1] + 1 < dim and not visited[popped[0] - 1][popped[1] + 1]:
            stack.append((popped[0] - 1,popped[1] + 1))
        if popped[0] + 1 < dim and popped[1] - 1 >= 0 and not visited[popped[0] + 1][popped[1] - 1]:
            stack.append((popped[0] + 1,popped[1] - 1))
        
        # print "now"
        # print stack
        # print "===================="
                
    return counter

    def Value_summation(matrix):
        return sum(map(sum, matrix))


def main():
    # data load
    (train_images_original, train_labels_original), (test_images_original, test_labels_original) = mnist.load_data()

    # data reshape and black and white
    train_images = train_images_original.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255.0

    test_images = test_images_original.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255.0

    print train_labels_original[:10]
    test_set = train_images_original[4]

    # test_set = [[0, 0, 0, 0, 0, 0, 0], 
    #             [0, 0, 0, 1, 0, 0, 0], 
    #             [0, 0, 1, 0, 1, 0 ,0],
    #             [0, 1, 1, 1, 1, 1, 0],
    #             [0, 0, 0, 0, 1, 0, 0],
    #             [0, 0, 0, 0, 1, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0]]
    # print test_set
    print CC_counter_eight(test_set)

    plt.figure(figsize=(1,1))
    plt.imshow(test_set)
    plt.grid(None)
    plt.show()


if __name__ == "__main__":
    main()
