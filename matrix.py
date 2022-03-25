class Matrix:

    def __init__(self, list2D):
        self.setMatrix(list2D)

    def setMatrix(self, newMatrix):
        print(len(newMatrix))
        self.matrix = newMatrix
        self.colSize = len(newMatrix[0])
        self.rowSize = len(newMatrix)

    def vector(self, newVector):
        matrixV = []
        for i in newVector:
            vector = [1]
            vector[0] = i
            matrixV.append(vector)
        self.matrix = matrixV
        self.colSize = 1
        self.rowSize = len(newVector)

    def printMatrix(self):
        for i in self.matrix:
            print(i);

    def setElement(self, pos, value):
        self.matrix[pos[0]][pos[1]] = value

    def at(self, pos):
        return self.matrix[pos[1]][pos[0]]

    def getCol(self, colPos):
        return self.matrix[colPos]

    def getRow(self, rowPos):
        row = []
        for i in self.matrix:
            row.append(i[rowPos])
        return row

    def getColSize(self):
        return self.colSize

    def getRowSize(self):
        return self.rowSize

    def addElement(self, pos, value):
        self.matrix[pos[0]][pos[1]] += value

    def add(self, matrix2):
        if self.colSize == matrix2.getColSize() and self.rowSize == matrix2.getRowSize():
            for j in range(len(self.matrix)):
                for i in range(len(self.matrix[i])):
                    self.addElement([i, j], matrix2.at([i, j]))

    def subtract(self, matrix2):
        if self.colSize == matrix2.getColSize() and self.rowSize == matrix2.getRowSize():
            for j in range(len(self.matrix)):
                for i in range(len(self.matrix[i])):
                    self.addElement([i, j], -matrix2.at([i, j]))

    def multiply(self, matrix2):
        nMatrix = [[]]
        if self.colSize == matrix2.getRowSize():
            colSizeN = matrix2.getColSize()
            rowSizeN = self.rowSize
            matrix = [[0 for x in range(rowSizeN)] for y in range(colSizeN)]
            for j in range(colSizeN):
                for i in range(rowSizeN):
                    # add together the multiplications of the col and row
                    sum = 0
                    for k in range(len(matrix2.getCol(i))):
                        sum += matrix2.getCol(i)[k] * self.getRow(j)[k]
                    matrix[i][j] = sum
            self.setMatrix(matrix)

        elif self.rowSize == matrix2.getColSize():
            colSizeN = self.colSize
            rowSizeN = matrix2.getRowSize()
            matrix = [[0 for x in range(rowSizeN)] for y in range(colSizeN)]
            for j in range(colSizeN):
                for i in range(rowSizeN):
                    # add together the multiplications of the col and row
                    sum = 0
                    for k in range(len(matrix2.getRow(i))):
                        sum += matrix2.getRow(i)[k] * self.getCol(j)[k]
                    matrix[i][j] = sum
            self.setMatrix(matrix)
        elif matrix2.getColSize() == 1 and matrix2.getRowSize() == 1:
            for j in range(len(self.matrix)):
                for i in range(len(self.matrix[i])):
                    self.setElement([i, j], self.at([i, j]) * matrix2.at([0, 0]))

    def transpose(self):
        matrixN = [[0 for x in range(self.colSize)] for y in range(self.rowSize)]
        for j in range(len(self.matrix)):
            for i in range(len(self.matrix[j])):
                matrixN[j][i] = self.at([i, j])
        self.setMatrix(matrixN)


vector = [2, 3, 4]
vector2 = [[2, 3, 4]]
matrix = [[1, 1, 1, 2]]
m = Matrix(matrix)
v = Matrix(vector2)
v.transpose()
m.vector(vector)
m.add(v)
m.printMatrix()


