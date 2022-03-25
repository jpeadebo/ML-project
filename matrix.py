class Matrix:

    def __init__(self, list2D):
        self.setMatrix(list2D)

    def setMatrix(self, newMatrix):
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
            print(i)

    def setElement(self, pos, value):
        self.matrix[pos[0]][pos[1]] = value

    def at(self, pos):
        return self.matrix[pos[0]][pos[1]]

    def getRow(self, rowPos):
        return self.matrix[rowPos]

    def getCol(self, colPos):
        col = []
        for i in self.matrix:
            col.append(i[colPos])
        return col

    def getColSize(self):
        return self.colSize

    def getRowSize(self):
        return self.rowSize

    def addElement(self, pos, value):
        self.matrix[pos[0]][pos[1]] += value

    def add(self, matrix2):
        if self.colSize == matrix2.getColSize() and self.rowSize == matrix2.getRowSize():
            for r in range(len(self.matrix)):
                for c in range(len(self.matrix[r])):
                    self.addElement([r,c], matrix2.at([r,c]))

    def subtract(self, matrix2):
        if self.colSize == matrix2.getColSize() and self.rowSize == matrix2.getRowSize():
            for r in range(len(self.matrix)):
                for c in range(len(self.matrix[r])):
                    self.addElement([r, c], -matrix2.at([r, c]))

    def multiply(self, matrix2):
        if self.colSize == matrix2.getRowSize():
            colSizeN = matrix2.getColSize()
            rowSizeN = self.rowSize
            matrix = [[0 for x in range(colSizeN)] for y in range(rowSizeN)]
            for r in range(rowSizeN):
                for c in range(colSizeN):
                    # add together the multiplications of the col and row
                    sum = 0
                    for k in range(len(matrix2.getCol(c))):
                        sum += matrix2.getCol(c)[k] * self.getRow(r)[k]
                    matrix[r][c] = sum
            self.setMatrix(matrix)

        elif self.rowSize == matrix2.getColSize():
            colSizeN = self.colSize
            rowSizeN = matrix2.getRowSize()
            matrix = [[0 for x in range(colSizeN)] for y in range(rowSizeN)]
            for r in range(rowSizeN):
                for c in range(colSizeN):
                    # add together the multiplications of the col and row
                    sum = 0
                    for k in range(len(matrix2.getRow(r))):
                        sum += matrix2.getRow(r)[k] * self.getCol(c)[k]
                    matrix[r][c] = sum
            self.setMatrix(matrix)
        elif matrix2.getColSize() == 1 and matrix2.getRowSize() == 1:
            for r in range(len(self.matrix)):
                for c in range(len(self.matrix[r])):
                    self.setElement([r, c], self.at([r, c]) * matrix2.at([r, c]))

    def transpose(self):
        matrixN = [[0 for x in range(self.rowSize)] for y in range(self.colSize)]
        for r in range(len(self.matrix)):
            for c in range(len(self.matrix[r])):
                matrixN[c][r] = self.at([r, c])
        self.setMatrix(matrixN)
