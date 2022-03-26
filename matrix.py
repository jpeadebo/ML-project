def transpose(list):
    matrixN = [[0 for x in range(1)] for y in range(len(list))]
    for r in range(len(list)):
        for c in range(1):
            matrixN[r][c] = list[r]
    return matrixN


class Matrix:

    def __init__(self, mat):
        if not isinstance(mat[0], list):
            matrix = [mat]
            mat = matrix
        # [row][col]
        self.matrix = mat
        if self.getRowSize() == 1:
            print("transposing matrix since its a unrotated vector", mat)
            self.transpose()

    def printMatrix(self):
        for r in self.matrix:
            print(r)

    # pos = [row, col]
    def at(self, pos):
        return self.matrix[pos[0]][pos[1]]

    def setMatrix(self, mat):
        self.matrix = mat

    def setRow(self, pos, row):
        self.matrix[pos] = row

    def getRow(self, row):
        return self.matrix[row]

    def getRowSize(self):
        return len(self.matrix)

    def getCol(self, col):
        list = [row[col] for row in self.matrix]
        return transpose(list)

    def getColSize(self):
        return len(self.getRow(0)) if isinstance(self.getRow(0), list) else 1

    def addElement(self, pos, value):
        self.matrix[pos[0]][pos[1]] += value

    def add(self, matrix2):
        if self.getColSize() == matrix2.getColSize() and self.getRowSize() == matrix2.getRowSize():
            for r in range(len(self.matrix)):
                for c in range(len(self.matrix[r])):
                    self.addElement([r, c], matrix2.at([r, c]))

    def subtract(self, matrix2):
        if self.getColSize() == matrix2.getColSize() and self.getRowSize() == matrix2.getRowSize():
            for r in range(len(self.matrix)):
                for c in range(len(self.matrix[r])):
                    self.addElement([r, c], -matrix2.at([r, c]))

    def vectorVectorMult(self, vector2):
        if isinstance(vector2, list):
            vector2 = Matrix(vector2)
        if self.getRowSize() == vector2.getRowSize() and self.getColSize() == 1 and vector2.getColSize() == 1:
            print("element wise vector vector mult")
            sum = 0
            for r in range(len(self.matrix)):
                sum += self.at([r, 0]) * vector2.at([r, 0])
            return sum
        elif self.getRowSize() == vector2.getColSize() and self.getColSize() == 1 and vector2.getRowSize() == 1:
            print("vector * vector transpose")
            mat = [[(self.at([r, 0]) * vector2.at([0, c])) for c in range(self.getRowSize())] for r in
                   range(vector2.getColSize())]
            return mat
        else:
            print("bruh enter the right shit into vector vector mult")

    def dotProduct(self, matrix2):
        if not isinstance(matrix2, Matrix):
            matrix2 = Matrix(matrix2)
        # take a layer
        dVector = []
        if matrix2.getColSize() > 1:
            for r in matrix2.matrix:
                dVector.append(self.vectorVectorMult(transpose(r)))
        elif self.getColSize() > 1:
            for r in self.matrix:
                dVector.append(matrix2.vectorVectorMult(transpose(r)))
        return dVector

    def multiply(self, matrix2):
        if not isinstance(matrix2, Matrix):
            matrix2 = Matrix(matrix2)
        if self.getColSize() == matrix2.getRowSize():
            colSizeN = matrix2.getColSize()
            rowSizeN = self.getRowSize()
            matrix = [[0 for x in range(colSizeN)] for y in range(rowSizeN)]
            for r in range(rowSizeN):
                for c in range(colSizeN):
                    # add together the multiplications of the col and row
                    sum = 0
                    for k in range(len(matrix2.getCol(c))):
                        sum += matrix2.getCol(c)[k][0] * self.getRow(r)[k]
                    matrix[r][c] = sum
            self.setMatrix(matrix)

        elif self.getRowSize() == matrix2.getColSize():
            colSizeN = self.getColSize()
            rowSizeN = matrix2.getRowSize()
            matrix = [[0 for x in range(colSizeN)] for y in range(rowSizeN)]
            for r in range(rowSizeN):
                for c in range(colSizeN):
                    # add together the multiplications of the col and row
                    sum = 0
                    for k in range(len(matrix2.getRow(r))):
                        sum += matrix2.getRow(r)[k] * self.getCol(c)[k][0]
                    matrix[r][c] = sum
            self.setMatrix(matrix)

        else:
            print("bruh put the right shit into multiply")

    def transpose(self):
        matrixN = [[0 for x in range(self.getRowSize())] for y in range(self.getColSize())]
        for r in range(len(self.matrix)):
            for c in range(len(self.matrix[r])):
                matrixN[c][r] = self.at([r, c])
        self.setMatrix(matrixN)
