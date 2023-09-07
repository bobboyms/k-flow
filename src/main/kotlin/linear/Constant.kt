package linear

interface Matrix {
    fun transpose(): Matrix
    fun shape():Array<Int>
    fun dot(other: Matrix): Matrix
    fun getValue(vararg indices: Int): Number
    fun getColumn(colIndex: Int): List<Number>
    fun pow(power: Matrix): Matrix
    fun size():Int
}

class Constant<T : Number>(
    private val values: List<T>,
    private val shape: Array<Int>
) : Matrix {

    init {
        val totalSize = shape.reduce { acc, i -> acc * i }
        if (totalSize != values.size) {
            throw IllegalArgumentException("Shape and values size are not compatible.")
        }
    }

    constructor(value: T) : this(listOf(value), arrayOf(1, 1))



    override fun dot(other: Matrix): Constant<T> {

        val shapeA = this.shape
        val shapeB = other.shape()

        // Check if the matrices are compatible for dot product
        if (shapeA.last() != shapeB.first()) {
            throw IllegalArgumentException("The last dimension of the first matrix must be equal to the first dimension of the second matrix.")
        }

        val commonDim = shapeA.last()

        val newShape = shapeA.dropLast(1) + shapeB.drop(1)

        val newValues = MutableList(newShape.reduce { acc, i -> acc * i }) { 0.0 }

        when {
            // Both matrices are 2D
            shapeA.size == 2 && shapeB.size == 2 -> {
                val newRows = shapeA[0]
                val newCols = shapeB[1]

                for (i in 0..<newRows) {
                    for (j in 0..<newCols) {
                        var sum = 0.0
                        for (k in 0..<commonDim) {
                            val aValue = this.getValue(i, k).toDouble()
                            val bValue = other.getValue(k, j).toDouble()
                            sum += aValue * bValue
                        }
                        newValues[i * newCols + j] = sum
                    }
                }
            }

            // Both matrices are 3D
            shapeA.size == 3 && shapeB.size == 3 -> {
                val newDepth = shapeA[0]
                val newRows = shapeA[1]
                val newCols = shapeB[2]

                for (d in 0..<newDepth) {
                    for (i in 0..<newRows) {
                        for (j in 0..<newCols) {
                            var sum = 0.0
                            for (k in 0..<commonDim) {
                                val aValue = this.getValue(d, i, k).toDouble()
                                val bValue = other.getValue(d, k, j).toDouble()
                                sum += aValue * bValue
                            }
                            newValues[d * newRows * newCols + i * newCols + j] = sum
                        }
                    }
                }
            }

            else -> {
                throw IllegalArgumentException("Unsupported dimensions for dot product.")
            }
        }

        val values = newValues.map {
            it.toT(values[0]::class.java)
        }

        return Constant(values, newShape.toTypedArray())
    }


    operator fun plus(other: Constant<T>): Constant<T> {
        return plus(other, this)
    }

    operator fun minus(other: Constant<T>): Constant<T> {
        return sub(other, this)
    }

    operator fun times(other: Constant<T>): Constant<T> {
        return mul(other, this)
    }

    operator fun div(other: Constant<T>): Constant<T> {
        return div(other, this)
    }

    fun values(): List<T> {
        return values
    }

    override fun transpose(): Constant<T> {

        return transpose(this)

    }

    override fun shape(): Array<Int> {
        return shape
    }

    override fun getColumn(colIndex: Int): List<Number> {
        val numRows = shape[0]
        return List(numRows) { rowIdx ->
            this.getValue(rowIdx, colIndex)
        }
    }

    override fun pow(power: Matrix): Constant<T> {
        return pow(this,power as Constant<T>)
    }

    override fun size(): Int {
        return values.size
    }


    override fun getValue(vararg indices: Int): T {
        var index = 0
        var size = 1
        for (i in shape.indices.reversed()) {
            index += indices[i] * size
            size *= shape[i]
        }
        return values[index]
    }

    fun printShape() {
        println("Shape: ${shape.joinToString(", ")}")
    }

    fun printMatrix() {
        when (shape.size) {
            2 -> {
                val rows = shape[0]
                val cols = shape[1]

                print("[")
                for (i in 0..<rows) {
                    if (i > 0) print(",\n ")
                    print("[")
                    for (j in 0..<cols) {
                        print("${getValue(i, j)}")
                        if (j < cols - 1) {
                            print(", ")
                        }
                    }
                    print("]")
                }
                print("]\n")
            }
            3 -> {
                val depth = shape[0]
                val rows = shape[1]
                val cols = shape[2]

                print("[")
                for (d in 0..<depth) {
                    if (d > 0) print(",\n")
                    print("[")
                    for (i in 0..<rows) {
                        if (i > 0) print(",\n ")
                        print("[")
                        for (j in 0..<cols) {
                            print("${getValue(d, i, j)}")
                            if (j < cols - 1) {
                                print(", ")
                            }
                        }
                        print("]")
                    }
                    print("]")
                }
                print("]\n")
            }
            else -> {
                println("Cannot print matrix with shape ${shape.joinToString(", ")}.")
            }
        }
    }

}
