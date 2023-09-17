package br.com.kflow.value

import br.com.kflow.linear.*
import java.lang.RuntimeException

class Value<T : Number>(
    private val values: List<T>,
    private val shape: Array<Int>,
) : DNarray {

    init {
        val totalSize = shape.reduce { acc, i -> acc * i }
        if (totalSize != values.size) {
            throw IllegalArgumentException("Shape and values size are not compatible.")
        }
    }

    constructor(value: T) : this(listOf(value), arrayOf(1, 1))
//    constructor(values: List<T>, shape: Array<Int>) : this(values, shape, false)

    override fun dot(other: DNarray): Value<T> {

        if (this.shape.size == 2) {
            return dot2d(this, other) as Value<T>
        }

        if (this.shape.size == 3) {
            return batchedMatmul(this, other as Value<T>)
        }

        throw RuntimeException("Not suported shape for dot product")
    }

    operator fun unaryMinus(): Value<T> {
        return br.com.kflow.linear.unaryMinus(this)
    }

    operator fun plus(other: Value<T>): Value<T> {
        return br.com.kflow.linear.plus(other, this)
    }

    operator fun minus(other: Value<T>): Value<T> {
        return sub(other, this)
    }

    operator fun times(other: Value<T>): Value<T> {
        return mul(other, this)
    }

    operator fun div(other: Value<T>): Value<T> {
        return div(other as Value<T>, this)
    }

    override fun values(): List<T> {
        return values
    }

    override fun transpose(): Value<T> {

        if (shape.size == 2) {
            return transpose2d(this)
        }

        if (shape.size == 3) {
            return transpose3d(this)
        }

        throw RuntimeException("this operation is not allowed for the shape " + shape.size + "D")

    }

    override fun shape(): Array<Int> {
        return shape
    }

    fun getColumn(colIndex: Int): List<Number> {
        val numRows = shape[0]
        return List(numRows) { rowIdx ->
            this.getValue(rowIdx, colIndex)
        }
    }

    override fun pow(power: DNarray): Value<T> {
        return br.com.kflow.linear.pow(this, power as Value<T>)
    }

    override fun exp(): Value<T> {
        return br.com.kflow.linear.exp(this)
    }

    override fun size(): Int {
        return values.size
    }


    fun getValue(vararg indices: Int): T {
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

    fun get2DSlice(batchIndex: Int): Value<T> {

        if (shape.size != 3) {
            throw RuntimeException("This operation is only allowed for 3D tensors.")
        }

        val rows = shape[1]
        val cols = shape[2]
        val newValues = MutableList(rows * cols) { 0.0 }

        for (i in 0..<rows) {
            for (j in 0..<cols) {
                newValues[i * cols + j] = getValue(batchIndex, i, j).toDouble()
            }
        }

        return Value(newValues as List<T>, arrayOf(rows, cols))
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
