package br.com.kflow.linear

import br.com.kflow.value.DNarray
import br.com.kflow.value.Value
import kotlin.math.exp
import kotlin.math.pow

fun <T:Number> dot(other: Value<T>, atual: Value<T>): Value<T> {

    if (!atual.shape().contentEquals(other.shape())) {
        throw IllegalArgumentException("Shapes must be the same for dot product.")
    }

    val newValues = atual.values().mapIndexed { index, value ->
        (value.toDouble() * other.values()[index].toDouble()).toT(value::class.java)
    }

    return Value(newValues, atual.shape())
}

fun <T:Number> transpose2d(value: Value<T>): Value<T> {

    val shape = value.shape()
    val values = value.values()

    val transposedValues = MutableList(values.size) { 0 as Any } as MutableList<T>
    val rows = shape[0]
    val cols = shape[1]

    for (i in 0..<rows) {
        for (j in 0..<cols) {
            transposedValues[j * rows + i] = values[i * cols + j]
        }
    }

    return Value(transposedValues, arrayOf(shape[1], shape[0]))
}

fun <T:Number> transpose3d(value: Value<T>): Value<T> {
    val shape = value.shape()
    val values = value.values()

    val d1 = shape[0]
    val d2 = shape[1]
    val d3 = shape[2]

    val transposedValues = MutableList(values.size) { 0 as Any } as MutableList<T>

    for (i in 0..<d1) {
        for (j in 0..<d2) {
            for (k in 0..<d3) {
                val oldIndex = i * d2 * d3 + j * d3 + k
                val newIndex = k * d2 * d1 + j * d1 + i
                transposedValues[newIndex] = values[oldIndex]
            }
        }
    }

    return Value(transposedValues, arrayOf(d3, d2, d1))
}

fun <T:Number> transposeLast2Dims(value: Value<T>): Value<T> {
    val shape = value.shape()
    val values = value.values()

    val d1 = shape[0]
    val d2 = shape[1]
    val d3 = shape[2]

    val transposedValues = MutableList(values.size) { 0 as Any } as MutableList<T>

    for (i in 0..<d1) {
        for (j in 0..<d2) {
            for (k in 0..<d3) {
                val oldIndex = i * d2 * d3 + j * d3 + k
                val newIndex = i * d2 * d3 + k * d2 + j
                transposedValues[newIndex] = values[oldIndex]
            }
        }
    }

    return Value(transposedValues, arrayOf(d1, d3, d2))
}



fun <T:Number> pow(atual: Value<T>, power: Value<T>): Value<T> {
    if (atual.shape()[0] == 1 && atual.shape()[1] == 1) {
        val result = atual.values()[0].toDouble().pow(power.values()[0].toDouble())
            .toT(atual.values()[0]::class.java)
        return Value(values = listOf(result), shape = atual.shape())
    } else if (power.shape()[0] == 1 && power.shape()[1] == 1) {
        val newValues = atual.values().map {
            it.toDouble().pow(power.values()[0].toDouble()).toT(it::class.java)
        }
        return Value(newValues, atual.shape())
    } else {
        if (!atual.shape().contentEquals(power.shape())) {
            throw IllegalArgumentException("Shapes must be the same for addition")
        }

        val newValues = atual.values().mapIndexed { index, value ->
            value.toDouble().pow(power.values()[index].toDouble()).toT(value::class.java)
        }

        return Value(newValues, atual.shape())
    }

}
fun <T:Number> unaryMinus(atual: Value<T>): Value<T> {
    val newValues = atual.values().map {
        (-1 * it.toDouble()).toT(it::class.java)
    }
    return Value(newValues, atual.shape())
}

fun <T:Number> plus(other: Value<T>, atual: Value<T>): Value<T> {
    if (atual.shape()[0] == 1 && atual.shape()[1] == 1) {
        val newValues = other.values().map {
            (it.toDouble() + atual.values()[0].toDouble()).toT(it::class.java)
        }
        return Value(newValues, other.shape())
    } else if (other.shape()[0] == 1 && other.shape()[1] == 1) {
        val newValues = atual.values().map {
            (it.toDouble() + other.values()[0].toDouble()).toT(it::class.java)
        }
        return Value(newValues, atual.shape())
    } else {
        if (!atual.shape().contentEquals(other.shape())) {
            throw IllegalArgumentException("Shapes must be the same for addition")
        }

        val newValues = atual.values().mapIndexed { index, value ->
            (value.toDouble() + other.values()[index].toDouble()).toT(value::class.java)
        }

        return Value(newValues, atual.shape())
    }
}

fun <T:Number> sub(other: Value<T>, atual: Value<T>): Value<T> {
    if (atual.shape()[0] == 1 && atual.shape()[1] == 1) {
        val newValues = other.values().map {
            (atual.values()[0].toDouble() - it.toDouble()).toT(it::class.java)
        }
        return Value(newValues, other.shape())
    } else if (other.shape()[0] == 1 && other.shape()[1] == 1) {
        val newValues = atual.values().map {
            (it.toDouble() - other.values()[0].toDouble()).toT(it::class.java)
        }
        return Value(newValues, atual.shape())
    } else {
        if (!atual.shape().contentEquals(other.shape())) {
            throw IllegalArgumentException("Shapes must be the same for addition")
        }

        val newValues = atual.values().mapIndexed { index, value ->
            (value.toDouble() - other.values()[index].toDouble()).toT(value::class.java)
        }

        return Value(newValues, atual.shape())
    }
}

fun <T:Number> mul(other: Value<T>, atual: Value<T>): Value<T> {

    if (atual.shape()[0] == 1 && atual.shape()[1] == 1) {
        val newValues = other.values().map {
            (it.toDouble() * atual.values()[0].toDouble()).toT(it::class.java)
        }
        return Value(newValues, other.shape())
    } else if (other.shape()[0] == 1 && other.shape()[1] == 1) {
        val newValues = atual.values().map {
            (it.toDouble() * other.values()[0].toDouble()).toT(it::class.java)
        }
        return Value(newValues, atual.shape())
    } else {
        if (!atual.shape().contentEquals(other.shape())) {
            throw IllegalArgumentException("Shapes must be the same for addition")
        }

        val newValues = atual.values().mapIndexed { index, value ->
            (value.toDouble() * other.values()[index].toDouble()).toT(value::class.java)
        }

        return Value(newValues, atual.shape())
    }
}

fun <T:Number> exp(atual: Value<T>): Value<T> {
    val values = atual.values().map {
        exp(it.toDouble()).toT(atual.values()[0]::class.java)
    }
    return Value(values = values, shape = atual.shape())
}

fun <T:Number> div(other: Value<T>, atual: Value<T>): Value<T> {

    if (atual.shape()[0] == 1 && atual.shape()[1] == 1) {
        val newValues = other.values().map {
            (atual.values()[0].toDouble() / it.toDouble()).toT(it::class.java)
        }
        return Value(newValues, other.shape())
    } else if (other.shape()[0] == 1 && other.shape()[1] == 1) {
        val newValues = atual.values().map {
            (it.toDouble() / other.values()[0].toDouble()).toT(it::class.java)
        }
        return Value(newValues, atual.shape())
    } else {
        if (!atual.shape().contentEquals(other.shape())) {
            throw IllegalArgumentException("Shapes must be the same for addition")
        }

        val newValues = atual.values().mapIndexed { index, value ->
            (value.toDouble() / other.values()[index].toDouble()).toT(value::class.java)
        }

        return Value(newValues, atual.shape())
    }
}

fun dot2d(local: DNarray, other: DNarray): Value<Number> {

    val shapeA = local.shape()
    val shapeB = other.shape()

    // Verifica se as matrizes são compatíveis para o produto escalar
    if (shapeA.last() != shapeB.first()) {
        throw IllegalArgumentException("The last dimension of the first matrix must be equal to the first dimension of the second matrix.")
    }

    val newRows = shapeA[0]
    val newCols = shapeB[1]
    val commonDim = shapeA.last()

    // Lista para armazenar os valores da nova matriz
    val newValues = MutableList(newRows * newCols) { 0.0 }

    for (i in 0..<newRows) {
        for (j in 0..<newCols) {
            var sum = 0.0
            for (k in 0..<commonDim) {
                val aValue = (local as Value<*>).getValue(i, k).toDouble()
                val bValue = (other as Value<*>).getValue(k, j).toDouble()
                sum += aValue * bValue
            }
            newValues[i * newCols + j] = sum
        }
    }

    // Converte os valores para o tipo de número correto
    val values = newValues.map {
        it.toT((local as Value<*>).values()[0]::class.java)
    }

    return Value(values, arrayOf(newRows, newCols))
}

fun <T : Number> batchedMatmul(local: Value<T>, other: Value<T>): Value<T> {
    val batchSize = local.shape()[0]
    val rowsA = local.shape()[1]
    val colsB = other.shape()[2]

    if (batchSize != other.shape()[0]) {
        throw IllegalArgumentException("The batch size is different between the 2 NDarray")
    }

    val cValues = MutableList(batchSize * rowsA * colsB) { 0.0 as T }

    for (slice in 0..<batchSize) {
        val x = local.get2DSlice(slice)
        val y = other.get2DSlice(slice)
        val c = x.matmul(y)

        for (i in 0..<rowsA) {
            for (j in 0..<colsB) {
                val value = c.getValue(i, j)
                val index = slice * rowsA * colsB + i * colsB + j
                cValues[index] = value
            }
        }
    }

    return Value(values = cValues, shape = arrayOf(batchSize, rowsA, colsB))
}