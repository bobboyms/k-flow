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

fun dot3d(local: DNarray, other: DNarray): Value<Number> {
    val shapeA = local.shape()
    val shapeB = other.shape()

    if (shapeA.last() != shapeB.first()) {
        throw IllegalArgumentException("The last dimension of the first tensor must be equal to the first dimension of the second tensor.")
    }

    val newDepth = if (shapeA.size > 2) shapeA[0] else 1
    val newRows = shapeA[shapeA.size - 2]
    val newCols = shapeB[shapeB.size - 1]
    val commonDim = shapeA.last()

    val newValues = MutableList(newDepth * newRows * newCols) { 0.0 }

    for (d in 0..<newDepth) {
        for (i in 0..<newRows) {
            for (j in 0..<newCols) {
                var sum = 0.0
                for (k in 0..<commonDim) {
                    val aValue = if (shapeA.size > 2) {
                        (local as Value<*>).getValue(d, i, k).toDouble()
                    } else {
                        (local as Value<*>).getValue(i, k).toDouble()
                    }

                    val bValue = if (shapeB.size > 2) {
                        (other as Value<*>).getValue(d, k, j).toDouble()
                    } else {
                        (other as Value<*>).getValue(k, j).toDouble()
                    }

                    sum += aValue * bValue
                }
                newValues[d * newRows * newCols + i * newCols + j] = sum
            }
        }
    }

    val values = newValues.map {
        it.toT((local as Value<*>).values()[0]::class.java)
    }

    return Value(values, arrayOf(newDepth, newRows, newCols))
}

//fun dotBatched(local: DNarray, other: DNarray): Value<Number> {
//    val shapeA = local.shape()
//    val shapeB = other.shape()
//
//    if (shapeA.size < 3 || shapeB.size < 3) {
//        throw IllegalArgumentException("Input tensors must have at least 3 dimensions for batched matrix multiplication.")
//    }
//
//    if (shapeA[0] != shapeB[0]) {
//        throw IllegalArgumentException("Batch sizes must match.")
//    }
//
//    if (shapeA.last() != shapeB[shapeB.size - 2]) {
//        throw IllegalArgumentException("The last dimension of the first tensor must be equal to the second to last dimension of the second tensor.")
//    }
//
//    val batchSize = shapeA[0]
//    val newRows = shapeA[shapeA.size - 2]
//    val newCols = shapeB[shapeB.size - 1]
//    val commonDim = shapeA.last()
//
//    val newValues = MutableList(batchSize * newRows * newCols) { 0.0 }
//
//    for (n in 0..<batchSize) {
//        for (i in 0..<newRows) {
//            for (j in 0..<newCols) {
//                var sum = 0.0
//                for (k in 0..<commonDim) {
//                    val aValue = (local as Value<*>).getValue(n, i, k).toDouble()
//                    val bValue = (other as Value<*>).getValue(n, k, j).toDouble()
//                    sum += aValue * bValue
//                }
//                newValues[n * newRows * newCols + i * newCols + j] = sum
//            }
//        }
//    }
//
//    val values = newValues.map {
//        it.toT((local as Value<*>).values()[0]::class.java)
//    }
//
//    return Value(values, arrayOf(batchSize, newRows, newCols))
//}

// Sua classe Value<T> continua a mesma

// Sua função dot agora usa dotBatched para tensores com mais de duas dimensões
fun dotBatched(local: DNarray, other: DNarray): Value<Number> {
    val shapeA = local.shape()
    val shapeB = other.shape()

//    if (shapeA[0] != shapeB[0] || shapeA[2] != shapeB[2]) {
//        throw IllegalArgumentException("Incompatible shapes.")
//    }

    val batchSize = shapeA[0]
    val newRows = shapeA[1]
    val newCols = shapeB[1]
    val commonDim = shapeA[2]

    val newValues = MutableList(batchSize * newRows * newCols) { 0.0 }

    for (n in 0 until batchSize) {
        for (i in 0 until newRows) {
            for (j in 0 until newCols) {
                var sum = 0.0
                for (k in 0 until commonDim) {
                    val aValue = (local as Value<*>).getValue(n, i, k).toDouble()
                    val bValue = (other as Value<*>).getValue(n, j, k).toDouble()
                    sum += aValue * bValue
                }
                newValues[(n * newRows * newCols) + (i * newCols) + j] = sum
            }
        }
    }

    return Value(newValues, arrayOf(batchSize, newRows, newCols))
}

fun <T : Number> batchedMatmul(A: Value<T>, B: Value<T>): Value<T> {
    val batch_size = A.shape()[0]
    val rows_A = A.shape()[1]
    val cols_A = A.shape()[2]
    val rows_B = B.shape()[1]
    val cols_B = B.shape()[2]

    if (cols_A != rows_B) throw IllegalArgumentException("The last dimension of A must match the second to last dimension of B")

    val cShape = arrayOf(batch_size, rows_A, cols_B)
    val cValues = MutableList(batch_size * rows_A * cols_B) { 0.0 as T }

    for (batch in 0..<batch_size) {
        for (i in 0..<rows_A) {
            for (j in 0..<cols_B) {
                var value = 0.0
                for (k in 0..<cols_A) {
                    value += A.getValue(batch, i, k).toDouble() * B.getValue(batch, k, j).toDouble()
                }
                val index = batch * rows_A * cols_B + i * cols_B + j
                cValues[index] = value.toT(A.values()[0]::class.java)
            }
        }
    }

    return Value(cValues as List<T>, cShape)
}