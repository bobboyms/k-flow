package br.com.kflow.linear

import kotlin.math.exp
import kotlin.math.pow

fun <T:Number> dot(other: Constant<T>, atual: Constant<T>): Constant<T> {

    if (!atual.shape().contentEquals(other.shape())) {
        throw IllegalArgumentException("Shapes must be the same for dot product.")
    }

    val newValues = atual.values().mapIndexed { index, value ->
        (value.toDouble() * other.values()[index].toDouble()).toT(value::class.java)
    }

    return Constant(newValues, atual.shape())
}

fun <T:Number> transpose(constant: Constant<T>): Constant<T> {

    val shape = constant.shape()
    val values = constant.values()

    if (shape.size == 2) {
        val transposedValues = MutableList(values.size) { 0 as Any } as MutableList<T>
        val rows = shape[0]
        val cols = shape[1]

        for (i in 0..<rows) {
            for (j in 0..<cols) {
                transposedValues[j * rows + i] = values[i * cols + j]
            }
        }

        return Constant(transposedValues, arrayOf(shape[1], shape[0]))

    } else if (shape.size == 3) {
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

        return Constant(transposedValues, arrayOf(d3, d2, d1))
    }

    return constant
}



fun <T:Number> pow(atual: Constant<T>, power: Constant<T>): Constant<T> {
    if (atual.shape()[0] == 1 && atual.shape()[1] == 1) {
        val result = atual.values()[0].toDouble().pow(power.values()[0].toDouble())
            .toT(atual.values()[0]::class.java)
        return Constant(values = listOf(result), shape = atual.shape())
    } else if (power.shape()[0] == 1 && power.shape()[1] == 1) {
        val newValues = atual.values().map {
            it.toDouble().pow(power.values()[0].toDouble()).toT(it::class.java)
        }
        return Constant(newValues, atual.shape())
    } else {
        if (!atual.shape().contentEquals(power.shape())) {
            throw IllegalArgumentException("Shapes must be the same for addition")
        }

        val newValues = atual.values().mapIndexed { index, value ->
            value.toDouble().pow(power.values()[index].toDouble()).toT(value::class.java)
        }

        return Constant(newValues, atual.shape())
    }

}

fun <T:Number> plus(other: Constant<T>, atual: Constant<T>): Constant<T> {
    if (atual.shape()[0] == 1 && atual.shape()[1] == 1) {
        val newValues = other.values().map {
            (it.toDouble() + atual.values()[0].toDouble()).toT(it::class.java)
        }
        return Constant(newValues, other.shape())
    } else if (other.shape()[0] == 1 && other.shape()[1] == 1) {
        val newValues = atual.values().map {
            (it.toDouble() + other.values()[0].toDouble()).toT(it::class.java)
        }
        return Constant(newValues, atual.shape())
    } else {
        if (!atual.shape().contentEquals(other.shape())) {
            throw IllegalArgumentException("Shapes must be the same for addition")
        }

        val newValues = atual.values().mapIndexed { index, value ->
            (value.toDouble() + other.values()[index].toDouble()).toT(value::class.java)
        }

        return Constant(newValues, atual.shape())
    }
}

fun <T:Number> sub(other: Constant<T>, atual: Constant<T>): Constant<T> {
    if (atual.shape()[0] == 1 && atual.shape()[1] == 1) {
        val newValues = other.values().map {
            (atual.values()[0].toDouble() - it.toDouble()).toT(it::class.java)
        }
        return Constant(newValues, other.shape())
    } else if (other.shape()[0] == 1 && other.shape()[1] == 1) {
        val newValues = atual.values().map {
            (it.toDouble() - other.values()[0].toDouble()).toT(it::class.java)
        }
        return Constant(newValues, atual.shape())
    } else {
        if (!atual.shape().contentEquals(other.shape())) {
            throw IllegalArgumentException("Shapes must be the same for addition")
        }

        val newValues = atual.values().mapIndexed { index, value ->
            (value.toDouble() - other.values()[index].toDouble()).toT(value::class.java)
        }

        return Constant(newValues, atual.shape())
    }
}

fun <T:Number> mul(other: Constant<T>, atual: Constant<T>): Constant<T> {

    if (atual.shape()[0] == 1 && atual.shape()[1] == 1) {
        val newValues = other.values().map {
            (it.toDouble() * atual.values()[0].toDouble()).toT(it::class.java)
        }
        return Constant(newValues, other.shape())
    } else if (other.shape()[0] == 1 && other.shape()[1] == 1) {
        val newValues = atual.values().map {
            (it.toDouble() * other.values()[0].toDouble()).toT(it::class.java)
        }
        return Constant(newValues, atual.shape())
    } else {
        if (!atual.shape().contentEquals(other.shape())) {
            throw IllegalArgumentException("Shapes must be the same for addition")
        }

        val newValues = atual.values().mapIndexed { index, value ->
            (value.toDouble() * other.values()[index].toDouble()).toT(value::class.java)
        }

        return Constant(newValues, atual.shape())
    }
}

fun <T:Number> exp(atual: Constant<T>): Constant<T> {
    val values = atual.values().map {
        exp(it.toDouble()).toT(atual.values()[0]::class.java)
    }
    return Constant(values = values, shape = atual.shape())
}

fun <T:Number> div(other: Constant<T>, atual: Constant<T>): Constant<T> {

    if (atual.shape()[0] == 1 && atual.shape()[1] == 1) {
        val newValues = other.values().map {
            (it.toDouble() / atual.values()[0].toDouble()).toT(it::class.java)
        }
        return Constant(newValues, other.shape())
    } else if (other.shape()[0] == 1 && other.shape()[1] == 1) {
        val newValues = atual.values().map {
            (it.toDouble() / other.values()[0].toDouble()).toT(it::class.java)
        }
        return Constant(newValues, atual.shape())
    } else {
        if (!atual.shape().contentEquals(other.shape())) {
            throw IllegalArgumentException("Shapes must be the same for addition")
        }

        val newValues = atual.values().mapIndexed { index, value ->
            (value.toDouble() / other.values()[index].toDouble()).toT(value::class.java)
        }

        return Constant(newValues, atual.shape())
    }
}