package br.com.kflow.linear

import br.com.kflow.value.Value
import java.lang.RuntimeException

abstract class Node<T: Number> {
    protected var gradient = Value(0.0 as T)
    abstract fun value(): Value<T>
    abstract fun backward(gradient: Value<T>)

    protected var value: Value<Number>? = null
    private var transposed = false

    fun transposed():Boolean {
        return transposed
    }

    fun grad(): Value<T> {
        return gradient
    }

    fun changeValue(value: Value<Number>) {
        this.value = value
    }

    fun T() : Node<T> {
        if (transposed) {
            return this
        }

        transposed = true
        if (value().shape().size == 2) {
            val t = value().transpose()
            value = Value(t.values(),shape=t.shape())
            return this
        }

        if (value().shape().size == 3) {

            val batch = value().shape()[0]
            if (value().shape()[1] == batch && value().shape()[2] == batch) {
                val t = value().transpose()
                value = Value(t.values(),shape=t.shape())
                return this
            }

            val t = value().transposeLast2Dims()
            value = Value(t.values(),shape=t.shape())
            return this
        }

        throw RuntimeException("Invalid shape for that operation")

    }

    open fun zeroGrad() {
        this.gradient = Value(0.0 as T)
    }

}

// Since Node is a generic class, you have to indicate its type
operator fun Node<Number>.plus(other: Node<Number>): Node<Number> {
    return Add(this, other)
}

operator fun Node<Number>.times(other: Node<Number>): Node<Number> {
    return Multiply(this, other)
}

operator fun Node<Number>.minus(other: Node<Number>): Node<Number> {
    return Sub(this, other)
}

operator fun Node<Number>.div(other: Node<Number>): Node<Number> {
    return Div(this, other)
}
operator fun Node<Number>.unaryMinus():Node<Number> {
    return UnaryMinus(this)
}