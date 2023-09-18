package br.com.kflow.linear

import br.com.kflow.value.Value
import java.lang.RuntimeException

// Use Node<Number> to specify that these operations apply to Nodes with Number types
class Add(private val nodeA: Node<Number>, private val nodeB: Node<Number>) : Node<Number>() {

    override fun value(): Value<Number> {
        if (value == null) {
            value = nodeA.value() + nodeB.value()
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
        nodeB.zeroGrad()
    }

    override fun backward(gradient: Value<Number>) {
        this.gradient += gradient
        nodeA.backward(gradient)  // Adjust as necessary
        nodeB.backward(gradient)  // Adjust as necessary
    }
}

class Exp(private val nodeA: Node<Number>) : Node<Number>() {

    override fun value(): Value<Number> {
        if (value == null) {
            value = Value(values = nodeA.value().exp().values(), shape = nodeA.value().shape())
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
    }

    override fun backward(gradient: Value<Number>) {
        this.gradient += gradient
        value()
        nodeA.backward(value!! * gradient)
    }

}

class UnaryMinus(private val nodeA: Node<Number>) : Node<Number>() {
    override fun value(): Value<Number> {
        if (value == null) {
            value = -nodeA.value()
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
    }

    override fun backward(gradient: Value<Number>) {
        this.gradient += gradient
        val consA = Value(values = List(nodeA.value().size()) { -1.0 as Number }, shape = nodeA.value().shape())

        nodeA.backward(consA * gradient)
    }
}

class Sum(private val nodeA: Node<Number>) : Node<Number>() {

    override fun value(): Value<Number> {
        if (value == null) {

            val tempValue = nodeA.value().values().reduce { acc, number ->  acc.toDouble() + number.toDouble()}
                .toT(nodeA.value().values()[0]::class.java)
            value = Value(tempValue)
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
    }

    override fun backward(gradient: Value<Number>) {
        this.gradient += gradient
        val consA = Value(values = List(nodeA.value().size()) { 1.0 as Number }, shape = nodeA.value().shape())
        nodeA.backward(consA * gradient)
    }

}

class Sub(private val nodeA: Node<Number>, private val nodeB: Node<Number>) : Node<Number>() {

    override fun value(): Value<Number> {
        if (value == null) {
            value = nodeA.value() - nodeB.value()
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
        nodeB.zeroGrad()
    }

    override fun backward(gradient: Value<Number>) {
        this.gradient += gradient

        val consA = Value(values = List(nodeA.value().size()) { 1.0 as Number }, shape = nodeA.value().shape())
        val consB = Value(values = List(nodeB.value().size()) { -1.0 as Number }, shape = nodeB.value().shape())

        nodeA.backward(consA * gradient)
        nodeB.backward(consB * gradient)
    }
}

class Div(private val nodeA: Node<Number>, private val nodeB: Node<Number>) : Node<Number>() {
    override fun value(): Value<Number> {
        if (value == null) {
            value = nodeA.value() / nodeB.value()
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
        nodeB.zeroGrad()
    }

    override fun backward(gradient: Value<Number>) {
        this.gradient += gradient

        val consA = Value(1.0 as Number) / nodeB.value()
        val consB = -(nodeA.value() / nodeB.value().pow(Value(2)))

        nodeA.backward(consA * gradient)
        nodeB.backward(consB * gradient)
    }
}


class Multiply(private val nodeA: Node<Number>, private val nodeB: Node<Number>) : Node<Number>() {

    override fun value(): Value<Number> {
        if (value == null) {
            value = nodeA.value() * nodeB.value()
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
        nodeB.zeroGrad()
    }

    override fun backward(gradient: Value<Number>) {
        this.gradient += gradient
        nodeA.backward(nodeB.value() * gradient)  // Derivada parcial em relação a 'a' é 'b'
        nodeB.backward(nodeA.value() * gradient)  // Derivada parcial em relação a 'b' é 'a'
    }
}

class Pow(private val nodeA: Node<Number>, private val power: Value<Number>) : Node<Number>() {
    override fun value(): Value<Number> {
        if (value == null) {
            value = nodeA.value().pow(power)
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
    }

    override fun backward(gradient: Value<Number>) {
        this.gradient += gradient

        val diff = power * nodeA.value().pow(power - Value(1))
        nodeA.backward(diff * gradient)
    }
}


class Matmul(private val nodeA: Node<Number>, private val nodeB: Node<Number>) : Node<Number>() {

    override fun value(): Value<Number> {
        if (value == null) {
            value = nodeA.value().matmul(nodeB.value())
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
        nodeB.zeroGrad()
    }

    override fun backward(gradient: Value<Number>) {
        this.gradient += gradient

        if (gradient.shape().size == 2) {
            if (nodeA.transposed()) {
                nodeA.backward(gradient.matmul(nodeB.value().transpose()).transpose())
            } else {
                nodeA.backward(gradient.matmul(nodeB.value().transpose()))
            }

            if (nodeB.transposed()) {
                // Calculate d(A * B^T)/dB = (A^T * gradient)^T
                val temp = nodeA.value().transpose().matmul(gradient)
                nodeB.backward(temp.transpose())
            } else {
                // Calculate d(A * B)/dB = A^T * gradient
                nodeB.backward(nodeA.value().transpose().matmul(gradient))
            }
        } else if (gradient.shape().size == 3) {
            if (nodeA.transposed()) {
                nodeA.backward(gradient.matmul(nodeB.value().transposeLast2Dims()))
            } else {
                if (nodeB.transposed()) {
                    nodeA.backward(gradient.matmul(nodeB.value().transposeLast2Dims()))
                } else {
                    nodeA.backward(gradient.matmul(nodeB.value()))
                }
            }

            if (nodeB.transposed()) {
                val temp = nodeA.value().transposeLast2Dims().matmul(gradient)
                nodeB.backward(temp.transposeLast2Dims())
            } else {
                nodeB.backward(nodeA.value().transposeLast2Dims().matmul(gradient))
            }
        } else {
            throw RuntimeException("Operation not supported for that NDarray shape")
        }


    }

}
