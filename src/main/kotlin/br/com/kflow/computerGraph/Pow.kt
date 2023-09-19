package br.com.kflow.computerGraph

import br.com.kflow.value.Value

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
