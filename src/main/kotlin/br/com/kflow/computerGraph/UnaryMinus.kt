package br.com.kflow.computerGraph

import br.com.kflow.value.Value

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
