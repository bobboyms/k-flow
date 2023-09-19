package br.com.kflow.computerGraph

import br.com.kflow.value.Value

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