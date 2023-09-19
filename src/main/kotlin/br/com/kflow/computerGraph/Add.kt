package br.com.kflow.computerGraph

import br.com.kflow.value.Value

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