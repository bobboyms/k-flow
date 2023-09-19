package br.com.kflow.computerGraph

import br.com.kflow.value.Value

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
