package br.com.kflow.computerGraph

import br.com.kflow.value.Value

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
