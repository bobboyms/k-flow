package br.com.kflow.computerGraph

import br.com.kflow.value.Value
import br.com.kflow.linear.max

class Max(private val maxValue: Value<Number>, private val nodeA: Node<Number>) : Node<Number>() {

    override fun value(): Value<Number> {
        if (value == null) {
            value = max(maxValue,nodeA.value())
        }
        return value!!
    }

    override fun zeroGrad() {
        super.zeroGrad()
        nodeA.zeroGrad()
    }

    override fun backward(gradient: Value<Number>) {
        this.gradient += gradient
        val tempValues = value!!.values().map {
            if (it.toDouble() > 0.0) {
                1.0 as Number
            } else {
                0.0 as Number
            }
        }
        nodeA.backward(Value(values = tempValues, shape = value!!.shape()) * gradient)
    }
}
