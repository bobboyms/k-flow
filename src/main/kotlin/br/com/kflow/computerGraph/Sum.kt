package br.com.kflow.computerGraph

import br.com.kflow.linear.toT
import br.com.kflow.value.Value

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

        val consA:Value<Number> = if (nodeA.isConstant()) {
            Value(values = List(nodeA.value().size()) { 0.0 as Number }, shape = nodeA.value().shape())
        } else {
            Value(values = List(nodeA.value().size()) { 1.0 as Number }, shape = nodeA.value().shape())
        }

        nodeA.backward(consA * gradient)
    }

    // Uma função para calcular a variância de uma lista de Double
    fun List<Double>.variance(): Double {
        val mean = this.sum() / this.size
        return this.map { it - mean }.map { it * it }.sum() / this.size
    }

}