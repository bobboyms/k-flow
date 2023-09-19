package br.com.kflow.computerGraph

import br.com.kflow.value.Value

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
        println("Gradiente")
        gradient.printMatrix()
        nodeA.value().printMatrix()
        nodeA.backward(nodeB.value() * gradient)  // Derivada parcial em relação a 'a' é 'b'
        nodeB.backward(nodeA.value() * gradient)  // Derivada parcial em relação a 'b' é 'a'
    }
}
