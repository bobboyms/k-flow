package br.com.kflow.computerGraph

import br.com.kflow.value.Value
import java.lang.RuntimeException

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

    private fun backward2D() {
        if (nodeA.transposed()) {
            nodeA.backward(gradient.matmul(nodeB.value().transpose()).transpose())
        } else {
            nodeA.backward(gradient.matmul(nodeB.value().transpose()))
        }

        if (nodeB.transposed()) {
            val temp = nodeA.value().transpose().matmul(gradient)
            nodeB.backward(temp.transpose())
        } else {
            nodeB.backward(nodeA.value().transpose().matmul(gradient))
        }
    }

    private fun backward3D() {

        if (nodeA.transposed()) {
            nodeA.backward(gradient.matmul(nodeB.value().transpose()))
        } else {
            if (nodeB.transposed()) {
                nodeA.backward(gradient.matmul(nodeB.value().transpose()))
            } else {
                nodeA.backward(gradient.matmul(nodeB.value()))
            }
        }

        if (nodeB.transposed()) {
            val temp = nodeA.value().transpose().matmul(gradient)
            nodeB.backward(temp.transpose())
        } else {
            nodeB.backward(nodeA.value().transpose().matmul(gradient))
        }

    }

    override fun backward(gradient: Value<Number>) {
        this.gradient += gradient

        if (gradient.shape().size == 2) {
            backward2D()
        } else if (gradient.shape().size == 3) {
            backward3D()
        } else {
            throw RuntimeException("Operation not supported for that NDarray shape")
        }


    }

}
