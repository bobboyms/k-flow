package br.com.kflow.nn

import br.com.kflow.computerGraph.Matmul
import br.com.kflow.computerGraph.Node
import br.com.kflow.computerGraph.Tensor
import br.com.kflow.computerGraph.plus
import br.com.kflow.value.Value
import br.com.kflow.math.randomNormal
import java.util.*

class Dense(input: Int, neurons: Int) {

    private var w = Tensor(values = randomNormal(neurons*input, seed = 42).toList(), shape = arrayOf(neurons, input), name = "w", requiresGrad = true)
    private var b = Tensor(values = randomNormal(neurons, seed = 42).toList(), shape = arrayOf(1, neurons), name = "b", requiresGrad = true)

    fun forward(x: Node<Number>, activation: (variable: Node<Number>) -> Node<Number>): Node<Number> {
        return activation(Matmul(x,w.transpose()) + b)
    }

    fun forward(x: Node<Number>): Node<Number> {
        return Matmul(x,w.transpose() + b)
    }

    fun w(): Node<Number> {
        return w
    }

    fun b(): Node<Number> {
        return b
    }

    fun changeW(values: Value<Number>){
        w.changeValue(values)
    }

    fun changeB(values: Value<Number>){
        b.changeValue(values)
    }


}