package br.com.kflow.activations

import br.com.kflow.computerGraph.Tensor
import br.com.kflow.computerGraph.Node
import br.com.kflow.computerGraph.Exp
import br.com.kflow.computerGraph.div
import br.com.kflow.computerGraph.plus
import br.com.kflow.computerGraph.unaryMinus

fun sigmoid(variable: Node<Number>): Node<Number> {
    return Tensor(1) / Tensor(1) + Exp(-variable)
}