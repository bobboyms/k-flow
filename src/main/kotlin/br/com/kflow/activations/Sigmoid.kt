package br.com.kflow.activations

import br.com.kflow.linear.*

fun sigmoid(variable: Node<Number>):Node<Number> {
    return Tensor(1) / Tensor(1) + Exp(-variable)
}