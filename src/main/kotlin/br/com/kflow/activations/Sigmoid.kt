package br.com.kflow.activations

import br.com.kflow.linear.*

fun sigmoid(variable: Node<Number>):Node<Number> {
    return Variable(1) / Variable(1) + Exp(-variable)
}