package br.com.kflow.activations

import br.com.kflow.computerGraph.*
import br.com.kflow.value.Value

fun relu(variable: Node<Number>): Node<Number> {
    return Max(Value(0.0), variable)
}