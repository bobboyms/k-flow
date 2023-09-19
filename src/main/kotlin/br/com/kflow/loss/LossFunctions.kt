package br.com.kflow.loss

import br.com.kflow.computerGraph.*
import br.com.kflow.value.Value

fun quadratic(actual: Node<Number>, predicted: Node<Number>): Node<Number> {
    return Pow((actual - predicted), Value(2.0))
}

fun MSE(actual: Node<Number>, predicted: Node<Number>): Node<Number> {
    return Sum(quadratic(actual,predicted)) / Tensor(actual.value().size().toFloat())
}