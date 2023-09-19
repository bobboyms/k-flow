package br.com.kflow.loss

import br.com.kflow.computerGraph.*
import br.com.kflow.value.Value

fun quadratic(target: Node<Number>, predicted: Node<Number>): Node<Number> {
    return Pow((predicted - target), Value(2.0))
}

fun MSE(target: Node<Number>, predicted: Node<Number>): Node<Number> {
    return Sum(quadratic(target,predicted)) / Tensor(target.value().size().toFloat())
}