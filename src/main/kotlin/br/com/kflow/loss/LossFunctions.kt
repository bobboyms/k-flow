package br.com.kflow.loss

import br.com.kflow.linear.*
import br.com.kflow.value.Value

fun quadratic(actual:Node<Number>, predicted:Node<Number>):Node<Number> {
    return Pow((actual - predicted), Value(2))
}

fun MSE(actual:Node<Number>, predicted:Node<Number>):Node<Number> {
    return Sum(quadratic(actual,predicted)) / Tensor(actual.value().size().toFloat())
}